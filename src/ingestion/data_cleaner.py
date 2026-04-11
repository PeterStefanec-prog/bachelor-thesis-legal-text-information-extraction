import os # paths
import re   # working with regex - e.g. searching some regex in text
import json
import unicodedata  # normalizing unicode data
from pathlib import Path    # more convenient work with paths for me
import pdfplumber   # extraction text from pdfs


# ==========================================
# FUNCTIONS (mostly from jupyter notebook where i tested them)
# ==========================================

# --- STEP 3.2: Light unicode ---
def clean_light(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize('NFC', text)   # for saving data in compact form -  for example letter with diacritic
    text = text.replace('\u200b', '').replace('\u00ad', '') # \u200b = zero-width space
    # \u00ad = soft hyphen      - those chars are often created in pdfs and makes mess
    text = text.replace('\r\n', '\n').replace('\r', '\n') # uniting new lines - for example in windows /r/n is used
    return text


# --- STEP 3.3: Metadata extraction ---
def extract_meta(header_text: str):
    meta = {}
    date_match = re.search(r"(\d{1,2}\.\s*\d{1,2}\.\s*\d{4})", header_text)
    if date_match:
        meta['date_raw'] = date_match.group(1).replace(" ", "")
    sz_match = re.search(r"Spisová\s*značka:?\s*([\w\d/]+)", header_text, re.IGNORECASE)
    if sz_match:
        meta['case_id'] = sz_match.group(1)
    ecli_match = re.search(r"(ECLI:[A-Z0-9:.]+)", header_text)
    if ecli_match:
        meta['ecli'] = ecli_match.group(1)
    # FIX: Added court and document_type extraction
    court_match = re.search(r"Súd:\s*(.+)", header_text)
    if court_match:
        meta['court'] = court_match.group(1).strip()
    doctype_match = re.search(r"\b(ROZSUDOK|UZNESENIE)\b", header_text, re.IGNORECASE)
    if doctype_match:
        meta['document_type'] = doctype_match.group(1).capitalize()
    return meta


# --- STEP 3.4: Remove pagination & separate main header ---
def remove_pagination(text: str) -> str:
    lines = text.split('\n')        # divides text into lines
    out = []
    for line in lines:
        s = line.strip()    # spaces at beggining and end removal
        # Remove "Strana X / X", requires dashes for bare numbers (- 3 -)
        if re.match(r"^strana\s+\d+(?:\s*z\s*\d+)?$", s, re.IGNORECASE):    # (?: ... )? non mandatory part
            continue
        # FIX: Now requires actual dashes, so lone '3' or '12' won't get removed
        if re.match(r"^-\s*\d+\s*-$", s):
            continue
        # FIX: Added Pokračovanie as specified in thesis spec
        if re.match(r"^pokračovanie\s*\.{0,3}$", s, re.IGNORECASE):
            continue
        out.append(line) # if not deleted (by any of if) add to output
    return "\n".join(out)   # put \n between each part of list out


def split_header_body(text: str):
    # Regex to find  start of the judgment body
    # Usually starts with "ROZSUDOK" or "UZNESENIE" in uppercase on a new line
    split_pat = re.compile(r"\n\s*(?:ROZSUDOK|UZNESENIE|ROZSUDOK\s+V\s+MENE\s+SLOVENSKEJ\s+REPUBLIKY)\s*\n",
                           re.IGNORECASE)   # saving regex as pattern for future using
    match = split_pat.search(text)      # returns object - match.start() - index where it start
    if match:
        return text[:match.start()].strip(), text[match.start():].strip()

    # Fallback to ECLI position
    ecli_match = re.search(r"ECLI:[^\n]+\n", text)
    if ecli_match:
        return text[:ecli_match.end()].strip(), text[ecli_match.end():].strip() # header until ECLI

    return text[:1000], text[1000:]  # Last resort fallback


# --- STEP 3.5: Layout mormalization ---
# is it necessary to merge lines ?
TITLES = r"(?:judr\.|mgr\.|ing\.|mudr\.|phdr\.|rndr\.|bc\.|doc\.|prof\.|akad\.|thlic\.|paeddr\.)"
# \b = word boundary so "sp." matches standalone "sp." but NOT end of "CSP." or "OSP."
# without \b, lines ending in "CSP." (Civilný sporový poriadok) would match "sp."
# and incorrectly join with the next numbered point (e.g. "36. Pokiaľ dovolateľka...")
ABBREVIATIONS = r"(?:\bods\.|písm\.|č\.|č\.k\.|č\.\s*k\.|čl\.|\bsp\.|\bzn\.|\bsp\.\s*zn\.|z\.\s*z\.|\bzb\.|\bresp\.|\bnapr\.|\btzn\.|\btzv\.|\bt\.j\.|" + TITLES + r")"
PREPOSITIONS = r"(?:\s|^)(?:v|z|zo|k|ku|o|po|pri|pre|na|do)$"


# re.sub(pattern, replacement, text)
def fix_spaced_words(text):
    return re.sub(
        r'(?<!\S)((?:[a-zA-ZÁ-Žá-ž]\s){2,}[a-zA-ZÁ-Žá-ž])(?=[\s.,;:!?()\[\]]|$)', # i fixed this (before there was (?!\S) ) - did sth like this 'Dovolanie o d m i e t a.' - 'Dovolanie odmiet a.'
                  lambda m: m.group(1).replace(" ", ""), text)
# (?<!\S)  - before match cannot be non-whitespace char

def is_spaced_header(text: str) -> bool:
    clean = text.rstrip(".:").strip()   # rstrip removes dots and double dots from the end of word
    if len(clean) < 5:
        return False
    return re.match(r"^\s*(?:[a-zA-ZÁ-Žá-ž]\s+){3,}[a-zA-ZÁ-Žá-ž]\s*$", clean) is not None  ## true if regex matched


def normalize_layout(text: str) -> str:
    lines = text.split('\n')
    merged = [] # list of paragraphs (odsek)
    buf = ""

    for line in lines:
        line = line.strip()
        if not line:  # Empty line = potential paragraph break
            if buf:
                # only treat as paragraph break if sentence looks complete
                # (ends with punctuation, NOT an abbreviation or preposition)
                # this prevents page breaks mid-sentence from splitting paragraphs
                looks_complete = re.search(r'[.?!:;]\s*$', buf) and \
                                 not re.search(rf'{ABBREVIATIONS}\s*$', buf, re.IGNORECASE) and \
                                 not re.search(PREPOSITIONS, buf, re.IGNORECASE)
                if looks_complete:
                    merged.append(buf)
                    buf = ""
                    merged.append("")
                # else: skip empty line, keep accumulating (likely page break mid-sentence)
            else:
                merged.append("")   # ignore empty line ( just page break, or bad layout)
            continue    # go to next line

        if not buf:
            buf = line
            continue

        prev = buf      # prev - previous paragrapfh
        should_join = False

        # A) JOIN RULES - join to previous lines
        if prev.endswith("-"):  # Hyphen
            buf = prev[:-1] + line;
            continue
        elif re.search(PREPOSITIONS, prev, re.IGNORECASE):  # preposition
            should_join = True
        elif re.search(rf"{ABBREVIATIONS}$", prev, re.IGNORECASE):  # abbrevat
            should_join = True
        elif re.search(r"\d$", prev) and re.match(r"^-\s*\d", line): # number range or minus formula
            should_join = True

        # B) STOP RULES  - do not merge line
        elif is_spaced_header(line): # new line is header
            should_join = False
        elif prev.strip().upper().endswith("SLOVENSKEJ REPUBLIKY"):
            should_join = False
        # List items (1., a)) but NOT dates (17. októbra) or numbers mid-sentence
        # Only treat as list item if previous line ends with punctuation (sentence complete)
        elif re.match(r"^(?:[a-z]\)|\d+\.(?!\s*\d)|•)\s", line) and re.search(r'[.?!:;]\s*$', prev):
            should_join = False
        elif line[0].isupper() and re.search(r"[.?!]$", prev):  # new line start with upper case and prev is ended
            should_join = False

        # C) GENERAL
        else:
            if not re.search(r"[.?!]$", prev):  # previous does not end with interpunction
                should_join = True
            elif line[0].islower(): # lower char at beggining of line
                should_join = True

        if should_join:
            buf += " " + line
        else:
            merged.append(buf)
            buf = line

    if buf:
        merged.append(buf)
    return fix_spaced_words("\n\n".join([m for m in merged if m.strip()]))
    #     i found a small bug in my text preprocessing. Originally in notebook,
    #     i joined the paragraphs using a single newline (\n).
    #     this looked fine to the human eye when reading the .txt files. However it caused an issue for the RAG system.
    #     The chunking algorithm (and json format) needs double newlines (\n\n) to correctly recognize where one paragraph ends and a new one begins.
    #     I fixed this by joining the text with \n\n before saving it to the JSON file.


# --- STEP 3.6: Advanced uinicode normalization ---
# Unifying dashes and quotes in whole text
def advanced_unicode(text: str) -> str:
    # Dashes
    t = text.replace('–', '-').replace('—', '-')
    # FIX: quote normalization. I use explicit unicode escapes here because
    # my editor kept replacing the straight quote “ (U+0022) with a smart
    # quote \u201d (U+201D) when i saved the file. That meant ALL THREE
    # replace() calls were producing \u201d instead of ASCII “. So the text
    # still had smart quotes after “normalization” which broke golden quote
    # matching. Using \u escapes avoids this editor problem completely.
    t = t.replace('\u201e', '\u0022')   # „ (low-9) -> “ (straight)
    t = t.replace('\u201c', '\u0022')   # “ (left)  -> “ (straight)
    t = t.replace('\u201d', '\u0022')   # “ (right) -> “ (straight)

    # FIX: remove spaces before punctuation marks. PDF extraction sometimes
    # produces “potvrdzuje .” or “Bratislava ,” which is clearly wrong.
    # I found this because the LLM was “fixing” these spaces in its evidence
    # quotes, making them NOT exact substrings of the original text. For example
    # the LLM would write “potvrdzuje.” but the text had “potvrdzuje .” so
    # the grounding check flagged it as TRUNCATED even though the quote was
    # correct. Easier to fix here in preprocessing than to handle in extraction.
    t = re.sub(r'\s+([.,;:!?\)])', r'\1', t)

    return t


# --- STEP 3.7: Segmentation ---
def segment_body(text: str):
    segs = {"verdict": "", "reasoning": "", "instruction": ""}

    # Regexes
    re_reasoning = re.compile(r"(?:[oó]\s*d\s*[oóô]\s*v\s*o\s*d\s*n\s*e\s*n\s*i\s*e|odôvodnenie)", re.IGNORECASE)
    re_instruction = re.compile(r"(?:p\s*o\s*u\s*č\s*e\s*n\s*i\s*e|poučenie)", re.IGNORECASE)

    # 1. finding reasoning start
    match_res = re_reasoning.search(text)
    if match_res:
        start_res = match_res.start()
        segs["verdict"] = text[:start_res].strip()
        rest = text[start_res:].strip()

        # 2. Find instruction start (inside rest)
        match_instr = re_instruction.search(rest)
        if match_instr:
            segs["reasoning"] = rest[:match_instr.start()].strip()
            segs["instruction"] = rest[match_instr.start():].strip()
        else:
            segs["reasoning"] = rest
    else:
        # Fallback if reasoning missing
        segs["verdict"] = text

    return segs


# ==========================================
# MAIN PIPELINE
# ==========================================

def process_single_pdf(pdf_path: Path):
    doc = {"filename": pdf_path.name, "raw_pages": [], "clean_pages": []}

    # --- STEP 3.1: Text extraction ---
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            doc["raw_pages"].append(text)

    # --- STeP 3.2: Light unicode ---
    doc["clean_pages"] = [clean_light(p) for p in doc["raw_pages"]]

    # --- STEP 3.3: Metadata extraction ---
    header_snippet = doc["clean_pages"][0][:2000] if doc["clean_pages"] else ""
    doc["extracted_meta"] = extract_meta(header_snippet)

    # --- STEP 3.4: Remove pagination & separate main header ---
    no_page_nums = [remove_pagination(p) for p in doc["clean_pages"]]
    full_text = "\n".join(p.strip() for p in no_page_nums)
    h_text, b_text = split_header_body(full_text)
    doc["header_text_raw"] = h_text
    doc["body_text_raw"] = b_text
    # i also keep page level normalized text for later evidence mapping.
    # this is not perfect page map, but it is already very useful for the new
    # hierarchical rag because later i can search a parent/child snippet back
    # to page text and get rough page_start/page_end.
    doc["pages_for_map"] = [advanced_unicode(normalize_layout(p)) for p in no_page_nums]

    # --- STEp 3.5: Layout normalization ---
    doc["body_text_layout"] = normalize_layout(doc["body_text_raw"])

    # --- STEP 3.6: Advanced unicode normalization ---
    doc["body_text_final"] = advanced_unicode(doc["body_text_layout"])

    # --- STEP 3.7: Segmentation ---
    doc["segments"] = segment_body(doc["body_text_final"])

    # Clean up big texts before saving to json to save space (optional, i keep what needed)
    return {
        "filename": doc["filename"],
        "metadata": doc["extracted_meta"],
        "header": doc["header_text_raw"],
        "segments": doc["segments"],
        "pages_for_map": doc["pages_for_map"],
    }


if __name__ == "__main__":
    # Setup folders
    DATA_DIR = Path("data")
    INPUT_DIR = DATA_DIR / "01_raw_pdfs"
    OUTPUT_DIR = DATA_DIR / "02_processed_json"

    # creating output folder if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # find all PDFs
    pdf_files = list(INPUT_DIR.rglob("*.pdf"))
    print(f"Starting processing of {len(pdf_files)} files...")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            result = process_single_pdf(pdf_path)

            # dave to JSON
            out_file = OUTPUT_DIR / f"{pdf_path.stem}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

    print("Done! Saved in  processed_json folder.")
