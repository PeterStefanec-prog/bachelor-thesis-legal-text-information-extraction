import os
import re
import json
import unicodedata
from pathlib import Path
import pdfplumber


# ==========================================
# FUNCTIONS (From Jupyter Notebook)
# ==========================================

# --- STEP 3.2: Light Unicode ---
def clean_light(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\u200b', '').replace('\u00ad', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text


# --- STEP 3.3: Metadata Extraction ---
def extract_meta(header_text: str):
    meta = {}
    date_match = re.search(r"(\d{1,2}\.\s*\d{1,2}\.\s*\d{4})", header_text)
    if date_match: meta['date_raw'] = date_match.group(1).replace(" ", "")
    sz_match = re.search(r"Spisová\s*značka:?\s*([\w\d/]+)", header_text, re.IGNORECASE)
    if sz_match: meta['case_id'] = sz_match.group(1)
    ecli_match = re.search(r"(ECLI:[A-Z0-9:.]+)", header_text)
    if ecli_match: meta['ecli'] = ecli_match.group(1)
    return meta


# --- STEP 3.4: Remove Pagination & Separate Main Header ---
def remove_pagination(text: str) -> str:
    lines = text.split('\n')
    out = []
    for line in lines:
        s = line.strip()
        # Remove "Strana X", "- X -", but KEEP "Spisová značka" etc.
        if re.match(r"^(?:strana\s+\d+(?:\s*z\s*\d+)?|\-?\s*\d+\s*\-?)$", s, re.IGNORECASE):
            continue
        out.append(line)
    return "\n".join(out)


def split_header_body(text: str):
    # Regex to find the start of the judgment body
    # Usually starts with "ROZSUDOK" or "UZNESENIE" in uppercase on a new line
    split_pat = re.compile(r"\n\s*(?:ROZSUDOK|UZNESENIE|ROZSUDOK\s+V\s+MENE\s+SLOVENSKEJ\s+REPUBLIKY)\s*\n",
                           re.IGNORECASE)
    match = split_pat.search(text)
    if match:
        return text[:match.start()].strip(), text[match.start():].strip()

    # Fallback to ECLI position
    ecli_match = re.search(r"ECLI:[^\n]+\n", text)
    if ecli_match:
        return text[:ecli_match.end()].strip(), text[ecli_match.end():].strip()

    return text[:1000], text[1000:]  # Last resort fallback


# --- STEP 3.5: Layout Normalization ---
TITLES = r"(?:judr\.|mgr\.|ing\.|mudr\.|phdr\.|rndr\.|bc\.|doc\.|prof\.|akad\.|thlic\.|paeddr\.)"
ABBREVIATIONS = r"(?:ods\.|písm\.|č\.|čl\.|sp\.|zn\.|sp\.\s*zn\.|z\.\s*z\.|zb\.|" + TITLES + r")"
PREPOSITIONS = r"(?:\s|^)(?:v|z|zo|k|ku|o|po|pri|pre|na|do)$"


def fix_spaced_words(text):
    return re.sub(r'(?<!\S)((?:[a-zA-ZÁ-Žá-ž]\s){2,}[a-zA-ZÁ-Žá-ž])(?!\S)',
                  lambda m: m.group(1).replace(" ", ""), text)


def is_spaced_header(text: str) -> bool:
    clean = text.rstrip(".:").strip()
    if len(clean) < 5: return False
    return re.match(r"^\s*(?:[a-zA-ZÁ-Žá-ž]\s+){3,}[a-zA-ZÁ-Žá-ž]\s*$", clean) is not None


def normalize_layout(text: str) -> str:
    lines = text.split('\n')
    merged = []
    buf = ""

    for line in lines:
        line = line.strip()
        if not line:  # Empty line = paragraph break
            if buf: merged.append(buf); buf = ""
            merged.append("")
            continue

        if not buf:
            buf = line;
            continue

        prev = buf
        should_join = False

        # A) JOIN RULES
        if prev.endswith("-"):  # Hyphen
            buf = prev[:-1] + line;
            continue
        elif re.search(PREPOSITIONS, prev, re.IGNORECASE):
            should_join = True
        elif re.search(rf"{ABBREVIATIONS}$", prev, re.IGNORECASE):
            should_join = True
        elif re.search(r"\d$", prev) and re.match(r"^-\s*\d", line):
            should_join = True

        # B) STOP RULES
        elif is_spaced_header(line):
            should_join = False
        elif prev.strip().upper().endswith("SLOVENSKEJ REPUBLIKY"):
            should_join = False
        # List items (1., a)) but NOT dates (20. 1.)
        elif re.match(r"^(?:[a-z]\)|\d+\.(?!\s*\d)|•)\s", line):
            should_join = False
        elif line[0].isupper() and re.search(r"[.?!]$", prev):
            should_join = False

        # C) GENERAL
        else:
            if not re.search(r"[.?!]$", prev):
                should_join = True
            elif line[0].islower():
                should_join = True

        if should_join:
            buf += " " + line
        else:
            merged.append(buf); buf = line

    if buf: merged.append(buf)
    return fix_spaced_words("\n\n".join([m for m in merged if m.strip()]))
    #     I found a small bug in my text preprocessing. Originally in notebook,
    #     I joined the paragraphs using a single newline (\n).
    #     This looked fine to the human eye when reading the .txt files. However, it caused an issue for the RAG system.
    #     The chunking algorithm (and json format) needs double newlines (\n\n) to correctly recognize where one paragraph ends and a new one begins.
    #     I fixed this by joining the text with \n\n before saving it to the JSON file.


# --- STEP 3.6: Advanced Unicode Normalization ---
# Unifying dashes and quotes.
def advanced_unicode(text: str) -> str:
    # Dashes
    t = text.replace('–', '-').replace('—', '-')
    # Quotes
    t = t.replace('„', '"').replace('“', '"').replace('”', '"')
    return t


# --- STEP 3.7: Segmentation ---
def segment_body(text: str):
    segs = {"verdict": "", "reasoning": "", "instruction": ""}

    # Regexes
    re_reasoning = re.compile(r"(?:[oó]\s*d\s*[oóô]\s*v\s*o\s*d\s*n\s*e\s*n\s*i\s*e|odôvodnenie)", re.IGNORECASE)
    re_instruction = re.compile(r"(?:p\s*o\s*u\s*č\s*e\s*n\s*i\s*e|poučenie)", re.IGNORECASE)

    # 1. Find Reasoning start
    match_res = re_reasoning.search(text)
    if match_res:
        start_res = match_res.start()
        segs["verdict"] = text[:start_res].strip()
        rest = text[start_res:].strip()

        # 2. Find Instruction start (inside rest)
        match_instr = re_instruction.search(rest)
        if match_instr:
            segs["reasoning"] = rest[:match_instr.start()].strip()
            segs["instruction"] = rest[match_instr.start():].strip()
        else:
            segs["reasoning"] = rest
    else:
        # Fallback if Reasoning missing
        segs["verdict"] = text

    return segs


# ==========================================
# MAIN PIPELINE
# ==========================================

def process_single_pdf(pdf_path: Path):
    doc = {"filename": pdf_path.name, "raw_pages": [], "clean_pages": []}

    # --- STEP 3.1: Text Extraction ---
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            doc["raw_pages"].append(text)

    # --- STEP 3.2: Light Unicode ---
    doc["clean_pages"] = [clean_light(p) for p in doc["raw_pages"]]

    # --- STEP 3.3: Metadata Extraction ---
    header_snippet = doc["clean_pages"][0][:2000] if doc["clean_pages"] else ""
    doc["extracted_meta"] = extract_meta(header_snippet)

    # --- STEP 3.4: Remove Pagination & Separate Main Header ---
    no_page_nums = [remove_pagination(p) for p in doc["clean_pages"]]
    full_text = "\n".join(no_page_nums)
    h_text, b_text = split_header_body(full_text)
    doc["header_text_raw"] = h_text
    doc["body_text_raw"] = b_text

    # --- STEP 3.5: Layout Normalization ---
    doc["body_text_layout"] = normalize_layout(doc["body_text_raw"])

    # --- STEP 3.6: Advanced Unicode Normalization ---
    doc["body_text_final"] = advanced_unicode(doc["body_text_layout"])

    # --- STEP 3.7: Segmentation ---
    doc["segments"] = segment_body(doc["body_text_final"])

    # Clean up big texts before saving to json to save space (optional, we keep what's needed)
    return {
        "filename": doc["filename"],
        "metadata": doc["extracted_meta"],
        "header": doc["header_text_raw"],
        "segments": doc["segments"]
    }


if __name__ == "__main__":
    # Setup folders
    DATA_DIR = Path("../../data")
    INPUT_DIR = DATA_DIR / "01_raw_pdfs"
    OUTPUT_DIR = DATA_DIR / "02_processed_json"

    # Create output folder if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all PDFs
    pdf_files = list(INPUT_DIR.rglob("*.pdf"))
    print(f"Starting processing of {len(pdf_files)} files...")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            result = process_single_pdf(pdf_path)

            # Save to JSON
            out_file = OUTPUT_DIR / f"{pdf_path.stem}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

    print("Done! Check the processed_json folder.")