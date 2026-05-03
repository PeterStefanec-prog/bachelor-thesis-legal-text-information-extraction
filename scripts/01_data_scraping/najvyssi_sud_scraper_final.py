

# This is my final scraper for Najvyssi sud SR (Supreme Court of Slovakia).
# I want to find decisions from commercial division where court used paragraph 301 Obchodneho zakonnika (moderation right - court can reduce
# contractual penalty because it was too high).
#
# Why brute-force? I tested that NS SR OpenData API has "art_obsah" search field, but it is not full decision text (just some parts of it ... wierd).
# Most paragraph 301 decisions I need are NOT in these annotations.
# So  only reliable way is to check every decision PDF text.
#
# How it works:
# 1) I iterate over ID range (NS SR IDs are not chronological so I need to cover wide range - see comment from old scraper).
# 2) For each ID I fetch metadata (cheap - small JSON).
# 3) Quick filter: must be commercial case AND date must be in my range.
# 4) Download PDF to memory and look for § 301 or moderation keywords.
# 5) Save if match, also save small snippet for easy manual check.
#
# this is similar on old scraper_with_moderation_extended_plus_keywords.py.
#
# Note: this is slow! For full ID range expect few hours runtime.
# Script resumes from existing CSV, so can be stop  and continuye later

import io       # for reading pdfs just in ram (not having to save them on disk)
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader # better and faster for light weight text extraction than pdfplumber (it is more roobustt, pages, tables, etc - no need for keyword check)
from tqdm import tqdm

#################################
#  ######### CONFIG ############

# year range
YEAR_FROM = 2015
YEAR_TO = 2025

# ID range to scan. NS SR IDs are not sorted by date
# Based on probing:
#   - ID 130000-280000 covers most of 2015-2025 (with some older mixed in)
#   - Old scraper used 245000 -> 10000 which is 235k IDs (too much)
#my range is   wide enough to catch most 2015+ cases.
ID_START = 280000         # 250000  # we iterate DOWN from this
ID_STOP = 100000           # 235000


# Output folders
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PDF_DIR = DATA_DIR / "final_nsud_pdfs"
CSV_PATH = DATA_DIR / "final_nsud_metadata.csv"

# NS SR OpenData endpoints
API_BASE = "https://www.nsud.sk/ws/opendata.php"    # path to getting metada of decisions
FILE_BASE = "https://www.nsud.sk/data/att/"         # path to decisions that id have been found

# Pause between request so server will block me (happpened already)
SLEEP_METADATA = 0.02
SLEEP_PDF = 0.05

#############################################
##########################################
# ## REGEXES ##

# Find paragraph 301.
# I used to accept OCR variants like "S 301", "g 301" but that give too many false positivess
# Biggest one was "I. ÚS 301/06" which is reference to Ústavný súd  case, not paragraph 301.
# So I keep only strict forms: §, par., ust., paragraf, odst., zmysle.
REGEX_PARAGRAPH_301 = re.compile(       # compiling regex so i can use it more effectively after
    r"(?:§|par\.|ust\.|ustanoven\w+|paragraf\w*|odst\.|v\s+zmysle)\s*301\b",    # par. 301, ustanov. 301, ..
    re.IGNORECASE,
)

# To kill remaining false positives i also check that PDF contains some commercial-code indicator
# If decision talks about § 301 ObchZ then PDF has to mention either "513/1991" (the law number) or "Obchodn... zákon" or "ObZ" somewhere.
REGEX_COMMERCIAL_INDICATOR = re.compile(
    r"513\s*/\s*1991|Obchodn\w*\s+zák|obchodného\s+zákonník|ObZ\b|Obch\.?\s*zák",
    re.IGNORECASE,
)

# If 301 is followed by /\d{3,4} it is law number reference (e.g. "301/2005"),  not  paragraph reference.
REGEX_LAW_NUMBER = re.compile(r"301\s*[/.]\s*\d{3,4}")      # i do not download decision like these

# Reject paragraph 301 if it refers to different code (CSP, Trestný, Správny).
REGEX_WRONG_CODE = re.compile(
    r"301[^0-9]{0,30}(CSP|C\.?\s*s\.?\s*p\.?|Civiln|Trest|Správ|"
    r"Tr\.?\s*z\.?|T\.?\s*z\.?)",
    re.IGNORECASE,
)

# If there is no paragraph 301 i maybe want decisions where they talk about moderation (reducing contractual penalty)
#  i search for words like neprimerana, znizena, moderacne near "zmluvna pokuta" (within 25 words)
REGEX_MODERATION_KEYWORDS = re.compile(
    r"((?:neprimeran|neúmern|znížen|znížil|moderač|primeran)[a-ž]*)"
    r"\W+(?:\w+\W+){0,25}?zmluvn[a-ž]*\s+pokut[a-z]*|"
    r"zmluvn[a-ž]*\s+pokut[a-z]*\W+(?:\w+\W+){0,25}?"
    r"((?:neprimeran|neúmern|znížen|znížil|moderač|primeran)[a-ž]*)",
    re.IGNORECASE,
)


# ### HELPERS ###


def clean_text(text: str) -> str:
    # normalizing whitespace. I strip nbsp (0xa0 - nonbreaking space) and tabs.
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\t", " ") # replace nonbreaking space and tabs with space
    return " ".join(text.split())       # all new lines, more spaces will be reduced to normal spaces


def parse_year(datum_str: str) -> int:
    # NS SR returns date like "2017-04-20" or "2017-04-20 00:00:00"
    if not datum_str:
        return 0
    try:
        return int(str(datum_str)[:4])      # return date
    except (ValueError, TypeError):
        return 0


def get_metadata(session: requests.Session, dec_id: int) -> dict | None:
    # Fetch metadata for one decision
    try:
        r = session.get(
            API_BASE,
            params={"getDecision": "", "id": str(dec_id)},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def is_commercial_case(meta: dict) -> bool:
    # Check if it is commercial case
    # I look at kolegium (code 2) and  spisova znacka (Obdo, Obo, Cob are commercial prefixes).
    kolegium = str(meta.get("kolegium", "")).strip()
    oblast = str(meta.get("oblast", "")).lower()
    spis = str(meta.get("cislo", "")).lower()

    if kolegium == "2":
        return True
    if "obchod" in oblast:
        return True
    if "obdo" in spis or "obo" in spis or "cob" in spis:
        return True
    return False

####
#### Download PDF to memory, extract text, check for paragraph 301 or keywords ######
def analyze_pdf(session: requests.Session, pdf_url: str):
    # Returns (found_flag, match_type, pdf_bytes, snippet)- exanoke - (True, "MATCH_KEYWORDS", b"%PDF-1.4...", "... zmluvná pokuta bola neprimeraná ...")
    if not pdf_url:
        return False, None, None, None

    full_url = pdf_url if pdf_url.startswith("http") else FILE_BASE + pdf_url.lstrip("/")   # creating full url

    try:
        r = session.get(full_url, timeout=20)   # downloading pdf
        if r.status_code != 200:
            return False, None, None, None

        # reading pdf document
        text = ""
        with io.BytesIO(r.content) as f:    # r.content - raw bytes of answer
            try:
                reader = PdfReader(f)
                for page in reader.pages:   # reader.pages - list of pages
                    t = page.extract_text()
                    if t:
                        text += t + " "
            except Exception:
                # Some PDFs are corrupted or encrypted
                return False, None, None, None

        # Very short text means it is probably scan - I skip for now (skipping ocr)
        if len(text.strip()) < 50:
            return False, "SCAN_SKIPPED", None, None

        text = clean_text(text) # normalizing texgt at least a little bit

        # Global check: does whole PDF reference Obchodny zakonnik?
        # If not, its probably not about paragraph 301 ObchZ - skip early.
        has_commercial = bool(REGEX_COMMERCIAL_INDICATOR.search(text))

        # 1) Find paragraph 301 (strict forms only - see regex comment above)
        for m in REGEX_PARAGRAPH_301.finditer(text):    # every m has - where match starts, ends, matched text
            # Skip law number refs like "301/2005"
            tail = text[m.end():m.end() + 15]
            if REGEX_LAW_NUMBER.search("301" + tail):
                continue

            # Skip if close context says different code (CSP, Trestny...)
            context_small = text[max(0, m.start()):m.end() + 40]
            if REGEX_WRONG_CODE.search(context_small):
                continue

            # Need commercial context somewhere in PDF (otherwise its not ObchZ)
            if not has_commercial:
                continue

            start_c = max(0, m.start() - 120)
            end_c = min(len(text), m.end() + 120)
            return True, "MATCH_PARAGRAF_301", r.content, text[start_c:end_c]

        # 2) No pargraph 301 found - try moderation keywords near zmluvna pokuta.
        # Also require commercial indicator so we dont get civil cases.
        if has_commercial:
            km = REGEX_MODERATION_KEYWORDS.search(text)
            if km:
                start_c = max(0, km.start() - 120)
                end_c = min(len(text), km.end() + 120)
                return True, "MATCH_KEYWORDS", r.content, text[start_c:end_c]   # creating snippet of match

        return False, "NO_MATCH", None, None        # nothing matches

    except Exception:
        return False, None, None, None


def save_decision(meta: dict, pdf_bytes: bytes, match_type: str, snippet: str, dec_id: int) -> dict:
    # Save PDF to disk and return row for CSV
    spis = str(meta.get("cislo", "nezname")).replace("/", "_").replace(" ", "_")
    fname = f"{dec_id}_{match_type}_{spis}.pdf"
    out_path = PDF_DIR / fname

    with open(out_path, "wb") as f:
        f.write(pdf_bytes)

    return {
        "id": dec_id,
        "match_type": match_type,
        "spisova_znacka": meta.get("cislo"),
        "datum": meta.get("datum"),
        "kolegium": meta.get("kolegium"),
        "senat": meta.get("senat"),
        "ecli": meta.get("ecli"),
        "merito": meta.get("merito"),
        "sudca": meta.get("sudca"),
        "snippet": snippet,
        "pdf_file": fname,
    }


# #### MAIN  ####


def main():
    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "NSUD-Thesis-Scraper/1.0"})

    # Resume from existing CSV
    done_ids: set[int] = set()  # process ids from csv
    rows: list[dict] = []
    if CSV_PATH.exists():
        try:
            df_old = pd.read_csv(CSV_PATH, sep="|")
            done_ids = set(df_old["id"].astype(int).tolist())
            rows = df_old.to_dict("records")
            print(f"[INFO] Loaded {len(done_ids)} already saved")
        except Exception as e:
            print(f"[WARN] cant load old CSV: {e}")

    # how many ids we are going through
    total_ids = ID_START - ID_STOP
    print(f"[INFO] Scanning ID range {ID_START} -> {ID_STOP} ({total_ids} IDs)")
    print(f"[INFO] Year filter: {YEAR_FROM}-{YEAR_TO}")

    # counters
    saved = 0
    checked_commercial = 0
    errors = 0

    pbar = tqdm(range(ID_START, ID_STOP, -1), desc="Scanning")
    for dec_id in pbar:
        if dec_id in done_ids:
            continue

        meta = get_metadata(session, dec_id)    # first try to request metadata (cheaper)
        time.sleep(SLEEP_METADATA)
        if not meta:
            errors += 1
            continue

        # Quick filter by date - before any expensive PDF work
        year = parse_year(meta.get("datum", ""))
        if year < YEAR_FROM or year > YEAR_TO:
            continue

        if not is_commercial_case(meta):
            continue

        checked_commercial += 1

        # Download PDF , extract text and  check content (by regex filtering)
        ok, match_type, pdf_bytes, snippet = analyze_pdf(session, meta.get("subor"))
        time.sleep(SLEEP_PDF)
        if not ok:
            continue

        # saving match
        row = save_decision(meta, pdf_bytes, match_type, snippet, dec_id)
        rows.append(row)    # save metadata
        saved += 1

        # Save CSV after every match (crash-safe, or pc shutdown ..)
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False, sep="|", encoding="utf-8")

        pbar.set_postfix(commercial=checked_commercial, saved=saved)    # add stats to bar

    print(f"\n[DONE]")
    print(f"  Commercial cases in range: {checked_commercial}")
    print(f"  Saved: {saved}")
    print(f"  Errors: {errors}")
    print(f"  CSV: {CSV_PATH}")
    print(f"  PDFs: {PDF_DIR}")


if __name__ == "__main__":
    main()
