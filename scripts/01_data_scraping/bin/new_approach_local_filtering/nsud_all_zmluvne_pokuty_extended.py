import requests
import json
import os
import time
import re
import io
from pypdf import PdfReader

# New approach: instead of trusting API search (art_obsah only matches
# annotation), I just brute force through ID range and check PDF text
# myself. Slower but catches way more cases.

# --- CONFIG ---
OUTPUT_DIR = "dataset_zmluvna_pokuta_filtered_2"
API_URL = "https://www.nsud.sk/ws/opendata.php"
BASE_FILE_URL = "https://www.nsud.sk/data/att/"

# Target years (both included)
ROK_OD = 2008
ROK_DO = 2025

# Wide ID range scan.
# I noticed NS SR IDs are NOT chronological, so I have to scan wide.
START_ID = 250000
MIN_ID = 5000

# --- REGEX ---

# 1. POSITIVE MATCH (what I want)
# Words: zmluvna pokuta, sankcia, penale, plus paragraphs § 544 (OZ)
# and § 300 (ObchZ - definition of contractual penalty).
REGEX_POKUTA = (
    r"zmluvn[a-záäčďéíľňóôŕšťúýž]*\s+pokut[a-záäčďéíľňóôŕšťúýž]*|"  # zmluvna pokuta
    r"pokut[a-záäčďéíľňóôŕšťúýž]*|"  # pokuta in general
    r"zmluvn[a-záäčďéíľňóôŕšťúýž]*\s+sankc[a-záäčďéíľňóôŕšťúýž]*|"  # zmluvna sankcia
    r"sankci[a-záäčďéíľňóôŕšťúýž]*|"  # sankcia/sankcie
    r"penál[a-záäčďéíľňóôŕšťúýž]*|"  # penale/penalizacia
    r"§\s*544|"  # § 544 (Civil code)
    r"§\s*300"  # § 300 (Commercial code)
)

# 2. NEGATIVE MATCH (what I want to EXCLUDE)
# Filters out jurisdiction disputes (competence decisions).
# \s* handles spaces between letters because some PDFs have weird
# formatting like "P r i s l u s n y m" instead of "Prislusnym".
# Two variants:
# A) "Prislusnym sudom..."
# B) "Prislusnym na prejednanie..." (without word "sudom")
REGEX_EXCLUDE = (
    r"P\s*r\s*í\s*s\s*l\s*u\s*š\s*n\s*[ýy]\s*m\s+"  # "Prislusnym" (with optional spaces)
    r"(?:"  # start of group for next word options
    r"s\s*[úu]\s*d\s*o\s*m|"  # option 1: "sudom" (spaced or normal)
    r"n\s*a\s+p\s*r\s*e\s*j\s*e\s*d\s*n\s*a\s*n\s*i\s*e"  # option 2: "na prejednanie"
    r")"
)


def get_decision_detail(decision_id):
    # download metadata
    try:
        response = requests.get(API_URL, params={"getDecision": "", "id": decision_id}, timeout=5)
        if response.status_code != 200: return None
        data = response.json()
        if isinstance(data, list) and data: return data[0]
        if isinstance(data, dict):
            if 'cislo' in data: return data
            first = next(iter(data.values()))
            if isinstance(first, dict) and 'cislo' in first: return first
        return None
    except:
        return None


def extract_year(data):
    # Try to get year from issue date or from spisova znacka
    # 1. try issue date
    datum = str(data.get('datum_vydania', ''))
    if len(datum) >= 4:
        if '-' in datum:
            try:
                return int(datum.split('-')[0])
            except:
                pass
        if '.' in datum:
            try:
                return int(datum.split('.')[-1])
            except:
                pass

    # 2. try file reference number
    cislo = str(data.get('cislo', ''))
    parts = cislo.split('/')
    if len(parts) >= 1:
        last_part = parts[-1]
        clean_year = ''.join(filter(str.isdigit, last_part))
        if len(clean_year) == 4:
            return int(clean_year)

    return 0  # year not found


def check_pdf_content(pdf_bytes):
    # Read PDF from memory:
    # 1. extract text
    # 2. check EXCLUDE regex (jurisdiction disputes)
    # 3. check POSITIVE regex (penalty)
    try:
        with io.BytesIO(pdf_bytes) as f:
            reader = PdfReader(f)
            full_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + " "

            # --- EXCLUDE FILTER ---
            # If text contains "Prislusnym sudom" or "Prislusnym na prejednanie"
            # (spaced or not), skip it.
            if re.search(REGEX_EXCLUDE, full_text, re.IGNORECASE):
                # print("DEBUG: skipped due to jurisdiction phrase")
                return False

            # --- POSITIVE FILTER ---
            if re.search(REGEX_POKUTA, full_text, re.IGNORECASE):
                return True

            return False
    except:
        return False


def download_decisions():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"=== STARTING FILTERED SCAN ({ROK_OD} - {ROK_DO}) ===")
    print(f"ID range: {START_ID} -> {MIN_ID}")
    print(f"Filter: commercial division + 'Pokuta/Sankcia' - 'Prislusnym ...'")

    total_saved = 0

    for current_id in range(START_ID, MIN_ID, -1):

        # progress info every 100 IDs
        if current_id % 100 == 0:
            print(f"... processing ID {current_id} (saved: {total_saved}) ...")

        data = get_decision_detail(current_id)
        if not data: continue

        spisova_znacka = str(data.get('cislo', '???'))
        kolegium = str(data.get('kolegium', '')).lower()

        # 1. FILTER: COMMERCIAL DIVISION
        # By spisova znacka (Obdo/Cob...) OR by kolegium metadata (Obchodne / 2)
        is_commercial_mark = any(x in spisova_znacka for x in ["Obdo", "Ndob", "Obo", "Cob"])
        is_commercial_collegium = 'obchod' in kolegium or '2' in kolegium

        if not (is_commercial_mark or is_commercial_collegium):
            continue

        # 2. FILTER: YEAR
        rok = extract_year(data)
        if rok < ROK_OD or rok > ROK_DO:
            continue

        # 3. DOWNLOAD AND CONTENT CHECK
        pdf_path = data.get('subor')
        if pdf_path:
            if pdf_path.startswith('http'):
                full_url = pdf_path
            else:
                if pdf_path.startswith('/'): pdf_path = pdf_path[1:]
                full_url = BASE_FILE_URL + pdf_path

            try:
                r = requests.get(full_url, timeout=10)
                if r.status_code == 200:
                    # check_pdf_content does text extraction and filtering
                    if check_pdf_content(r.content):
                        filename = f"{rok}_{spisova_znacka.replace('/', '_')}_{current_id}.pdf"
                        file_path = os.path.join(OUTPUT_DIR, filename)

                        with open(file_path, 'wb') as f:
                            f.write(r.content)

                        print(f"[MATCH {rok}] {spisova_znacka} (ID: {current_id}) -> saved")
                        total_saved += 1
            except Exception as e:
                print(f"Error {spisova_znacka}: {e}")

        # short pause - server doesnt complain
        time.sleep(0.02)

    print(f"\n=== DONE ===")
    print(f"Scan complete. Files saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    download_decisions()





# What I improved here vs older versions:
#
# 1. Wider keyword regex
#    Older versions only looked for exact "zmluvna pokuta". Now I catch:
#    - "zmluvna pokuta", "sankcia", "sankcionovanie", "penale", "penalizacia"
#    - paragraphs § 544 (OZ - definition) and § 300 (ObchZ - moderation)
#    - skloňovanie covered by [a-z...] character class
#
# 2. Wide ID range scan (brute force)
#    Found out NS SR IDs are NOT in chronological order. So I cant just
#    say "start from year 2015 ID" - have to scan a wide range.
#    Old approach: narrow range, assume lower ID = older year.
#    New approach: brute-force from 250000 down to 5000.
#
# 3. Better commercial detection
#    Old: only check spisova znacka for "Obdo".
#    New: also check kolegium metadata field. Catches edge cases with
#    weird numbering.
#
# 4. Negative filter for procedural decisions
#    Many decisions were not really about contractual penalty - they
#    were just "prislusnym sudom je..." (which court should decide).
#    Added REGEX_EXCLUDE for these.
#    Also handles weird spaced text like "P r i s l u s n y m"
#    (some PDFs have letter spacing for emphasis).
#
# 5. In-memory PDF processing
#    All filter checks happen in RAM before saving. So I dont save
#    irrelevant PDFs to disk first and delete them later.
