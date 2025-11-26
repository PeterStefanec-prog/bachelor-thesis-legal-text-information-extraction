import requests
import json
import os
import time
import re
import io
from pypdf import PdfReader

# --- CONFIGURATION ---
OUTPUT_DIR = "dataset_zmluvna_pokuta_2010_2025_1"
API_URL = "https://www.nsud.sk/ws/opendata.php"
BASE_FILE_URL = "https://www.nsud.sk/data/att/"

# Target years (inclusive)
ROK_OD = 2009
ROK_DO = 2025

# ID range to scan (Estimated based on your logs)
# Start high (year 2025+) and go down to 2008/2009
START_ID = 250000
MIN_ID = 160000

# EXTENDED REGEX for contractual penalties, sanctions, and relevant law paragraphs
# Covers: "pokuta", "sankcia", "penále", "§ 544" (Civ. Code), "§ 300" (Comm. Code)
# [a-záäčďéíľňóôŕšťúýž]* handles Slovak declensions (sankcia, sankcie, sankciou...)
REGEX_POKUTA = (
    r"zmluvn[a-záäčďéíľňóôŕšťúýž]*\s+pokut[a-záäčďéíľňóôŕšťúýž]*|"  # zmluvná pokuta
    r"pokut[a-záäčďéíľňóôŕšťúýž]*|"                                # pokuta (general)
    r"zmluvn[a-záäčďéíľňóôŕšťúýž]*\s+sankc[a-záäčďéíľňóôŕšťúýž]*|" # zmluvná sankcia
    r"sankci[a-záäčďéíľňóôŕšťúýž]*|"                               # sankcia/sankcie
    r"penál[a-záäčďéíľňóôŕšťúýž]*|"                                # penále/penalizácia
)

def get_decision_detail(decision_id):
    """Downloads decision metadata."""
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
    """Extracts year from issue date or file reference number."""
    # 1. Try issue date (format YYYY-MM-DD or DD.MM.YYYY)
    datum = str(data.get('datum_vydania', ''))
    if len(datum) >= 4:
        if '-' in datum:  # 2024-01-01
            try:
                return int(datum.split('-')[0])
            except:
                pass
        if '.' in datum:  # 01.01.2024
            try:
                return int(datum.split('.')[-1])
            except:
                pass

    # 2. Try file reference number (e.g., 1Obdo/15/2012)
    cislo = str(data.get('cislo', ''))
    parts = cislo.split('/')
    if len(parts) >= 1:
        last_part = parts[-1]  # Usually the year
        # Clean up spaces and other characters
        clean_year = ''.join(filter(str.isdigit, last_part))
        if len(clean_year) == 4:
            return int(clean_year)

    return 0  # Year not found


def check_pdf_content(pdf_bytes):
    """Reads PDF in memory and searches for the keywords defined in REGEX_POKUTA."""
    try:
        with io.BytesIO(pdf_bytes) as f:
            reader = PdfReader(f)
            full_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + " "

            if re.search(REGEX_POKUTA, full_text, re.IGNORECASE):
                return True
            return False
    except:
        return False


def download_decisions():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"=== STARTING LARGE SCAN ({ROK_OD} - {ROK_DO}) ===")
    print(f"ID Range: {START_ID} -> {MIN_ID}")
    print(f"Condition: Commercial Division + PDF Content: 'pokuta', 'sankcia', 'penále', '§ 544', '§ 300'")

    total_saved = 0

    # Iterate IDs downwards
    for current_id in range(START_ID, MIN_ID, -1):

        # Progress info
        if current_id % 100 == 0:
            print(f"... processing ID {current_id} (Saved: {total_saved}) ...")

        data = get_decision_detail(current_id)
        if not data: continue

        spisova_znacka = str(data.get('cislo', '???'))

        # 1. FILTER: COMMERCIAL DIVISION (Obchodné kolégium)
        # Searching only for "Obdo", "Ndob", "Obo" (exclude Cdo, Tdo, Sž...)
        if not any(x in spisova_znacka for x in ["Obdo", "Ndob", "Obo", "Cob"]):
            continue

        # 2. FILTER: YEAR
        rok = extract_year(data)
        if rok < ROK_OD or rok > ROK_DO:
            # If year is out of range, just skip (don't stop, as IDs are mixed)
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

                    # CHECK PDF CONTENT IN MEMORY
                    if check_pdf_content(r.content):
                        filename = f"{rok}_{spisova_znacka.replace('/', '_')}_{current_id}.pdf"
                        file_path = os.path.join(OUTPUT_DIR, filename)

                        with open(file_path, 'wb') as f:
                            f.write(r.content)

                        print(f"[MATCH {rok}] {spisova_znacka} -> Saved.")
                        total_saved += 1
                    # else: PDF does not contain the penalty keyword
            except Exception as e:
                print(f"Error {spisova_znacka}: {e}")

        # Pause to reduce server load
        time.sleep(0.05)

    print(f"\n=== DONE ===")
    print(f"All contracts from {ROK_OD} to {ROK_DO} saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    download_decisions()