import requests
import json
import os
import time
import re
import io
from pypdf import PdfReader

# --- CONFIGURATION ---
OUTPUT_DIR = "dataset_zmluvna_pokuta_filtered_2"
API_URL = "https://www.nsud.sk/ws/opendata.php"
BASE_FILE_URL = "https://www.nsud.sk/data/att/"

# Target years (inclusive)
ROK_OD = 2008
ROK_DO = 2025

# WIDE SCAN RANGE
# Scanning strictly to ensure coverage even with non-chronological IDs
START_ID = 250000
MIN_ID = 5000

# --- REGEX SETTINGS ---

# 1. POSITIVE MATCH (What we WANT)
# Keywords: zmluvná pokuta, sankcia, penále, relevant paragraphs (§ 544, § 300)
REGEX_POKUTA = (
    r"zmluvn[a-záäčďéíľňóôŕšťúýž]*\s+pokut[a-záäčďéíľňóôŕšťúýž]*|"  # zmluvná pokuta
    r"pokut[a-záäčďéíľňóôŕšťúýž]*|"  # pokuta (general)
    r"zmluvn[a-záäčďéíľňóôŕšťúýž]*\s+sankc[a-záäčďéíľňóôŕšťúýž]*|"  # zmluvná sankcia
    r"sankci[a-záäčďéíľňóôŕšťúýž]*|"  # sankcia/sankcie
    r"penál[a-záäčďéíľňóôŕšťúýž]*|"  # penále/penalizácia
    r"§\s*544|"  # § 544 (Civ. Code)
    r"§\s*300"  # § 300 (Comm. Code)
)

# 2. NEGATIVE MATCH (What we want to EXCLUDE)
# Filters out jurisdiction disputes (competence decisions).
# The regex \s* handles spaces between letters (e.g., "P r í s l u š n ý m")
# Updated to handle variations:
# A) "Príslušným súdom..."
# B) "Príslušným na prejednanie..." (without word 'súdom')
REGEX_EXCLUDE = (
    r"P\s*r\s*í\s*s\s*l\s*u\s*š\s*n\s*[ýy]\s*m\s+"  # "Príslušným" (with optional spaces between chars)
    r"(?:"  # Start of non-capturing group for next word options
    r"s\s*[úu]\s*d\s*o\s*m|"  # Option 1: "súdom" (spaced or normal)
    r"n\s*a\s+p\s*r\s*e\s*j\s*e\s*d\s*n\s*a\s*n\s*i\s*e"  # Option 2: "na prejednanie" (spaced or normal)
    r")"
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
    # 1. Try issue date
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

    # 2. Try file reference number
    cislo = str(data.get('cislo', ''))
    parts = cislo.split('/')
    if len(parts) >= 1:
        last_part = parts[-1]
        clean_year = ''.join(filter(str.isdigit, last_part))
        if len(clean_year) == 4:
            return int(clean_year)

    return 0  # Year not found


def check_pdf_content(pdf_bytes):
    """
    Reads PDF from memory.
    1. Extracts text.
    2. Checks EXCLUSION regex (jurisdiction disputes).
    3. Checks POSITIVE regex (penalties).
    """
    try:
        with io.BytesIO(pdf_bytes) as f:
            reader = PdfReader(f)
            full_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + " "

            # --- EXCLUSION FILTER ---
            # If the text contains "Príslušným súdom" or "Príslušným na prejednanie" (spaced or not), skip it.
            if re.search(REGEX_EXCLUDE, full_text, re.IGNORECASE):
                # print("DEBUG: Skipped due to jurisdiction/competence phrase.")
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
    print(f"ID Range: {START_ID} -> {MIN_ID}")
    print(f"Filter: Commercial Division + 'Pokuta/Sankcia' - 'Príslušným ...'")

    total_saved = 0

    for current_id in range(START_ID, MIN_ID, -1):

        # Progress info every 100 IDs
        if current_id % 100 == 0:
            print(f"... processing ID {current_id} (Saved: {total_saved}) ...")

        data = get_decision_detail(current_id)
        if not data: continue

        spisova_znacka = str(data.get('cislo', '???'))
        kolegium = str(data.get('kolegium', '')).lower()

        # 1. FILTER: COMMERCIAL DIVISION
        # Filter by file mark (Obdo/Cob...) OR by metadata 'kolegium' (Obchodné / 2)
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
                    # Logic inside check_pdf_content handles text extraction and filtering
                    if check_pdf_content(r.content):
                        filename = f"{rok}_{spisova_znacka.replace('/', '_')}_{current_id}.pdf"
                        file_path = os.path.join(OUTPUT_DIR, filename)

                        with open(file_path, 'wb') as f:
                            f.write(r.content)

                        print(f"[MATCH {rok}] {spisova_znacka} (ID: {current_id}) -> Saved.")
                        total_saved += 1
            except Exception as e:
                print(f"Error {spisova_znacka}: {e}")

        # Pause slightly less to speed up the wider scan
        time.sleep(0.02)

    print(f"\n=== DONE ===")
    print(f"Scan complete. Files saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    download_decisions()






# Zhrnutie vylepšení NSUD Scrapera (Zmluvná pokuta)
#
# Dnešné úpravy transformovali skript z jednoduchého sťahovača na inteligentný filter. Tu sú hlavné body zmien:
#
# 1. Rozšírenie kľúčových slov (Positive Filter)
#
# Pôvodne skript hľadal iba presnú frázu "zmluvná pokuta". Teraz REGEX_POKUTA zachytáva oveľa širší kontext:
#
# Terminológia: "zmluvná pokuta", "sankcia", "sankcionovanie", "penále", "penalizácia".
#
# Paragrafy: § 544 (Občiansky zákonník - definícia zmluvnej pokuty) a § 300 (Obchodný zákonník - moderačné právo súdu).
#
# Gramatika: Regex počíta so skloňovaním (sankcia, sankcie, sankciou...) pomocou [a-záä...].
#
# 2. Zmena stratégie skenovania ID (Wide Range)
#
# Zistili sme, že ID v databáze NSUD nejdú chronologicky (napr. ID 130 000 bolo z roku 2019, zatiaľ čo iné vyššie ID boli staršie).
#
# Pôvodne: Úzky rozsah a predpoklad, že nižšie ID = starší rok.
#
# Teraz: "Brute-force" skenovanie širokého rozsahu (300 000 až 50 000).
#
# Logika: Skript sa nezastaví, ak nájde rok mimo rozsahu, ale pokračuje ďalej, pretože relevantné rozhodnutia môžu byť "rozhádzané" kdekoľvek.
#
# 3. Presnejšia identifikácia Obchodného kolégia
#
# Pôvodne sme sa spoliehali len na spisovú značku (napr. text "Obdo" v čísle spisu).
#
# Vylepšenie: Pridali sme kontrolu metadát kolegium.
#
# Logika: Rozhodnutie sa stiahne, ak má v značke "Obdo/Cob" ALEBO ak má v metadátach kolégium označené ako "Obchodné" (hodnota "2"). Týmto zachytíme aj prípady s neštandardným číslovaním.
#
# 4. Vylúčenie procesných rozhodnutí (Negative Filter)
#
# Veľké množstvo nájdených dokumentov neriešilo pokutu vecne, ale len určovalo príslušnosť súdu (napr. "Príslušným súdom na prejednanie je...").
#
# Riešenie: Pridali sme REGEX_EXCLUDE.
#
# Funkcia: Ak PDF obsahuje frázu "Príslušným súdom" alebo "Príslušným na prejednanie", súbor sa ignoruje.
#
# OCR odolnosť: Regex P\s*r\s*í\s*s... zachytáva túto vetu aj vtedy, keď je v PDF napísaná s medzerami medzi písmenami (častý jav v súdnych dokumentoch: "P r í s l u š n ý m").
#
# 5. Technické detaily
#
# Preklad: Kód (komentáre a výpisy) bol preložený do angličtiny pre lepšiu udržateľnosť.
#
# In-Memory Processing: Všetky kontroly (pozitívne aj negatívne regexy) prebiehajú nad textovou vrstvou PDF priamo v RAM pamäti, bez nutnosti ukladať nerelevantné súbory na disk.