import requests
import pandas as pd
import re
import time
import os
import io
from tqdm import tqdm
from pypdf import PdfReader

# --- CONFIGURATION ---
OUTPUT_CSV = "dataset_moderacne_pravo_301_BBBB.csv"
PDF_DIR = "rozhodnutia_301_pdf_BBB"

# ID range for searching (Brute force)
# Current IDs (end of 2024/2025) are approx 275000.
# Year 2015 is approx 150000.
START_ID = 240000
MIN_ID = 19000

# --- REGEXES (CORE SCRIPT) ---

# 1. What exactly we are looking for: § 301
# Looking for: § 301, ust. 301, ustanovenie § 301
# (?i) means case-insensitive
REGEX_TARGET = re.compile(r'(§|ust\.|ustanovenie|zmysle)\s*301', re.IGNORECASE)

# 2. What we must EXCLUDE (False Positives)
# If § 301 related to the Criminal Procedure or Civil Non-Contentious Procedure.
# Note: In commercial matters, § 301 CSP (Civil Dispute Order) concerns legal costs/witness fees,
# but to be safe, we primarily want the Commercial Code.
# If it finds "§ 301 Trestného", we discard it.
REGEX_EXCLUDE_CONTEXT = re.compile(r'301\s+(trest|civiln|súdneho|správn)', re.IGNORECASE)


class ModeracnePravoScraper:
    def __init__(self):
        self.base_url = "https://www.nsud.sk/ws/opendata.php"
        self.file_base_url = "https://www.nsud.sk/data/att/"

        if not os.path.exists(PDF_DIR):
            os.makedirs(PDF_DIR)

        self.results = []
        # Loading existing data to continue interrupted work
        if os.path.exists(OUTPUT_CSV):
            try:
                self.results = pd.read_csv(OUTPUT_CSV, sep='|').to_dict('records')
                print(f"--- Načítaných {len(self.results)} už stiahnutých rozhodnutí ---")
            except:
                pass

    def get_metadata(self, decision_id):
        """Downloads metadata about the decision."""
        try:
            params = {'getDecision': '', 'id': decision_id}
            # Timeout 5 seconds is enough, if no response, we move on
            r = requests.get(self.base_url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data: return data[0]
                if isinstance(data, dict) and 'cislo' in data: return data
            return None
        except:
            return None

    def analyze_pdf(self, pdf_url):
        """
        Downloads PDF to RAM and returns text if it contains § 301.
        """
        try:
            # URL fix (domain is sometimes missing)
            full_url = pdf_url if pdf_url.startswith('http') else self.file_base_url + pdf_url.lstrip('/')

            r = requests.get(full_url, timeout=10)
            if r.status_code != 200:
                return False, None, None

            # Reading PDF via pypdf (in memory, without saving to disk yet)
            with io.BytesIO(r.content) as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + " "

            # --- VALIDATION LOGIC ---

            # 1. Looking for § 301
            match = REGEX_TARGET.search(text)
            if match:
                # 2. Context Window check
                # Check text around the match to see if it is not Criminal Procedure
                start_idx = max(0, match.start() - 50)
                end_idx = min(len(text), match.end() + 100)
                context_snippet = text[start_idx:end_idx]

                # If we see the word "Trestného" or "Civilného mimosporového" around 301, ignore.
                # (Most judges write just "§ 301", which in a commercial matter means Commercial Code, so we take it).
                if REGEX_EXCLUDE_CONTEXT.search(context_snippet):
                    return False, None, None

                return True, r.content, text

            return False, None, None

        except Exception as e:
            # print(f"Chyba PDF: {e}")
            return False, None, None

    def save_match(self, meta, pdf_bytes, full_text, decision_id):
        """Saves the result."""
        spisova_znacka = str(meta.get('cislo', 'nezname')).replace('/', '_')
        clean_date = str(meta.get('datum', 'neznamy')).replace('.', '_')

        # Filename: YEAR_FILE_ID.pdf
        filename = f"{decision_id}_{spisova_znacka}.pdf"
        file_path = os.path.join(PDF_DIR, filename)

        # Save PDF physically to disk
        with open(file_path, 'wb') as f:
            f.write(pdf_bytes)

        # Find context for Excel (to see the sentence where it is mentioned)
        match = REGEX_TARGET.search(full_text)
        snippet = "..."
        if match:
            start = max(0, match.start() - 200)
            end = min(len(full_text), match.end() + 200)
            snippet = full_text[start:end].replace('\n', ' ').replace('\r', '')

        record = {
            'id': decision_id,
            'spisova_znacka': meta.get('cislo'),
            'datum_vydania': meta.get('datum'),
            'sudca': meta.get('sudca'),
            'kolegium': meta.get('kolegium'),
            'merito': meta.get('merito', ''),  # What it was about (keywords from court)
            'najdeny_kontext_301': snippet,  # Sentence around § 301
            'cesta_k_pdf': file_path,
            'url_na_web': f"https://www.nsud.sk{meta.get('subor')}"
        }

        self.results.append(record)

        # Continuous writing to CSV (Overwrite mode - always overwriting file with new data)
        # Using '|' as separator because there are commas in legal texts
        df = pd.DataFrame(self.results)
        df.to_csv(OUTPUT_CSV, index=False, sep='|', encoding='utf-8')

    def run(self):
        print(f"=== ŠTART SCRAPERA: MODERAČNÉ PRÁVO (§ 301) ===")
        print(f"Hľadám v ID: {START_ID} -> {MIN_ID}")
        print(f"Filter: Obchodné kolégium + Text '§ 301'")

        count_saved = 0

        # Progress bar via tqdm
        for current_id in tqdm(range(START_ID, MIN_ID, -1), desc="Analyzujem"):

            # A) Get metadata
            meta = self.get_metadata(current_id)
            if not meta:
                continue

            # B) Filter: Only Commercial matters
            # Check if it is Commercial Law College (code 2) OR if file reference contains "Ob"
            kolegium = str(meta.get('kolegium', '')).lower()
            spis = str(meta.get('cislo', ''))

            is_commercial = ('2' in kolegium) or \
                            ('ob' in spis.lower()) or \
                            ('cob' in spis.lower())

            if not is_commercial:
                continue  # Skipping civil and criminal matters (saving 80% time)

            # C) Filter: PDF Content
            pdf_url = meta.get('subor')
            if pdf_url:
                found, pdf_bytes, text = self.analyze_pdf(pdf_url)

                if found:
                    self.save_match(meta, pdf_bytes, text, current_id)
                    count_saved += 1
                    tqdm.write(f" NÁJDENÉ! ID {current_id} ({spis}) -> Obsahuje § 301")

            # Short pause for server
            time.sleep(0.05)

        print(f"\n=== HOTOVO ===")
        print(f"Celkovo uložených rozhodnutí: {count_saved}")
        print(f"Dáta nájdeš v súbore: {OUTPUT_CSV}")


if __name__ == "__main__":
    # Installation of necessary libraries:
    # pip install requests pandas pypdf tqdm
    scraper = ModeracnePravoScraper()
    scraper.run()