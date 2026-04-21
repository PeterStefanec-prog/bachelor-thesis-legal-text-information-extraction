import requests
import pandas as pd
import re
import time
import os
import io
from tqdm import tqdm
from pypdf import PdfReader

# This is when I narrowed it down to specifically § 301 ObchZ (moderation
# right). Looking only for decisions where judge actualy used this paragraph.

# --- CONFIG ---
OUTPUT_CSV = "dataset_moderacne_pravo_301_BBBB.csv"
PDF_DIR = "rozhodnutia_301_pdf_BBB"

# ID range for searching (brute force)
# Current IDs (end of 2024/2025) are around 275000.
# Year 2015 is around 150000.
START_ID = 240000
MIN_ID = 19000

# --- REGEXES (CORE) ---

# 1. What I look for: § 301
# Forms: § 301, ust. 301, ustanovenie § 301
# (?i) = case insensitive
REGEX_TARGET = re.compile(r'(§|ust\.|ustanovenie|zmysle)\s*301', re.IGNORECASE)

# 2. What I have to EXCLUDE (false positives)
# § 301 also exists in CSP (civil procedure) and Trestny zakon.
# In commercial decisions § 301 CSP is rare but I want to be safe.
# If it finds "§ 301 Trestneho", throw it away.
REGEX_EXCLUDE_CONTEXT = re.compile(r'301\s+(trest|civiln|súdneho|správn)', re.IGNORECASE)


class ModeracnePravoScraper:
    def __init__(self):
        self.base_url = "https://www.nsud.sk/ws/opendata.php"
        self.file_base_url = "https://www.nsud.sk/data/att/"

        if not os.path.exists(PDF_DIR):
            os.makedirs(PDF_DIR)

        self.results = []
        # load existing data so I can continue interrupted run
        if os.path.exists(OUTPUT_CSV):
            try:
                self.results = pd.read_csv(OUTPUT_CSV, sep='|').to_dict('records')
                print(f"--- Loaded {len(self.results)} existing decisions ---")
            except:
                pass

    def get_metadata(self, decision_id):
        # download metadata about decision
        try:
            params = {'getDecision': '', 'id': decision_id}
            # 5 sec timeout is enough, if no response just move on
            r = requests.get(self.base_url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data: return data[0]
                if isinstance(data, dict) and 'cislo' in data: return data
            return None
        except:
            return None

    def analyze_pdf(self, pdf_url):
        # Download PDF to RAM and return text if it contains § 301
        try:
            # URL fix (sometimes domain is missing)
            full_url = pdf_url if pdf_url.startswith('http') else self.file_base_url + pdf_url.lstrip('/')

            r = requests.get(full_url, timeout=10)
            if r.status_code != 200:
                return False, None, None

            # Read PDF via pypdf (in memory, not yet to disk)
            with io.BytesIO(r.content) as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + " "

            # --- VALIDATION LOGIC ---

            # 1. look for § 301
            match = REGEX_TARGET.search(text)
            if match:
                # 2. context window check
                # Check text around match - is it not Trestny zakon?
                start_idx = max(0, match.start() - 50)
                end_idx = min(len(text), match.end() + 100)
                context_snippet = text[start_idx:end_idx]

                # If we see "Trestneho" or "Civilneho mimosporoveho" near 301, ignore.
                # Most judges just write "§ 301" which in commercial case means
                # ObchZ, so we take it.
                if REGEX_EXCLUDE_CONTEXT.search(context_snippet):
                    return False, None, None

                return True, r.content, text

            return False, None, None

        except Exception as e:
            # print(f"PDF error: {e}")
            return False, None, None

    def save_match(self, meta, pdf_bytes, full_text, decision_id):
        # save the result
        spisova_znacka = str(meta.get('cislo', 'nezname')).replace('/', '_')
        clean_date = str(meta.get('datum', 'neznamy')).replace('.', '_')

        # filename: ID_SPIS.pdf
        filename = f"{decision_id}_{spisova_znacka}.pdf"
        file_path = os.path.join(PDF_DIR, filename)

        # save PDF to disk
        with open(file_path, 'wb') as f:
            f.write(pdf_bytes)

        # find context for excel (so I can see the sentence)
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
            'merito': meta.get('merito', ''),  # what it was about (court keywords)
            'najdeny_kontext_301': snippet,  # sentence around § 301
            'cesta_k_pdf': file_path,
            'url_na_web': f"https://www.nsud.sk{meta.get('subor')}"
        }

        self.results.append(record)

        # write to CSV continuously (overwrite mode - always rewrites with new data)
        # using '|' as separator because legal texts have commas
        df = pd.DataFrame(self.results)
        df.to_csv(OUTPUT_CSV, index=False, sep='|', encoding='utf-8')

    def run(self):
        print(f"=== STARTING SCRAPER: MODERATION RIGHT (§ 301) ===")
        print(f"Looking in IDs: {START_ID} -> {MIN_ID}")
        print(f"Filter: commercial division + text '§ 301'")

        count_saved = 0

        # tqdm progress bar
        for current_id in tqdm(range(START_ID, MIN_ID, -1), desc="Analyzing"):

            # A) get metadata
            meta = self.get_metadata(current_id)
            if not meta:
                continue

            # B) FILTER: only commercial cases
            # Check if its commercial division (code 2) OR if spisova znacka contains "Ob"
            kolegium = str(meta.get('kolegium', '')).lower()
            spis = str(meta.get('cislo', ''))

            is_commercial = ('2' in kolegium) or \
                            ('ob' in spis.lower()) or \
                            ('cob' in spis.lower())

            if not is_commercial:
                continue  # skip civil and criminal (saves 80% time)

            # C) FILTER: PDF content
            pdf_url = meta.get('subor')
            if pdf_url:
                found, pdf_bytes, text = self.analyze_pdf(pdf_url)

                if found:
                    self.save_match(meta, pdf_bytes, text, current_id)
                    count_saved += 1
                    tqdm.write(f" FOUND! ID {current_id} ({spis}) -> contains § 301")

            # short server pause
            time.sleep(0.05)

        print(f"\n=== DONE ===")
        print(f"Total saved decisions: {count_saved}")
        print(f"Data in: {OUTPUT_CSV}")


if __name__ == "__main__":
    # need: pip install requests pandas pypdf tqdm
    scraper = ModeracnePravoScraper()
    scraper.run()
