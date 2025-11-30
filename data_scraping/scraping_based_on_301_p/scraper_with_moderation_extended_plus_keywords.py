import requests
import pandas as pd
import re
import time
import os
import io
from tqdm import tqdm
from pypdf import PdfReader

# --- CONFIGURATION ---
OUTPUT_CSV = "dataset_obchod_pokuty_plus_keywords.csv"
PDF_DIR = "../data/rozhodnutia_obchod_plus_kewords"

# ID range - recommend going in smaller blocks if it crashes
START_ID = 245000
MIN_ID = 10000

# --- REGEXES ---

# 1. § 301 (Commercial Code)
REGEX_PARAGRAPH = re.compile(
    r'(?:§|par|ust|odst|čl|bod|zm|S|s|6|z|g)\.?\s*301',
    re.IGNORECASE
)

# 2. KEYWORDS (Contractual penalty + moderation)
REGEX_STRICT_ZMLUVNA = re.compile(
    r'((?:neprimeran|neúmern|znížen|znížil|moderač|primeran)[a-ž]*)\W+(?:\w+\W+){0,25}?zmluvn[a-ž]*\s+pokut[a-z]*|'
    r'zmluvn[a-ž]*\s+pokut[a-z]*\W+(?:\w+\W+){0,25}?((?:neprimeran|neúmern|znížen|znížil|moderač|primeran)[a-ž]*)',
    re.IGNORECASE
)


class FinalScraper:
    def __init__(self):
        self.base_url = "https://www.nsud.sk/ws/opendata.php"
        self.file_base_url = "https://www.nsud.sk/data/att/"

        if not os.path.exists(PDF_DIR):
            os.makedirs(PDF_DIR)

        self.results = []
        if os.path.exists(OUTPUT_CSV):
            try:
                self.results = pd.read_csv(OUTPUT_CSV, sep='|').to_dict('records')
                print(f"Načítaných {len(self.results)} existujúcich záznamov.")
            except:
                pass

    def clean_text(self, text):
        if not text: return ""
        return " ".join(text.split())

    def get_metadata(self, decision_id):
        # We pull clean data by ID, without filters in the URL (caused problems)
        try:
            params = {'getDecision': '', 'id': decision_id}
            r = requests.get(self.base_url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data: return data[0]
                return data
            return None
        except:
            return None

    def is_commercial_case(self, meta):
        """
        Decides if it is a commercial case based on metadata.
        Checks: College, Area and mainly FILE REFERENCE (Obdo, Obo...).
        """
        txt_kolegium = str(meta.get('kolegium', '')).lower()
        txt_oblast = str(meta.get('oblast', '')).lower()
        txt_spis = str(meta.get('cislo', '')).lower()  # E.g. 1 Obdo 55/2010

        # 1. Clear designation in metadata
        if 'obchod' in txt_kolegium or 'obchod' in txt_oblast:
            return True

        # 2. File reference (Strongest indicator for Supreme Court SR)
        # Obdo = Commercial appellate review, Obo = Commercial appeal, etc.
        if 'obdo' in txt_spis or 'obo' in txt_spis or 'cob' in txt_spis:
            return True

        # Sometimes there is just 'ob', but watch out for 'dob' (criminal).
        # So we look for 'ob ' or '/ob'.
        if 'ob ' in txt_spis or '/ob' in txt_spis:
            return True

        return False

    def analyze_pdf(self, pdf_url):
        try:
            full_url = pdf_url if pdf_url.startswith('http') else self.file_base_url + pdf_url.lstrip('/')
            r = requests.get(full_url, timeout=10)
            if r.status_code != 200: return False, None, None, None, None

            text_content = ""
            with io.BytesIO(r.content) as f:
                try:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted: text_content += extracted + " "
                except:
                    return False, None, None, None, None

            if len(text_content.strip()) < 50:
                return False, "SCAN", None, None, None  # Ignoring scans for now, if you want

            normalized = self.clean_text(text_content)

            # --- SEARCHING ---

            # 1. § 301
            for match in REGEX_PARAGRAPH.finditer(normalized):
                # Check for false positives (301/2005)
                end_pos = match.end()
                if end_pos < len(normalized):
                    next_char = normalized[end_pos]
                    if next_char in ['/', '.'] or next_char.isdigit():
                        # One more check - if it is a dot followed by a space, it is OK (§ 301. )
                        if next_char == '.' and (end_pos + 1 < len(normalized)) and normalized[end_pos + 1] == ' ':
                            pass
                        elif next_char == '/' or next_char.isdigit():
                            continue

                context = normalized[max(0, match.start() - 80):min(len(normalized), match.end() + 80)]
                return True, "MATCH_PARAGRAF_301", r.content, context, "PARAGRAF"

            # 2. KEYWORDS
            keyword_match = REGEX_STRICT_ZMLUVNA.search(normalized)
            if keyword_match:
                start_c = max(0, keyword_match.start() - 80)
                end_c = min(len(normalized), keyword_match.end() + 80)
                context = normalized[start_c:end_c]
                return True, "MATCH_STRICT_KEYWORDS", r.content, context, "SLOVNE"

            return False, "No_Match", None, None, None

        except Exception as e:
            return False, None, None, None, None

    def save_match(self, meta, pdf_bytes, reason, snippet, match_type, decision_id):
        spis = str(meta.get('cislo', 'nezname')).replace('/', '_')
        filename = f"{decision_id}_{match_type}_{spis}.pdf"
        file_path = os.path.join(PDF_DIR, filename)

        with open(file_path, 'wb') as f:
            f.write(pdf_bytes)

        record = {
            'id': decision_id,
            'typ_zhody': match_type,
            'dovod_detail': reason,
            'spis': meta.get('cislo'),
            'datum': meta.get('datum'),
            'sud': meta.get('sud'),
            'oblast': meta.get('oblast'),
            'snippet': snippet,
            'file': filename
        }
        self.results.append(record)
        pd.DataFrame(self.results).to_csv(OUTPUT_CSV, index=False, sep='|', encoding='utf-8')

    def run(self):
        print("=== FINAL SCRAPER (OPRAVENÝ) ===")
        print("Stratégia: Ťahám ID -> Kontrolujem či je to OBCHOD (podľa značky spisu Obdo/Ob) -> Hľadám text.")

        matches = 0
        checked_commercial = 0

        # TQDM progress bar
        pbar = tqdm(range(START_ID, MIN_ID, -1))

        for current_id in pbar:
            meta = self.get_metadata(current_id)
            if not meta: continue

            # --- 1. FILTER: Is it a commercial case? ---
            if not self.is_commercial_case(meta):
                # If it is not commercial, ignore and move on
                continue

            checked_commercial += 1

            # --- 2. PDF ANALYSIS ---
            pdf_url = meta.get('subor')
            if pdf_url:
                save, reason, pdf_bytes, snippet, match_type = self.analyze_pdf(pdf_url)

                if save:
                    matches += 1
                    self.save_match(meta, pdf_bytes, reason, snippet, match_type, current_id)

            # Update description in progress bar
            pbar.set_description(f"Obchodné: {checked_commercial} | Nájdené zhody: {matches}")

            time.sleep(0.01)


if __name__ == "__main__":
    scraper = FinalScraper()
    scraper.run()