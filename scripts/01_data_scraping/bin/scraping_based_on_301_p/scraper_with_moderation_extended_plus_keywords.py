import requests
import pandas as pd
import re
import time
import os
import io
from tqdm import tqdm
from pypdf import PdfReader

# Latest version of § 301 scraper. I added a second match path - even
# if PDF doesnt explicitly say "§ 301" but talks about "neprimerana
# zmluvna pokuta" or "moderacne pravo", I still keep it. Catches more
# cases this way.

# --- CONFIG ---
OUTPUT_CSV = "dataset_obchod_pokuty_plus_keywords.csv"
PDF_DIR = "../data/rozhodnutia_obchod_plus_kewords"

# ID range - if it crashes go in smaller blocks
START_ID = 245000
MIN_ID = 10000

# --- REGEXES ---

# 1. § 301 (ObchZ)
REGEX_PARAGRAPH = re.compile(
    r'(?:§|par|ust|odst|čl|bod|zm|S|s|6|z|g)\.?\s*301',
    re.IGNORECASE
)

# 2. KEYWORDS (contractual penalty + moderation)
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
                print(f"Loaded {len(self.results)} existing records.")
            except:
                pass

    def clean_text(self, text):
        if not text: return ""
        return " ".join(text.split())

    def get_metadata(self, decision_id):
        # I just pull clean data by ID, no filters in URL (those caused problems)
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
        # Is it commercial case based on metadata?
        # Check: kolegium, oblast and mainly SPISOVA ZNACKA (Obdo, Obo...)
        txt_kolegium = str(meta.get('kolegium', '')).lower()
        txt_oblast = str(meta.get('oblast', '')).lower()
        txt_spis = str(meta.get('cislo', '')).lower()  # e.g. "1 Obdo 55/2010"

        # 1. clear marker in metadata
        if 'obchod' in txt_kolegium or 'obchod' in txt_oblast:
            return True

        # 2. spisova znacka (strongest indicator for NS SR)
        # Obdo = commercial appellate review, Obo = commercial appeal etc.
        if 'obdo' in txt_spis or 'obo' in txt_spis or 'cob' in txt_spis:
            return True

        # Sometimes there is just 'ob', but careful - 'dob' is criminal.
        # So I look for 'ob ' or '/ob'.
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
                return False, "SCAN", None, None, None  # ignore scans for now

            normalized = self.clean_text(text_content)

            # --- SEARCHING ---

            # 1. § 301
            for match in REGEX_PARAGRAPH.finditer(normalized):
                # check for false positives like "301/2005"
                end_pos = match.end()
                if end_pos < len(normalized):
                    next_char = normalized[end_pos]
                    if next_char in ['/', '.'] or next_char.isdigit():
                        # one more check - if its a dot followed by space, OK ("§ 301. ")
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
        print("=== FINAL SCRAPER (FIXED) ===")
        print("Strategy: pull ID -> check if its COMMERCIAL (by spis Obdo/Ob) -> search text")

        matches = 0
        checked_commercial = 0

        # tqdm progress bar
        pbar = tqdm(range(START_ID, MIN_ID, -1))

        for current_id in pbar:
            meta = self.get_metadata(current_id)
            if not meta: continue

            # --- 1. FILTER: is it commercial? ---
            if not self.is_commercial_case(meta):
                # if not commercial, ignore and continue
                continue

            checked_commercial += 1

            # --- 2. PDF ANALYSIS ---
            pdf_url = meta.get('subor')
            if pdf_url:
                save, reason, pdf_bytes, snippet, match_type = self.analyze_pdf(pdf_url)

                if save:
                    matches += 1
                    self.save_match(meta, pdf_bytes, reason, snippet, match_type, current_id)

            # update progress bar description
            pbar.set_description(f"Commercial: {checked_commercial} | Found matches: {matches}")

            time.sleep(0.01)


if __name__ == "__main__":
    scraper = FinalScraper()
    scraper.run()
