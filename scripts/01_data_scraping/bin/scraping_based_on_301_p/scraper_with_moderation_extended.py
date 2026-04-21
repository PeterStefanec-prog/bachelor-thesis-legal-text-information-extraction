import requests
import pandas as pd
import re
import time
import os
import io
from tqdm import tqdm
from pypdf import PdfReader

# Better version of § 301 scraper. I made the regex stricter so I get
# less false positives. Also added more anti-filters for stuff that
# pretends to be § 301 ObchZ but is actualy something else.

# --- CONFIG ---
OUTPUT_CSV = "dataset_obchodne_moderacne_pravo_extended.csv"
PDF_DIR = "../data/rozhodnutia_obchodne_extended"

START_ID = 245000
MIN_ID = 10000

# --- FINAL REGEXES (VERY STRICT) ---

# FIX: moved (?i) from middle of string to re.IGNORECASE flag
# 1. searching § 301
REGEX_STRICT_301 = re.compile(
    r'(?:^|\s)(?:§|par|ust|odst|čl|bod|zm|S|s|6|z|g)\.?\s*301(?![\d.,])',
    re.IGNORECASE
)

# 2. ANTI-FILTER: procedural law
REGEX_EXCLUDE_SUFFIX = re.compile(
    r'301\s*(CSP|Civil|Trest|Správ|Súd|Z\.?z|Tr\.?|T\.?z)',
    re.IGNORECASE
)

# 3. COMMERCIAL CONTEXT
REGEX_COMMERCIAL_CONTEXT = re.compile(
    r'(obchodn|513\/1991|obch\.?\s*zák)',
    re.IGNORECASE
)


class UltimateScraper:
    def __init__(self):
        self.base_url = "https://www.nsud.sk/ws/opendata.php"
        self.file_base_url = "https://www.nsud.sk/data/att/"

        if not os.path.exists(PDF_DIR):
            os.makedirs(PDF_DIR)

        self.results = []
        if os.path.exists(OUTPUT_CSV):
            try:
                self.results = pd.read_csv(OUTPUT_CSV, sep='|').to_dict('records')
                print(f"--- Loaded {len(self.results)} existing records ---")
            except:
                pass

    def clean_text(self, text):
        if not text: return ""
        text = text.replace('\xa0', ' ').replace('\t', ' ')
        return " ".join(text.split())

    def get_metadata(self, decision_id):
        try:
            params = {'getDecision': '', 'id': decision_id}
            # also handle SSL errors just in case
            r = requests.get(self.base_url, params=params, timeout=5)
            if r.status_code == 200:
                try:
                    data = r.json()
                    if isinstance(data, list) and data: return data[0]
                    return data
                except ValueError:
                    return None
            return None
        except:
            return None

    def check_metadata_strict(self, meta):
        kolegium = str(meta.get('kolegium', ''))
        spis = str(meta.get('cislo', '')).lower()
        oblast = str(meta.get('oblast', '')).lower()

        # 1. IMMEDIATE REJECTION
        if '3' in kolegium or '1' in kolegium or '4' in kolegium or '5' in kolegium or '8' in kolegium:
            return -1
        if 'ndc' in spis or 'cdo' in spis or 'tdo' in spis or 'sž' in spis:
            return -1

        # 2. IMMEDIATE APPROVAL
        if '2' in kolegium: return 1
        if 'obdo' in spis or 'obo' in spis or 'cob' in spis: return 1
        if 'obchod' in oblast: return 1

        return 0

    def analyze_pdf(self, pdf_url, force_commercial_check=False):
        try:
            full_url = pdf_url if pdf_url.startswith('http') else self.file_base_url + pdf_url.lstrip('/')
            r = requests.get(full_url, timeout=10)
            if r.status_code != 200: return False, None, None, None

            text_content = ""
            with io.BytesIO(r.content) as f:
                try:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted: text_content += extracted + " "
                except:
                    # PDF might be corrupt or encrypted
                    return False, None, None, None

            if len(text_content.strip()) < 50:
                if force_commercial_check: return False, "SCAN_UNCERTAIN", None, None
                return True, "SCAN_MANUAL_CHECK", r.content, "SCAN"

            normalized = self.clean_text(text_content)

            if force_commercial_check:
                if not REGEX_COMMERCIAL_CONTEXT.search(normalized):
                    return False, "TEXT_NOT_COMMERCIAL", None, None

            # --- CORE FILTER ---
            matches = list(REGEX_STRICT_301.finditer(normalized))

            for match in matches:
                snippet_check = normalized[max(0, match.start()):min(len(normalized), match.end() + 25)]
                if REGEX_EXCLUDE_SUFFIX.search(snippet_check):
                    continue

                context = normalized[max(0, match.start() - 100):min(len(normalized), match.end() + 100)]
                return True, "MATCH_CLEAN", r.content, context

            return False, "NO_MATCH", None, None

        except Exception as e:
            # print(f"PDF analysis error: {e}")
            return False, None, None, None

    def save_match(self, meta, pdf_bytes, reason, snippet, decision_id):
        # sanitize filename for Windows (forbidden chars)
        safe_spis = str(meta.get('cislo', 'nezname')).replace('/', '_').replace('\\', '_').replace(':', '')
        filename = f"{decision_id}_{safe_spis}.pdf"
        file_path = os.path.join(PDF_DIR, filename)

        try:
            with open(file_path, 'wb') as f:
                f.write(pdf_bytes)
        except Exception as e:
            print(f"Error saving file {filename}: {e}")
            return

        record = {
            'id': decision_id,
            'dovod': reason,
            'spis': meta.get('cislo'),
            'datum': meta.get('datum'),
            'sud': meta.get('sud'),
            'snippet': snippet,
            'file': filename
        }
        self.results.append(record)

        # save every time so I dont lose data
        try:
            df = pd.DataFrame(self.results)
            df.to_csv(OUTPUT_CSV, index=False, sep='|', encoding='utf-8')
        except:
            pass

    def run(self):
        print("=== ULTIMATE SCRAPER (FIXED REGEX) ===")
        print("1. metadata filter: ignore civil (3) and criminal (1)")
        print("2. money filter: ignore 166.301,53 etc")
        print("3. CSP filter: ignore § 301 CSP")

        for current_id in tqdm(range(START_ID, MIN_ID, -1)):
            try:
                meta = self.get_metadata(current_id)
                if not meta: continue

                status = self.check_metadata_strict(meta)
                if status == -1: continue

                should_force_check = (status == 0)

                pdf_url = meta.get('subor')
                if pdf_url:
                    save, reason, pdf_bytes, snippet = self.analyze_pdf(pdf_url,
                                                                        force_commercial_check=should_force_check)
                    if save:
                        self.save_match(meta, pdf_bytes, reason, snippet, current_id)

                time.sleep(0.02)

            except KeyboardInterrupt:
                print("Stopping...")
                break
            except Exception as e:
                # catch unexpected errors so loop keeps going
                # print(f"Error at ID {current_id}: {e}")
                continue


if __name__ == "__main__":
    scraper = UltimateScraper()
    scraper.run()
