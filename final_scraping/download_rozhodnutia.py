import requests
import json
import os
import time
import pandas as pd
import re

# --- KONFIGURÁCIA ---
BASE_API_URL = "https://obcan.justice.sk/pilot/api/ress-isu-service"
SEARCH_ENDPOINT = "/v1/rozhodnutie"
OUTPUT_DIR = "stiahnute_podla_csv_final"
TIMEOUT = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "sk-SK,sk;q=0.9,en-US;q=0.8,en;q=0.7"
}


def normalize_court_name(csv_name):
    """Zjednotí názvy súdov."""
    mapping = {
        "KS Banská Bystrica": "Krajský súd Banská Bystrica",
        "KS Bratislava": "Krajský súd Bratislava",
        "KS Košice": "Krajský súd Košice",
        "KS Nitra": "Krajský súd Nitra",
        "KS Prešov": "Krajský súd Prešov",
        "KS Trenčín": "Krajský súd Trenčín",
        "KS Trnava": "Krajský súd Trnava",
        "KS Žilina": "Krajský súd Žilina",
        "NS SR": "Najvyšší súd SR",
        "ÚS SR": "Ústavný súd SR"
    }
    return mapping.get(csv_name, csv_name)


def generate_id_variants(case_id):
    """
    Vygeneruje rôzne formáty spisovej značky.
    Vstup: 14Co/371/2012
    Výstup: ['14Co/371/2012', '14 Co 371/2012', '14Co 371/2012']
    """
    variants = [case_id]  # Originál

    # Skúsime pridať medzery medzi text a čísla (častý formát na súdoch)
    # Regex rozdelí "14Co" a "371/2012"
    match = re.match(r"([0-9]+)([a-zA-Z]+)/(.+)", case_id)
    if match:
        senat, agenda, zvysok = match.groups()
        # Variant: 14 Co 371/2012
        variants.append(f"{senat} {agenda} {zvysok}")
        # Variant: 14Co 371/2012
        variants.append(f"{senat}{agenda} {zvysok}")

    return list(set(variants))


def download_file(url, folder, filename):
    try:
        if not url.startswith("http"):
            url = f"https://obcan.justice.sk{url}"

        with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
            if r.status_code == 200:
                if "." not in filename[-5:]:
                    ct = r.headers.get("Content-Type", "").lower()
                    if "pdf" in ct:
                        filename += ".pdf"
                    elif "word" in ct:
                        filename += ".docx"
                    elif "zip" in ct:
                        filename += ".zip"
                    else:
                        filename += ".pdf"

                file_path = os.path.join(folder, filename)
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"            ✅ [SÚBOR] {filename}")
                return True
    except:
        pass
    return False


def get_decision_detail(guid):
    try:
        resp = requests.get(f"{BASE_API_URL}/v1/rozhodnutie/{guid}", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


def find_and_download(row_index, court_raw, case_id):
    court_official = normalize_court_name(court_raw)
    clean_id = str(case_id).strip()

    print(f"\n[{row_index}] 🔎 Hľadám: {clean_id} ({court_official})")

    # Stratégie vyhľadávania (názov parametra v URL)
    search_params = ['spisovaZnacka', 'znacka', 'q']

    # Varianty formátu značky (s medzerami, bez medzier)
    id_variants = generate_id_variants(clean_id)

    found_decision = None

    # --- CYKLUS HĽADANIA ---
    for param_name in search_params:
        if found_decision: break

        for variant in id_variants:
            if found_decision: break

            # Skúsime API call
            params = {
                param_name: f'"{variant}"' if param_name == 'q' else variant,
                "size": 20
            }

            # Ak používame parameter 'spisovaZnacka', môžeme pridať aj súd pre presnosť
            if param_name == 'spisovaZnacka':
                # Niektoré API podporujú filter={"sud": "..."} v GET, ale skúsime radšej bez, aby sme to nekomplikovali
                pass

            try:
                # print(f"   ... skúšam ?{param_name}={variant}") # Debug výpis
                resp = requests.get(f"{BASE_API_URL}{SEARCH_ENDPOINT}", headers=HEADERS, params=params, timeout=10)

                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get('rozhodnutieList', [])

                    # Overenie kandidátov
                    for cand in candidates:
                        cand_znacka = cand.get('spisovaZnacka', '').replace(" ", "").lower()
                        target_znacka = clean_id.replace(" ", "").lower()

                        cand_sud = cand.get('sud', {}).get('nazov', '').lower()
                        target_sud_part = court_official.replace("Krajský súd ", "").lower()

                        # Musí sedieť značka AJ súd
                        if cand_znacka == target_znacka and target_sud_part in cand_sud:
                            found_decision = cand
                            print(f"   🎯 NAŠIEL SOM! (Stratégia: {param_name}={variant})")
                            break
            except Exception as e:
                pass

    if not found_decision:
        print(f"   ❌ Nenájdené ani jednou stratégiou.")
        return

    # --- SŤAHOVANIE ---
    guid = found_decision.get('guid')

    safe_court = court_official.replace(" ", "_")
    safe_id = clean_id.replace("/", "_")
    case_folder = os.path.join(OUTPUT_DIR, safe_court, safe_id)

    if not os.path.exists(case_folder):
        os.makedirs(case_folder)

    detail = get_decision_detail(guid)
    if detail:
        # Metadáta
        with open(os.path.join(case_folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=4)

        # Dokumenty
        docs = []
        if 'dokument' in detail and detail['dokument']:
            val = detail['dokument']
            docs.extend(val if isinstance(val, list) else [val])
        if 'dokumenty' in detail: docs.extend(detail['dokumenty'])
        if 'prilohy' in detail: docs.extend(detail['prilohy'])

        if docs:
            print(f"      📥 Sťahujem {len(docs)} príloh...")
            for i, doc in enumerate(docs):
                doc_name = doc.get('nazov', f"doc_{i}")
                safe_name = "".join([c for c in doc_name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()

                url = doc.get('url')
                if not url and doc.get('id'):
                    url = f"/v1/rozhodnutie/dokument/{doc.get('id')}/stiahnut"

                if url:
                    download_file(url, case_folder, safe_name)
        else:
            # Text fallback
            full_text = detail.get('text') or detail.get('obsah')
            if full_text:
                with open(os.path.join(case_folder, "text.txt"), "w", encoding="utf-8") as f:
                    f.write(full_text)
                print("      📄 Uložený text (bez PDF).")
            else:
                print("      ⚠️ Rozhodnutie je prázdne (žiadne prílohy).")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("--- SŤAHOVANIE PODĽA CSV (Multi-Strategy) ---")

    try:
        df = pd.read_csv('rozhodnutia_raw_list.csv')
    except:
        print("Chýba súbor rozhodnutia_raw_list.csv")
        return

    for index, row in df.iterrows():
        find_and_download(index, row['court_name'], row['case_id'])
        time.sleep(0.2)


if __name__ == "__main__":
    main()