import requests
import json
import os
import time
import pandas as pd
import re

# --- CONFIGURATION ---
BASE_API_URL = "https://obcan.justice.sk/pilot/api/ress-isu-service"
SEARCH_ENDPOINT = "/v1/rozhodnutie"
OUTPUT_DIR = "stiahnute_podla_csv_final_03"
TIMEOUT = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "sk-SK,sk;q=0.9,en-US;q=0.8,en;q=0.7"
}


def normalize_court_name(csv_name):
    """
    Fix court names because API needs specific format.
    Example: 'KS Banská Bystrica' -> 'Krajský súd Banská Bystrica'
    """
    mapping = {
        "KS Banská Bystrica": "Krajský súd Banská Bystrica",
        "KS Bratislava": "Krajský súd Bratislava",
        "KS Košice": "Krajský súd Košice",
        "KS Nitra": "Krajský súd Nitra",
        "KS Prešov": "Krajský súd Prešov",
        "KS Trenčín": "Krajský súd Trenčín",
        "KS Trnava": "Krajský súd Trnava",
        "KS Žilina": "Krajský súd Žilina",
        "NS SR": "Najvyšší súd Slovenskej republiky",
        "ÚS SR": "Ústavný súd Slovenskej republiky"
    }
    return mapping.get(csv_name, csv_name)


def generate_id_variants(case_id):
    """
    Create different ID versions to help the search.
    Input: 14Co/371/2012
    Output: ['14Co/371/2012', '14 Co 371/2012', '14Co 371/2012']
    """
    clean_id = str(case_id).strip().replace("\\", "/")
    variants = {clean_id}

    # 1. Remove all spaces just in case
    no_spaces = clean_id.replace(" ", "")
    variants.add(no_spaces)

    # 2. Try to split numbers and text to add spaces smartly
    # Pattern: (number)(text)/(rest)
    match = re.search(r"(\d+)\s*([a-zA-Z]+)\s*[/\s]*\s*(.+)", clean_id)
    if match:
        senat, agenda, zvysok = match.groups()
        variants.add(f"{senat}{agenda}/{zvysok}")  # Standard format # 14Co/371/2012
        variants.add(f"{senat} {agenda} {zvysok}")  # With spaces    # 14 Co 371/2012
        variants.add(f"{senat}{agenda} {zvysok}")  # Combined        # 14Co 371/2012
        variants.add(f"{senat} {agenda} {zvysok.replace('/', '/ ')}")  # Extra spaces

    return list(variants)


def download_file(url, folder, filename):
    """
    Download the file. Add .pdf or .docx extension if missing.
    """
    try:
        if not url.startswith("http"):
            url = f"https://obcan.justice.sk{url}"

        with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
            if r.status_code == 200:
                # Check content type to guess the extension
                final_filename = filename
                if "." not in filename[-5:]:
                    ct = r.headers.get("Content-Type", "").lower()
                    if "pdf" in ct:
                        final_filename += ".pdf"
                    elif "word" in ct or "officedocument" in ct:
                        final_filename += ".docx"
                    elif "zip" in ct:
                        final_filename += ".zip"
                    else:
                        final_filename += ".pdf"  # Default to PDF

                file_path = os.path.join(folder, final_filename)

                # Write file in chunks (better for memory)
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"            ✅ [SÚBOR] {final_filename}")
                return True
    except Exception as e:
        print(f"            ⚠️ Chyba sťahovania: {e}")
    return False


def get_decision_detail(guid):
    """
    Get full details (JSON) about the decision using its ID (guid).
    """
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

    print(f"\n[{row_index}] 🔎 Hľadám: {clean_id} ({court_raw})")

    # Skip Constitutional Court (not in this database)
    if "Ústavný súd" in court_official:
        print("   ⚠️ Ústavný súd SR nie je v tejto DB. Preskakujem.")
        return

    # --- SEARCH LOGIC ---
    # We only use 'spisovaZnacka' now to be efficient.
    id_variants = generate_id_variants(clean_id)
    found_decision = None

    for variant in id_variants:
        if found_decision: break

        # Prepare params for the API request
        params = {
            "spisovaZnacka": variant,
            "size": 20
        }

        # Add filter for Supreme Court if needed
        if "Najvyšší súd" in court_official:
            params["sud"] = "Najvyšší súd Slovenskej republiky"

        try:
            # Send request to API
            resp = requests.get(f"{BASE_API_URL}{SEARCH_ENDPOINT}", headers=HEADERS, params=params, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('rozhodnutieList', [])

                # Check results to find the correct match
                for cand in candidates:
                    # Clean up strings for comparison
                    cand_znacka = cand.get('spisovaZnacka', '').replace(" ", "").lower()
                    target_znacka_clean = clean_id.replace(" ", "").replace("/", "").lower()
                    cand_znacka_clean = cand_znacka.replace("/", "")

                    cand_sud = cand.get('sud', {}).get('nazov', '').lower()

                    # Verify court name
                    court_match = False
                    if "najvyšší" in court_official.lower() and "najvyšší" in cand_sud:
                        court_match = True
                    elif court_official.replace("Krajský súd ", "").lower() in cand_sud:
                        court_match = True

                    # If ID and Court match, we found it!
                    if target_znacka_clean in cand_znacka_clean and court_match:
                        found_decision = cand
                        print(f"   🎯 NAŠIEL SOM! ({cand.get('spisovaZnacka')})")
                        break
        except Exception:
            pass

    if not found_decision:
        print(f"   ❌ Nenájdené.")
        return

    # --- DOWNLOAD LOGIC ---
    guid = found_decision.get('guid')

    # Create folder structure
    safe_court = court_raw.replace(" ", "_")
    safe_id = clean_id.replace("/", "_").replace(" ", "_")
    case_folder = os.path.join(OUTPUT_DIR, safe_court, safe_id)

    if not os.path.exists(case_folder):
        os.makedirs(case_folder)

    # Get details
    detail = get_decision_detail(guid)
    if detail:
        # Save metadata JSON
        detail["_csv_metadata"] = {"court": court_raw, "id": case_id, "topic": row_index}
        with open(os.path.join(case_folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=4)

        # Collect all attachments
        docs = []
        if 'dokument' in detail and detail['dokument']:
            val = detail['dokument']
            docs.extend(val if isinstance(val, list) else [val])
        if 'dokumenty' in detail: docs.extend(detail['dokumenty'])
        if 'prilohy' in detail: docs.extend(detail['prilohy'])

        if docs:
            print(f"      📥 Sťahujem {len(docs)} príloh...")
            for i, doc in enumerate(docs):
                # 1. Get original name
                raw_name = doc.get('nazov', 'dokument')

                # 2. Clean the filename
                clean_name = re.sub(r'[^\w\-\. ]', '_', raw_name).strip()
                if not clean_name: clean_name = "dokument"

                # 3. Add prefix to keep order (00_, 01_)
                final_name = f"{i:02d}_{clean_name}"

                url = doc.get('url')
                if not url and doc.get('id'):
                    url = f"/v1/rozhodnutie/dokument/{doc.get('id')}/stiahnut"

                if url:
                    download_file(url, case_folder, final_name)
        else:
            # Fallback: Save plain text if no PDF available
            full_text = detail.get('text') or detail.get('obsah')
            if full_text:
                with open(os.path.join(case_folder, "text.txt"), "w", encoding="utf-8") as f:
                    f.write(full_text)
                print("      📄 Uložený čistý text.")
            else:
                print("      ⚠️ Rozhodnutie je prázdne.")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("--- DOWNLOADING COURT DECISIONS ---")
    csv_file = 'rozhodnutia_raw_list.csv'

    if not os.path.exists(csv_file):
        print(f"ERROR: Create file '{csv_file}' first!")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file, quotechar='"', skipinitialspace=True)
    except Exception as e:
        print(f"CSV Error: {e}")
        return

    # Loop through all rows in CSV
    for index, row in df.iterrows():
        c_name = row.get('court_name')
        c_id = row.get('case_id')
        if pd.isna(c_name) or pd.isna(c_id): continue

        find_and_download(index, c_name, c_id)

        # Sleep a bit to be polite to the server
        time.sleep(0.5)


if __name__ == "__main__":
    main()