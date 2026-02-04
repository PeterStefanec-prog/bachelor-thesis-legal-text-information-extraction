import requests
import json
import os
import time
import pandas as pd
import re
from pathlib import Path

# --- CONFIGURATION ---
BASE_API_URL = "https://obcan.justice.sk/pilot/api/ress-isu-service"
SEARCH_ENDPOINT = "/v1/rozhodnutie"
OUTPUT_DIR = "data/01_raw_pdfs"
# Zistí cestu k priečinku, kde je tento skript
current_dir = Path(__file__).resolve().parent

OUTPUT_DIR = current_dir.parent.parent / "data" / "01_raw_pdfs"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT = 30

# Headers to make our script look like a normal web browser.
# If we don't set this, the server might block us.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "sk-SK,sk;q=0.9,en-US;q=0.8,en;q=0.7"
}


# --- COLORS FOR TERMINAL OUTPUT ---
# Simple class to make the text colored in the command line.
# Green for success, Red for failure, Blue for info.
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'


def normalize_court_name(csv_name):
    """
    We need to fix the court names because the CSV has short names like 'KS Zilina',
    but the API is strict and wants the full official name 'Krajský súd Žilina'.
    It's basically a simple dictionary translation.
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
    This is the tricky part. The database is messy.
    Regional courts usually like slashes (14Cob/20/2019).
    The Supreme Court (NS SR) usually likes spaces (1 Cdo 10 2020).

    This function creates a LIST of all possible ways to write the ID
    so we can try them one by one until we find a match.
    """
    clean_id = str(case_id).strip().replace("\\", "/")
    variants = {clean_id}

    # 1. Try removing all spaces.
    # Good for standard inputs like "14Cob/20/2019"
    no_spaces = clean_id.replace(" ", "")
    variants.add(no_spaces)

    # 2. Use Regex to split the ID into parts.
    # Example: If we have "1Cdo/123/2020", we split it into:
    # Senate: "1", Agenda: "Cdo", The rest: "123/2020"
    match = re.search(r"^(\d+)\s*([a-zA-Z]+)\s*[/\s]*\s*(.+)$", clean_id)
    if match:
        senat, agenda, zvysok = match.groups()

        # Format A: Compact with slash (Standard for Regional Courts)
        # Result: 1Cdo/123/2020
        variants.add(f"{senat}{agenda}/{zvysok}")

        # Format B: Separated by spaces (Standard for Supreme Court)
        # Result: 1 Cdo 123/2020
        variants.add(f"{senat} {agenda} {zvysok}")

        # Format C: Spaces everywhere, no slash
        # Result: 1 Cdo 123 2020
        variants.add(f"{senat} {agenda} {zvysok.replace('/', ' ')}")

        # Format D: Agenda attached to number
        # Result: 1Cdo 123/2020
        variants.add(f"{senat}{agenda} {zvysok}")

    # 3. Fix common typos from the CSV, like a space after the slash.
    variants.add(clean_id.replace("/ ", "/"))

    return list(variants)


def download_file(url, folder, filename):
    """
    The actual downloading happens here.
    We try to guess if it's a PDF or DOCX based on the server headers.
    """
    try:
        if not url.startswith("http"):
            url = f"https://obcan.justice.sk{url}"

        # We use stream=True to download big files in chunks (pieces)
        # so we don't crash the memory.
        with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
            if r.status_code == 200:
                final_filename = filename

                # If the filename doesn't have an extension, we guess it.
                if "." not in filename[-5:]:
                    ct = r.headers.get("Content-Type", "").lower()
                    if "pdf" in ct:
                        final_filename += ".pdf"
                    elif "word" in ct or "officedocument" in ct:
                        final_filename += ".docx"
                    elif "zip" in ct:
                        final_filename += ".zip"
                    else:
                        final_filename += ".pdf"

                file_path = os.path.join(folder, final_filename)

                # Write to disk piece by piece (8kb chunks)
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"            {Colors.GREEN}[OK]{Colors.RESET} File saved: {final_filename}")
                return True
    except Exception as e:
        print(f"            {Colors.RED}[ERROR]{Colors.RESET} Download failed: {e}")
    return False


def get_decision_detail(guid):
    """
    Downloads the full JSON details (metadata) for a specific decision ID.
    """
    try:
        resp = requests.get(f"{BASE_API_URL}/v1/rozhodnutie/{guid}", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


def find_and_download(row_index, court_raw, case_id):
    """
    Main logic for one row of the CSV.
    1. Normalizes the court name.
    2. Generates ID variants.
    3. Searches the API.
    4. Matches the result.
    5. Downloads files.
    """
    court_official = normalize_court_name(court_raw)
    clean_id = str(case_id).strip()

    print(f"\n[{row_index}] {Colors.BLUE}[SEARCHING]{Colors.RESET} {clean_id} ({court_raw})")

    # The Constitutional Court is not in this database system.
    if "Ústavný súd" in court_official:
        print(f"   {Colors.YELLOW}[INFO]{Colors.RESET} Constitutional Court is not in this DB. Skipping.")
        return

    # --- SEARCH LOGIC ---
    id_variants = generate_id_variants(clean_id)
    found_decision = None

    # Loop through all generated ID variants until we find something
    for variant in id_variants:
        if found_decision: break

        params = {
            "spisovaZnacka": variant,
            "size": 20
        }
        # If it's the Supreme Court, we add a filter to ignore regional courts
        if "Najvyšší súd" in court_official:
            params["sud"] = "Najvyšší súd Slovenskej republiky"

        try:
            resp = requests.get(f"{BASE_API_URL}{SEARCH_ENDPOINT}", headers=HEADERS, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('rozhodnutieList', [])

                # Filter results to make sure it's the right court and ID
                for cand in candidates:
                    cand_znacka = cand.get('spisovaZnacka', '').replace(" ", "").lower()
                    target_znacka_clean = clean_id.replace(" ", "").replace("/", "").lower()
                    cand_znacka_clean = cand_znacka.replace("/", "")
                    cand_sud = cand.get('sud', {}).get('nazov', '').lower()

                    # Check if court name matches (fuzzy match)
                    court_match = False
                    if "najvyšší" in court_official.lower() and "najvyšší" in cand_sud:
                        court_match = True
                    elif court_official.replace("Krajský súd ", "").lower() in cand_sud:
                        court_match = True

                    # If both ID and Court match, we are golden!
                    if target_znacka_clean in cand_znacka_clean and court_match:
                        found_decision = cand
                        print(f"   {Colors.GREEN}[FOUND]{Colors.RESET} Match in system: {cand.get('spisovaZnacka')}")
                        break
        except Exception:
            pass

    if not found_decision:
        print(f"   {Colors.RED}[NOT FOUND]{Colors.RESET} Rozhodnutie sa v databaze nenachadza.")
        return

    # --- DOWNLOAD LOGIC ---
    guid = found_decision.get('guid')

    # Create folders: Output_Dir / Court_Name / Case_ID
    safe_court = court_raw.replace(" ", "_")
    safe_id = clean_id.replace("/", "_").replace(" ", "_")
    case_folder = os.path.join(OUTPUT_DIR, safe_court, safe_id)

    if not os.path.exists(case_folder):
        os.makedirs(case_folder)

    # Get details and save metadata
    detail = get_decision_detail(guid)
    if detail:
        detail["_csv_metadata"] = {"court": court_raw, "id": case_id, "topic": row_index}
        with open(os.path.join(case_folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=4)

        # Collect all attachments (documents)
        docs = []
        if 'dokument' in detail and detail['dokument']:
            val = detail['dokument']
            docs.extend(val if isinstance(val, list) else [val])
        if 'dokumenty' in detail: docs.extend(detail['dokumenty'])
        if 'prilohy' in detail: docs.extend(detail['prilohy'])

        if docs:
            print(f"      {Colors.BLUE}[INFO]{Colors.RESET} Downloading {len(docs)} attachments...")
            for i, doc in enumerate(docs):
                raw_name = doc.get('nazov', 'dokument')
                # Clean filenames so they don't break the OS
                clean_name = re.sub(r'[^\w\-\. ]', '_', raw_name).strip()
                if not clean_name: clean_name = "dokument"

                final_name = f"{i:02d}_{clean_name}"
                url = doc.get('url')

                # Sometimes URL is missing, so we construct it from ID
                if not url and doc.get('id'):
                    url = f"/v1/rozhodnutie/dokument/{doc.get('id')}/stiahnut"

                if url:
                    download_file(url, case_folder, final_name)
        else:
            # Fallback: If no PDF exists, try to save the plain text body
            full_text = detail.get('text') or detail.get('obsah')
            if full_text:
                with open(os.path.join(case_folder, "text.txt"), "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(f"      {Colors.GREEN}[OK]{Colors.RESET} Saved plain text (PDF not available).")
            else:
                print(f"      {Colors.RED}[WARNING]{Colors.RESET} Decision is empty (no text, no files).")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("--- AUTOMATIC COURT DECISION DOWNLOADER ---")

    csv_file = 'rozhodnutia_raw_list.csv'
    if not os.path.exists(csv_file):
        print(f"{Colors.RED}[ERROR]{Colors.RESET} File '{csv_file}' does not exist!")
        return

    try:
        df = pd.read_csv(csv_file, quotechar='"', skipinitialspace=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    for index, row in df.iterrows():
        c_name = row.get('court_name')
        c_id = row.get('case_id')

        # Skip empty rows
        if pd.isna(c_name) or pd.isna(c_id): continue

        find_and_download(index, c_name, c_id)

        # Sleep for a bit to be polite to the server and avoid bans
        time.sleep(0.5)


if __name__ == "__main__":
    main()