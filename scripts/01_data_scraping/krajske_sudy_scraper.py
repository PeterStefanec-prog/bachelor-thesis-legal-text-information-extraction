
# Scraper for Krajske sudy (Regional courts) of Slovakia.
# I want to find their decisions where court used paragraph 301 Obchodneho zakonnika (moderation right on contractual penalty).
#
# Source: obcan.justice.sk/pilot/api (Ministry of Justice open API).
# This is the best source because:
#   - nsud.sk OpenData only has Najvyssi sud, nothing from KS
#   - otvorenesudy.sk is blocked by Cloudflare (tested in our old docs)
#   - slov-lex.sk has Liferay token protection
#   - obcan.justice.sk has proper public API with Swagger at
#     /pilot/api/ress-isu-service/v3/api-docs
#
#  API has super useful filter "odkazovanePredpisy" which takes paragraph reference like "/SK/ZZ/1991/513/#paragraf-301"
#  - this is  exactly paragraph 301 of law 513/1991 (Obchodny zakonnik).
#  So I dont even need fulltext search - I filter directly by metadata.
#
# I combine this with:
#   - typSuduFacetFilter = "Krajsky sud" (only regional courts)
#   - vydaniaOd / vydaniaDo for date range
#   - oblastPravnejUpravyFacetFilter = "Obchodne pravo" (to be safe)

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

#################################
# ############ CONFIG ################

# Last 10 years -
DATE_OD = "2015-01-01"
DATE_DO = "2025-12-31"

# Output paths. I put it into same data folder as NS SR scraper
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PDF_DIR = DATA_DIR / "final_ks_pdfs"
JSON_DIR = DATA_DIR / "final_ks_json"
CSV_PATH = DATA_DIR / "final_ks_metadata.csv"

# ministry of Justice API
API_BASE = "https://obcan.justice.sk/pilot/api/ress-isu-service/v1/rozhodnutie"

#  !!! great key filter - reference to paragraph 301 Obchodny zakonnik (zakon 513/1991) - exact match which is in API
PARAGRAF_301 = "/SK/ZZ/1991/513/#paragraf-301"

# HTTP headers - I pretend to be normal browser.
# Sometimes servers  block requests from "python-requests" user-agent  - # lowering chance of blocking from website
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, xstefanec) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
##########################################
##########################################


# ### HELPERS ###

#### MAIN SEARCHING FUNCITON #####
def search_decisions(session: requests.Session) -> list[dict]:      # session remembers headers, cookies
    # i use pagination because API returns max 20-100 jsons per page (so response will not be that big)
    all_results: list[dict] = []    # list of all found decisions
    page = 0
    size = 100

    # repeat requests until API responds with new decisions     (returns only summary of decision)
    while True:
        # Params of requestt - list of tuples
        params = [
            ("odkazovanePredpisy", PARAGRAF_301),   # has to have par 301
            ("typSuduFacetFilter", "Krajský súd"),
            ("oblastPravnejUpravyFacetFilter", "Obchodné právo"),   # are in obchodne pravo
            ("vydaniaOd", DATE_OD),
            ("vydaniaDo", DATE_DO),
            ("page", str(page)),
            ("size", str(size)),
        ]
        try:
            # request to open justice openAPI
            r = session.get(API_BASE, params=params, timeout=30) # library add params into request automatically
            if r.status_code != 200:    # if status is other than 200 OK
                print(f"[WARN] search page {page} HTTP {r.status_code}")
                break
            data = r.json() # parsin json
        except Exception as e:
            print(f"[WARN] search page {page} failed: {e}")
            break

        batch = data.get("rozhodnutieList", []) # results are in this list
        if not batch:
            break
        all_results.extend(batch)   # adding decisions one by one as elements

        total = data.get("numFound", 0) # number of results that exists
        print(f"[INFO] page={page} got={len(batch)} total_so_far={len(all_results)}/{total}")

        # if i have all results processed - finish - otherwise go on next page
        if len(all_results) >= total:
            break
        page += 1
        time.sleep(0.3)

    return all_results


def get_decision_detail(session: requests.Session, guid: str) -> dict | None:
    # Gettin full detail of one decision by GUID
    try:
        url = f"{API_BASE}/{guid}"      # .../rozhodnutie/{guid}
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def safe_filename(s: str) -> str:
    # Remove or replace chars that OS doesnt like in filenames (spisove znacky contains "/" and spaces...  buut OS do notl ike that
    if not s:
        return "file"
    # change "3Cob/12/2020" to "3Cob_12_2020"
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = re.sub(r"[^a-zA-Z0-9._\- ]+", "_", s)   # cutting _ at the end or beggining
    return s.strip("_") or "file"


def download_document(session: requests.Session, doc_url: str, out_path: Path) -> bool:
    # Download dcoument
    try:
        if not doc_url.startswith("http"):
            doc_url = "https://obcan.justice.sk" + doc_url
        with session.get(doc_url, stream=True, timeout=30) as r:    # stream True - so downloading bigger pdfs is more easy
            if r.status_code != 200:
                return False
            with open(out_path, "wb") as f:     # writing into file chunk by chunk - binary
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"  [ERROR] download failed: {e}")
        return False


# ### MAIN ###
def main():
    # creating folders
    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)
    JSON_DIR.mkdir(exist_ok=True)

    # one session for all requests
    session = requests.Session()
    session.headers.update(HEADERS)

    # Resume if CSV already exists - dont redownload
    existing_guids: set[str] = set()    # already downloaded guids
    rows: list[dict] = []   # rows for csv
    if CSV_PATH.exists():
        try:
            df_old = pd.read_csv(CSV_PATH, sep="|") # loading csv with already donwloaded
            existing_guids = set(df_old["guid"].astype(str).tolist())
            rows = df_old.to_dict("records")
            print(f"[INFO] Loaded {len(existing_guids)} already saved")
        except Exception as e:
            print(f"[WARN] cant read old CSV: {e}")

    print(f"[STEP 1] Search for KS decisions with § 301 ({DATE_OD}..{DATE_DO})")
    results = search_decisions(session)     # list of found decisions
    print(f"[INFO] Found {len(results)} decisions")

    # Skip ones already downloaded - keep only those whose guid is not in in csv already
    todo = [r for r in results if str(r.get("guid")) not in existing_guids]
    print(f"[INFO] New to download: {len(todo)}")

    print(f"[STEP 2] Download PDF + metadata for each")
    saved = 0
    errors = 0

    for res in tqdm(todo, desc="Downloading"):  # got through all new decisions
        guid = res.get("guid")
        spis = res.get("spisovaZnacka", "unknown")
        datum = res.get("datumVydania", "unknown")
        sud = res.get("sud", {}).get("nazov", "unknown")

        # Get full detail (main list does not have dokument url)
        detail = get_decision_detail(session, guid)
        if not detail:
            errors += 1
            continue

        # Saving detail JSON for later analysis - thesis also needs metadata
        safe_spis = safe_filename(spis)
        safe_sud = safe_filename(sud.replace("Krajský súd ", "KS_"))
        json_name = f"{safe_sud}_{safe_spis}.json"
        with open(JSON_DIR / json_name, "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)      # daving detailed metadata

        # Find document URL. Structure is {"dokument": {"url": "...", "name": "..."}}
        doc = detail.get("dokument")
        doc_url = None
        doc_name = None
        if isinstance(doc, dict):   # array is dictionary
            doc_url = doc.get("url")
            doc_name = doc.get("name")
        elif isinstance(doc, list) and doc:     # array is list
            doc_url = doc[0].get("url")
            doc_name = doc[0].get("name")

        if not doc_url:
            # Fallback: try plain text body if there is no pdf
            text_body = detail.get("text") or detail.get("obsah")
            if text_body:
                txt_path = PDF_DIR / f"{safe_sud}_{safe_spis}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text_body)
                rows.append({
                    "guid": guid,
                    "sud": sud,
                    "spisova_znacka": spis,
                    "datum": datum,
                    "file_name": txt_path.name,
                    "file_type": "txt",
                    "odkazovane_predpisy": ";".join(
                        p.get("nazov", "") for p in detail.get("odkazovanePredpisy", [])
                    ),
                })
                saved += 1
            else:
                errors += 1
            continue

        # Download the PDF/DOCX
        ext = (Path(doc_name or "").suffix or ".pdf").lower()   # what is the suffix?
        pdf_filename = f"{safe_sud}_{safe_spis}{ext}"
        out_path = PDF_DIR / pdf_filename

        ok = download_document(session, doc_url, out_path)  # downloading document
        if not ok:
            errors += 1
            continue

        rows.append({
            "guid": guid,
            "sud": sud,
            "spisova_znacka": spis,
            "datum": datum,
            "file_name": pdf_filename,
            "file_type": ext.strip("."),
            "odkazovane_predpisy": ";".join(
                p.get("nazov", "") for p in detail.get("odkazovanePredpisy", [])
            ),
        })
        saved += 1

        # Save CSV after every row (crash-safe)
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False, sep="|", encoding="utf-8")

        time.sleep(0.2)

    print(f"\n[DONE]")
    print(f"  Saved: {saved}")
    print(f"  Errors: {errors}")
    print(f"  CSV: {CSV_PATH}")
    print(f"  PDFs: {PDF_DIR}")
    print(f"  JSON metadata: {JSON_DIR}")


if __name__ == "__main__":
    main()
