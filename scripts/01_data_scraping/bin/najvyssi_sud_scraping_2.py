#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# My second try at NS SR scraper. Same idea as version 1 but smarter
# filtering. I added more terms in OBSAH_TERMS and more excludes for
# stuff I dont want (Ndob, ZKR/konkurz, procesne veci).
#
# Same as before:
#   - only OpenData API, no HTML scraping
#   - day-by-day search via art_obsah
#   - keep only commercial division (kolegium 2)
#
# New stuff in v2:
#   - excludes execution cases (already had)
#   - excludes Ndob (procedural - "prikazanie veci" etc)
#   - excludes konkurz/ZKR (bankruptcy)
#   - excludes other procedural-only merito words
#   - also saves full JSON metadata per decision (not just CSV)
#
# Note: this still uses art_obsah which only matches short annotation,
# not full PDF text. Found this out later, so v2 still misses stuff.

import csv
import json
import os
import re
import time
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================
# Config
# =====================================

BASE_API = "https://www.nsud.sk/ws/opendata.php"
BASE_SITE = "https://www.nsud.sk"

# Date range (both included). Adjust as needed.
DATE_START = "2025-01-01"
DATE_END   = "2025-03-10"

# Keywords for art_obsah - all variants of "zmluvna pokuta" I could think of.
# I dont put § 261 / § 301 / "ObZ" here on purpose - those are matched
# later when analyzing the PDF.
OBSAH_TERMS = [
    # core - "zmluvna pokuta" (different cases / diacritics)
    "zmluvná pokuta",
    "zmluvnu pokutu",
    "zmluvnú pokutu",
    "zmluvnej pokuty",
    "zmluvné pokuty",
    "zmluvne pokuty",

    # "zmluvna sankcia" - bit broader but still on topic
    "zmluvná sankcia",
    "zmluvnu sankciu",
    "zmluvnú sankciu",
    "zmluvnej sankcie",
    "zmluvné sankcie",
    "zmluvne sankcie",

    # more specific phrases (still about contractual penalty)
    "zníženie zmluvnej pokuty",
    "moderácia zmluvnej pokuty",
    "neprimeraná zmluvná pokuta",
    "primeranosť zmluvnej pokuty",
]


# Kolegium codes:
# 1 = civil, 2 = commercial. I keep only commercial.
REQUIRED_KOLEGIUMS = {2}

# Skip execution cases - they are about enforcing the penalty, not deciding
# the merits.
EXCLUDE_EXECUTION = True

# Skip Ndob - those are usualy proposals/transfers, not real decisions
EXCLUDE_NDOB = True

# Skip konkurz/insolvency (ZKR)
EXCLUDE_ZKR = True

ZKR_KEYWORDS = [
    "zákon č. 7/2005", "zákona č. 7/2005", "zkr",
    "konkurz", "konkurze",
    "oddlžen", "oddĺžen",  # also catches "oddlzenie", "oddlzenie"
]

# Skip clearly procedural merito (transfer of case, venue etc)
EXCLUDE_PROCEDURAL_ONLY = True

PROCEDURAL_MERITO_KEYWORDS = [
    "prikázanie veci", "prikazanie veci",
    "prikázanie sporu", "prikazanie sporu",
    "návrh na prikázanie", "navrh na prikazanie",
    "určenie príslušnosti", "urcenie prislusnosti",
    "určenie miestnej príslušnosti", "urcenie miestnej prislusnosti",
    "nesúhlas s postúpením", "nesuhlas s postupenim",
]

# Output paths
OUT_DIR_PDFS = "data/nsud_pdfs2"
OUT_DIR_JSON = "data/nsud_json2"
OUT_CSV_PATH = "data/nsud_metadata2.csv"

# Network
MAX_WORKERS = 8
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30
PER_REQUEST_SLEEP = 0.02
SESSION_HEADERS = {
    "User-Agent": "NSUD-OpenData-Downloader/2.0",
    "Accept": "application/json,text/*;q=0.9,*/*;q=0.1",
}

# =====================================
# Helpers
# =====================================


def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def _fmt_dmy(date_iso: str) -> str:
    # Convert YYYY-MM-DD to DD.MM.YYYY because API is weird
    y, m, d = date_iso.split("-")
    return f"{d}.{m}.{y}"


def _daterange_days(start_dt: datetime, end_dt: datetime):
    # Yield ISO date for every day in range
    cur = start_dt
    one = timedelta(days=1)
    while cur <= end_dt:
        yield cur.strftime("%Y-%m-%d")
        cur += one


def _robust_get_json(session: requests.Session, params: dict):
    # GET with retries because API sometimes hiccups
    retries = 3
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(
                BASE_API,
                params=params,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            # raise for 4xx/5xx
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"GET failed for params={params}: {last_exc}")


def _extract_ids(payload) -> list[int]:
    # API returns IDs in different shapes - this handles all of them
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "id" in payload[0]:
            return [int(x["id"]) for x in payload if "id" in x]
        return [int(x) for x in payload]
    if isinstance(payload, dict):
        for k in ("ids", "data", "result"):
            if k in payload and isinstance(payload[k], list):
                return _extract_ids(payload[k])
    return []


def _search_ids_one_day(session: requests.Session, day_iso: str, obsah_term: str) -> list[int]:
    # Search IDs for ONE specific day and ONE term.
    # Try both ISO and D.M.Y formats. No fallback without dates.
    variants = [
        (day_iso, day_iso),  # YYYY-MM-DD
        (_fmt_dmy(day_iso), _fmt_dmy(day_iso)),  # DD.MM.YYYY
    ]
    for idx, (od, do) in enumerate(variants, start=1):
        params = {
            "searchDecision": "",
            "art_datum_od": od,
            "art_datum_do": do,
            "art_obsah": obsah_term,
        }
        data = _robust_get_json(session, params)
        ids = _extract_ids(data)
        if ids:
            print(f"[INFO] {day_iso} term='{obsah_term}' -> {len(ids)} ids (format {idx})")
            return ids
    return []


def _get_decision(session: requests.Session, dec_id: int) -> dict | None:
    # Get full metadata for one decision
    params = {"getDecision": "", "id": str(dec_id)}
    try:
        data = _robust_get_json(session, params)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
    except Exception:
        return None
    return None


def _candidate_pdf_urls(subor_value: str | None) -> list[str]:
    # Build PDF URL from "subor" field. Sometimes its full URL, sometimes
    # relative path - need to handle both.
    if not subor_value:
        return []
    s = subor_value.strip()
    if s.startswith(("http://", "https://")):
        return [s]
    if s.startswith("/data/att/"):
        return [urljoin(BASE_SITE, s)]
    if s.startswith("data/att/"):
        return [urljoin(BASE_SITE + "/", s)]
    # fallback guesses
    return [
        urljoin(BASE_SITE + "/", "data/att/" + s.lstrip("/")),
        urljoin(BASE_SITE + "/", s.lstrip("/")),
    ]


def _safe_filename(s: str) -> str:
    # remove chars that OS doesnt like
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s or "").strip("_") or "file"


def _download_pdf(session: requests.Session, urls: list[str], out_path: str) -> tuple[str, int, str]:
    # Try to download PDF from list of candidate URLs.
    # Returns (status_label, http_status, used_url)
    # status_label is one of: "ok", "http_404", "http_other"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for url in urls:
        try:
            with session.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=True,
            ) as r:
                if r.status_code != 200:
                    continue
                ctype = r.headers.get("Content-Type", "").lower()
                # accept real PDFs by content type, extension is just hint
                if "pdf" not in ctype and not url.lower().endswith(".pdf"):
                    continue
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
            return ("ok", 200, url)
        except Exception:
            continue

    if urls:
        try:
            h = session.head(
                urls[0],
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=True,
            )
            if h.status_code == 404:
                return ("http_404", 404, urls[0])
            return ("http_other", h.status_code, urls[0])
        except Exception:
            pass
    return ("http_other", 0, urls[0] if urls else "")


# =====================================
# Main
# =====================================


def main():
    # 1) check date range
    start_dt = _parse_date(DATE_START)
    end_dt = _parse_date(DATE_END)
    if end_dt < start_dt:
        raise ValueError("DATE_END must be >= DATE_START")

    # Session for ID search (single thread, safe)
    search_session = requests.Session()
    search_session.headers.update(SESSION_HEADERS)

    # 2) prepare CSV header and output dirs
    os.makedirs(os.path.dirname(OUT_CSV_PATH) or ".", exist_ok=True)
    os.makedirs(OUT_DIR_PDFS, exist_ok=True)
    os.makedirs(OUT_DIR_JSON, exist_ok=True)

    csv_headers = [
        "id",
        "cislo",
        "senat",
        "ecli",
        "datum",
        "kolegium",
        "is_civil",
        "is_commercial",
        "merito",
        "sudca",
        "subor",
        "pdf_url",
        "saved_path",
        "download_status",
        "http_status",
        "json_path",
    ]

    # 3) collect all IDs by days and terms (art_obsah)
    all_ids: set[int] = set()
    for day in _daterange_days(start_dt, end_dt):
        for term in OBSAH_TERMS:
            try:
                ids = _search_ids_one_day(search_session, day, term)
                if ids:
                    all_ids.update(ids)
            except Exception as e:
                print(f"[WARN] searchDecision failed for {day} term='{term}': {e}")
            time.sleep(PER_REQUEST_SLEEP)

    all_ids = sorted(all_ids)
    print(f"[INFO] Collected {len(all_ids)} unique candidate IDs.")

    # 4) for each ID: filter kolegium, optionaly exclude executions/procedural,
    #    download PDF, store JSON + CSV row

    def _process_one(dec_id: int) -> dict | None:
        # Process one decision:
        # - get metadata via getDecision
        # - filter by kolegium, docket type, execution/procedural
        # - download PDF
        # - save JSON metadata
        # - return CSV row
        # Each worker has its own Session so threads dont fight
        with requests.Session() as local_session:
            local_session.headers.update(SESSION_HEADERS)

            meta = _get_decision(local_session, dec_id)
            time.sleep(PER_REQUEST_SLEEP)
            if not meta:
                print(f"[WARN] getDecision returned nothing for ID={dec_id}")
                return None

            # kolegium filter
            try:
                kol = int(meta.get("kolegium", -1))
            except Exception:
                return None

            if kol not in REQUIRED_KOLEGIUMS:
                return None

            cislo = str(meta.get("cislo") or "")

            # skip Ndob (proposals/venue/transfer decisions)
            if "Ndob" in cislo:
                return None

            # combine merito + obsah for keyword screening
            preview = (
                str(meta.get("merito", "")) + " " +
                str(meta.get("obsah", ""))
            ).lower()

            # 1) skip execution cases
            if EXCLUDE_EXECUTION:
                if any(x in preview for x in ["exekuč", "exekúcia", "exekučné konanie"]):
                    return None

            # 2) skip konkurz/insolvency (ZKR)
            if EXCLUDE_ZKR:
                if any(kw in preview for kw in ZKR_KEYWORDS):
                    return None

            # 3) skip procedural-only merito (transfer, venue etc)
            if EXCLUDE_PROCEDURAL_ONLY:
                merito = (meta.get("merito") or "").lower()
                if any(kw in merito for kw in PROCEDURAL_MERITO_KEYWORDS):
                    return None


            # candidate PDF URLs
            candidates = _candidate_pdf_urls(meta.get("subor"))
            saved_path = ""
            status_label, http_status, used_url = ("http_other", 0, "")

            if candidates:
                fname = (
                    f"{meta.get('id', dec_id)}__"
                    f"{_safe_filename((meta.get('datum') or '').replace(' ', '_'))}__"
                    f"{_safe_filename(meta.get('cislo') or '')}.pdf"
                )
                out_path = os.path.join(OUT_DIR_PDFS, fname)
                status_label, http_status, used_url = _download_pdf(local_session, candidates, out_path)
                if status_label == "ok":
                    saved_path = out_path
                else:
                    print(
                        f"[WARN] download failed for ID={dec_id} "
                        f"status={http_status} url={used_url} reason={status_label}"
                    )
            else:
                print(f"[WARN] missing 'subor' for ID={dec_id}")

            # save full JSON metadata for later analysis
            json_path = os.path.join(OUT_DIR_JSON, f"{meta.get('id', dec_id)}.json")
            try:
                with open(json_path, "w", encoding="utf-8") as fj:
                    json.dump(meta, fj, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] failed to save JSON for ID={dec_id}: {e}")
                json_path = ""

            # build CSV row
            is_civil = 1 if kol == 1 else 0
            is_commercial = 1 if kol == 2 else 0

            return {
                "id": meta.get("id", dec_id),
                "cislo": meta.get("cislo"),
                "senat": meta.get("senat"),
                "ecli": meta.get("ecli"),
                "datum": meta.get("datum"),
                "kolegium": meta.get("kolegium"),
                "is_civil": is_civil,
                "is_commercial": is_commercial,
                "merito": meta.get("merito"),
                "sudca": meta.get("sudca"),
                "subor": meta.get("subor"),
                "pdf_url": used_url or (candidates[0] if candidates else ""),
                "saved_path": saved_path,
                "download_status": status_label,
                "http_status": http_status,
                "json_path": json_path,
            }

    # 5) run processing in a thread pool
    with open(OUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=csv_headers,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_process_one, did): did for did in all_ids}
            for fut in as_completed(futures):
                row = fut.result()
                if row is None:
                    continue
                writer.writerow(row)
                fcsv.flush()

    print(f"[INFO] Done. Metadata CSV: {OUT_CSV_PATH}")
    print(f"[INFO] PDFs saved under:   {OUT_DIR_PDFS}")
    print(f"[INFO] JSON metadata under:{OUT_DIR_JSON}")


if __name__ == "__main__":
    main()




# =====================================================================
# Older draft that I kept for reference - this one tried to filter
# more strictly (only "real" pokuta cases). Keeping it commented out
# in case I want to compare results later.
# =====================================================================

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# import csv
# import json
# import os
# import time
# import re
# from datetime import datetime, timedelta
# from urllib.parse import urljoin
# import requests
#
# # --- CONFIG ---
# # I search in years where case-law is settled (not in the future 2025)
# DATE_START = "2015-01-01"
# DATE_END = "2023-12-31"
#
# # Keywords for API search
# SEARCH_TERMS = ["zmluvná pokuta", "zmluvnej pokuty"]
#
# # Output folders
# OUT_DIR_PDFS = "dataset_final/pdfs"
# OUT_DIR_JSON = "dataset_final/json"
# OUT_CSV_PATH = "dataset_final/metadata.csv"
#
# # API stuff
# BASE_API = "https://www.nsud.sk/ws/opendata.php"
# BASE_SITE = "https://www.nsud.sk"
#
#
# def get_decision_ids(session, date_iso, term):
#     # Get IDs for one day and one term. Handle date format issue.
#     # NS SR API sometimes wants YYYY-MM-DD, sometimes D.M.YYYY
#     d_obj = datetime.strptime(date_iso, "%Y-%m-%d")
#     date_dmy = d_obj.strftime("%d.%m.%Y")
#
#     params = {
#         "searchDecision": "",
#         "art_datum_od": date_dmy,
#         "art_datum_do": date_dmy,
#         "art_obsah": term
#     }
#
#     try:
#         r = session.get(BASE_API, params=params, timeout=5)
#         if r.status_code == 200:
#             data = r.json()
#             # API returns IDs in various shapes
#             if isinstance(data, list):
#                 return [int(x['id']) for x in data if isinstance(x, dict) and 'id' in x]
#             if isinstance(data, dict) and 'ids' in data:
#                 return [int(x) for x in data['ids']]
#     except Exception as e:
#         pass
#     return []
#
#
# def get_decision_detail(session, dec_id):
#     # download decision detail
#     try:
#         r = session.get(BASE_API, params={"getDecision": "", "id": str(dec_id)}, timeout=5)
#         if r.status_code == 200:
#             data = r.json()
#             if isinstance(data, list) and len(data) > 0: return data[0]
#             if isinstance(data, dict): return data
#     except:
#         pass
#     return None
#
#
# def is_relevant(meta):
#     # Decides if decision is relevant for thesis.
#     # Filters out procedural junk.
#     # 1. must be commercial division (2)
#     if str(meta.get('kolegium')) != '2':
#         return False
#
#     merito = str(meta.get('merito', '')).lower()
#     obsah = str(meta.get('obsah', '')).lower()
#     text = str(meta.get('text', '')).lower()  # sometimes text snippet is there
#
#     # 2. throw out procedural stuff
#     stop_words = ["prikázanie veci", "miestnej príslušnosti", "odmietnutie", "trovy konania", "exekúci"]
#     if any(sw in merito for sw in stop_words):
#         return False
#
#     # 3. must be about money or determining a right.
#     # Look for "pokuta" + "zaplatenie/urcenie/znizenie"
#     strong_keywords = ["zaplatenie", "určenie", "zníženie", "moderačn"]
#
#     has_strong_kw = any(kw in merito or kw in obsah for kw in strong_keywords)
#     has_pokuta = "pokut" in merito or "pokut" in obsah
#
#     if has_pokuta and has_strong_kw:
#         return True
#
#     return False
#
#
# def download_pdf(session, url, path):
#     try:
#         with session.get(url, stream=True, timeout=20) as r:
#             if r.status_code == 200:
#                 with open(path, "wb") as f:
#                     for chunk in r.iter_content(chunk_size=8192):
#                         f.write(chunk)
#                 return True
#     except:
#         pass
#     return False
#
#
# def main():
#     # prepare folders
#     os.makedirs(OUT_DIR_PDFS, exist_ok=True)
#     os.makedirs(OUT_DIR_JSON, exist_ok=True)
#
#     session = requests.Session()
#     session.headers.update({"User-Agent": "Research-Bot/1.0"})
#
#     # CSV writer
#     f_csv = open(OUT_CSV_PATH, "w", newline="", encoding="utf-8-sig")
#     writer = csv.DictWriter(f_csv, fieldnames=["id", "spisova_znacka", "datum", "merito", "subor", "relevancia"])
#     writer.writeheader()
#
#     processed_ids = set()
#     start_date = datetime.strptime(DATE_START, "%Y-%m-%d")
#     end_date = datetime.strptime(DATE_END, "%Y-%m-%d")
#
#     print(f"Starting download from {DATE_START} to {DATE_END}...")
#
#     current_date = start_date
#     while current_date <= end_date:
#         date_iso = current_date.strftime("%Y-%m-%d")
#
#         # for each term find IDs
#         found_ids_today = set()
#         for term in SEARCH_TERMS:
#             ids = get_decision_ids(session, date_iso, term)
#             found_ids_today.update(ids)
#
#         if found_ids_today:
#             print(f"Date {date_iso}: found {len(found_ids_today)} candidates.")
#
#             for dec_id in found_ids_today:
#                 if dec_id in processed_ids:
#                     continue
#
#                 meta = get_decision_detail(session, dec_id)
#                 if not meta: continue
#
#                 # --- CRITICAL CHECK ---
#                 # If API returns decision from different year (because it ignores
#                 # filter), skip it. This eliminates "cyclic" duplicates.
#                 real_date = meta.get('datum', '')
#                 if real_date != date_iso:
#                     continue
#
#                 if is_relevant(meta):
#                     print(f"   -> [DOWNLOADED] ID {dec_id} | {meta.get('cislo')} | {meta.get('merito')[:40]}...")
#
#                     # save JSON
#                     with open(os.path.join(OUT_DIR_JSON, f"{dec_id}.json"), "w") as f:
#                         json.dump(meta, f, indent=2, ensure_ascii=False)
#
#                     # download PDF
#                     subor_url = meta.get('subor')
#                     if subor_url:
#                         if not subor_url.startswith("http"):
#                             subor_url = urljoin(BASE_SITE, subor_url.lstrip('/'))
#                             if "data/att" not in subor_url and "/data/att" not in subor_url:
#                                 # fix for weird links
#                                 subor_url = subor_url.replace("nsud.sk/", "nsud.sk/data/att/")
#
#                         fname = f"{dec_id}_{re.sub(r'[^a-zA-Z0-9]', '_', meta.get('cislo', ''))}.pdf"
#                         download_pdf(session, subor_url, os.path.join(OUT_DIR_PDFS, fname))
#
#                     # write to CSV
#                     writer.writerow({
#                         "id": dec_id,
#                         "spisova_znacka": meta.get('cislo'),
#                         "datum": real_date,
#                         "merito": meta.get('merito'),
#                         "subor": subor_url,
#                         "relevancia": "vysoka"
#                     })
#                     f_csv.flush()
#                     processed_ids.add(dec_id)
#
#         current_date += timedelta(days=1)
#         # small pause so I dont kill server, but not too long
#         # time.sleep(0.1)
#
#     print("Done.")
#
#
# if __name__ == "__main__":
#     main()
