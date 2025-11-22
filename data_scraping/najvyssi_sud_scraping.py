#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NS SR (Najvyšší súd SR) – downloader pre obchodnoprávne rozhodnutia (kolegium == 2).

- Iba oficiálne OpenData API (žiadny HTML scraping).
- Vyhľadávanie po DŇOCH cez `searchDecision` s parametrom art_obsah.
- Metadata cez `getDecision`.
- Sťahovanie PDF podľa poľa `subor`.
- CSV bez viacriadkových polí (neukladáme 'obsah'), takže je čisté.

Dôležité:
- ŽIADEN fallback „bez dátumu“ – ak v daný deň nič nie je, tak nič nepridáme.
- API niekedy ignoruje rozsah dátumov; preto ideme day-by-day.
"""

import csv
import os
import re
import time
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================
# Konfigurácia
# =============================

BASE_API  = "https://www.nsud.sk/ws/opendata.php"
BASE_SITE = "https://www.nsud.sk"

# Dátumové okno (vrátane)
DATE_START = "2015-01-01"
DATE_END   = "2015-01-05"

# Kľúčové slová – zmluvné pokuty a príbuzné pojmy
OBSAH_TERMS = [
    # jadro – explicitná zmluvná pokuta (rôzne pády/diakritika)
    "zmluvná pokuta",
    "zmluvnu pokutu",
    "zmluvnej pokuty",
    "zmluvné pokuty",
    "zmluvne pokuty",

    # synonymá/terminológia, ktoré súdy používajú pri pokutách
    "zmluvná sankcia",
    "zmluvnu sankciu",
    "zmluvnej sankcie",
    "zmluvné sankcie",
]
#
# OBSAH_TERMS = [
#     "zmluvná pokuta", "zmluvnej pokuty"
# ]

# OBSAH_TERMS = [
#     "zmluvná pokuta", "zmluvnej pokuty",
#     "zníženie zmluvnej pokuty", "moderácia zmluvnej pokuty", "primeranosť zmluvnej pokuty"
# ]


# iba kolegium 2 (obchodnoprávne)
REQUIRED_KOLEGIUM = 2

# Výstupy
OUT_DIR_PDFS = "data/nsud_pdfs"
OUT_CSV_PATH = "data/nsud_metadata.csv"

# Sieť
MAX_WORKERS       = 8
CONNECT_TIMEOUT   = 10
READ_TIMEOUT      = 30
PER_REQUEST_SLEEP = 0.5
SESSION_HEADERS = {
    "User-Agent": "NSUD-OpenData-Downloader/1.4",
    "Accept": "application/json,text/*;q=0.9,*/*;q=0.1",
}

# =============================
# Pomocné funkcie
# =============================

def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")

def _fmt_dmy(date_iso: str) -> str:
    y, m, d = date_iso.split("-")
    return f"{d}.{m}.{y}"

def _daterange_days(start_dt: datetime, end_dt: datetime):
    cur = start_dt
    one = timedelta(days=1)
    while cur <= end_dt:
        yield cur.strftime("%Y-%m-%d")
        cur += one

def _robust_get_json(session: requests.Session, params: dict):
    retries = 3
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(BASE_API, params=params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"GET failed for params={params}: {last_exc}")

def _extract_ids(payload) -> list[int]:
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
    """
    Vyhľadá ID pre JEDEN deň a jeden term. Skúsi 2 formáty dátumu.
    Žiadny fallback bez dátumu.
    """
    variants = [
        (day_iso, day_iso),                 # YYYY-MM-DD
        (_fmt_dmy(day_iso), _fmt_dmy(day_iso)),  # DD.MM.YYYY
    ]
    for idx, (od, do) in enumerate(variants, start=1):
        params = {"searchDecision": "", "art_datum_od": od, "art_datum_do": do, "art_obsah": obsah_term}
        data = _robust_get_json(session, params)
        ids = _extract_ids(data)
        if ids:
            # stručný log nech vieš, že ten formát fungoval
            print(f"[INFO] {day_iso} term='{obsah_term}' -> {len(ids)} ids (fmt {idx})")
            return ids
    return []

def _get_decision(session: requests.Session, dec_id: int) -> dict | None:
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
    if not subor_value:
        return []
    s = subor_value.strip()
    if s.startswith(("http://", "https://")):
        return [s]
    if s.startswith("/data/att/"):
        return [urljoin(BASE_SITE, s)]
    if s.startswith("data/att/"):
        return [urljoin(BASE_SITE + "/", s)]
    return [
        urljoin(BASE_SITE + "/", "data/att/" + s.lstrip("/")),
        urljoin(BASE_SITE + "/", s.lstrip("/")),
    ]

def _safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s or "").strip("_") or "file"

def _download_pdf(session: requests.Session, urls: list[str], out_path: str) -> tuple[str, int, str]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for url in urls:
        if not url.lower().endswith(".pdf"):
            continue
        try:
            with session.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=True) as r:
                if r.status_code == 404:
                    continue
                if r.status_code != 200:
                    continue
                ctype = r.headers.get("Content-Type", "").lower()
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
            h = session.head(urls[0], timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=True)
            return ("http_404" if h.status_code == 404 else "http_other", h.status_code, urls[0])
        except Exception:
            pass
    return ("http_other", 0, urls[0] if urls else "")

# =============================
# Hlavná časť
# =============================

def main():
    # 1) validácia dátumov (naozaj ako datetime)
    start_dt = _parse_date(DATE_START)
    end_dt   = _parse_date(DATE_END)
    if end_dt < start_dt:
        raise ValueError("DATE_END must be >= DATE_START")

    session = requests.Session()
    session.headers.update(SESSION_HEADERS)

    # 2) CSV hlavička
    os.makedirs(os.path.dirname(OUT_CSV_PATH) or ".", exist_ok=True)
    csv_headers = [
        "id", "cislo", "senat", "ecli", "datum",
        "kolegium", "merito", "sudca", "subor",
        "pdf_url", "saved_path", "download_status", "http_status",
    ]

    # 3) zber ID – po dňoch a po termínoch
    all_ids = set()
    for day in _daterange_days(start_dt, end_dt):
        for term in OBSAH_TERMS:
            try:
                ids = _search_ids_one_day(session, day, term)
                if ids:
                    all_ids.update(ids)
            except Exception as e:
                print(f"[WARN] searchDecision failed for {day} term='{term}': {e}")
            time.sleep(PER_REQUEST_SLEEP)

    all_ids = sorted(all_ids)
    print(f"[INFO] Collected {len(all_ids)} unique candidate IDs.")

    # 4) spracovanie -> filter kolegium -> download -> CSV
    os.makedirs(OUT_DIR_PDFS, exist_ok=True)
    with open(OUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=csv_headers, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()

        def _process_one(dec_id: int) -> dict | None:

            meta = _get_decision(session, dec_id)

            preview = (str(meta.get("merito", "")) + " " + str(meta.get("obsah", ""))).lower()

            # 1) bez pokút ma to nezaujíma vôbec
            if "pokut" not in preview:
                return None

            # 2) ak chceš striktne zmluvné pokuty:
            if "zmluv" not in preview:
                return None


            time.sleep(PER_REQUEST_SLEEP)
            if not meta:
                print(f"[WARN] getDecision returned nothing for ID={dec_id}")
                return None

            try:
                if int(meta.get("kolegium", -1)) != REQUIRED_KOLEGIUM:
                    return None
            except Exception:
                return None

            # vynechaj exekučné veci (ale nie „vymoženie“)
            preview = (str(meta.get("merito", "")) + " " + str(meta.get("obsah", ""))).lower()
            if any(x in preview for x in ["exekuč", "exekúcia", "exekučné konanie"]):
                return None

            candidates = _candidate_pdf_urls(meta.get("subor"))
            saved_path = ""
            status_label, http_status, used_url = ("http_other", 0, "")

            if candidates:
                fname = f"{meta.get('id', dec_id)}__{_safe_filename((meta.get('datum') or '').replace(' ', '_'))}__{_safe_filename(meta.get('cislo') or '')}.pdf"
                out_path = os.path.join(OUT_DIR_PDFS, fname)
                status_label, http_status, used_url = _download_pdf(session, candidates, out_path)
                if status_label == "ok":
                    saved_path = out_path
                else:
                    print(f"[WARN] download failed for ID={dec_id} status={http_status} url={used_url} reason={status_label}")
            else:
                print(f"[WARN] missing 'subor' for ID={dec_id}")

            return {
                "id":       meta.get("id", dec_id),
                "cislo":    meta.get("cislo"),
                "senat":    meta.get("senat"),
                "ecli":     meta.get("ecli"),
                "datum":    meta.get("datum"),
                "kolegium": meta.get("kolegium"),
                "merito":   meta.get("merito"),
                "sudca":    meta.get("sudca"),
                "subor":    meta.get("subor"),
                "pdf_url":  used_url or (candidates[0] if candidates else ""),
                "saved_path": saved_path,
                "download_status": status_label,
                "http_status": http_status,
            }

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_process_one, did): did for did in all_ids}
            for fut in as_completed(futures):
                row = fut.result()
                if row is None:
                    continue
                writer.writerow(row)
                fcsv.flush()

    print(f"[INFO] Done. Metadata CSV: {OUT_CSV_PATH}")
    print(f"[INFO] PDFs saved under: {OUT_DIR_PDFS}")

if __name__ == "__main__":
    main()
