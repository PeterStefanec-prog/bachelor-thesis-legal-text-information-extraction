#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NS SR (Najvyšší súd SR) – downloader tailored for contractual penalty (zmluvná pokuta) cases.

Optimalizácie:
- Hľadanie po MESIACOCH (nie po dňoch) cez `searchDecision` s art_obsah.
- Používa iba formát dátumu YYYY-MM-DD (format 1 z tvojich logov).
- Thread-local requests.Session pre getDecision + sťahovanie PDF (bez per-ID Session).
- Žiadny zbytočný sleep vo workeroch.
- Nepreťahuje PDF, ak súbor už existuje.
"""

import csv
import json
import os
import re
import time
from datetime import datetime, timedelta
from urllib.parse import urljoin
import threading

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================
# Global configuration
# =====================================

BASE_API = "https://www.nsud.sk/ws/opendata.php"
BASE_SITE = "https://www.nsud.sk"

# Date window (inclusive) – môžeš nastaviť napr. "2010-01-01" až "2025-12-31"
DATE_START = "2010-01-01"
DATE_END   = "2010-12-31"

# Core search terms – musí byť plný tvar, API nepodporuje stemmy
OBSAH_TERMS = [
    # core – contractual penalty (various cases / diacritics)
    "zmluvná pokuta",
    "zmluvnu pokutu",
    "zmluvnú pokutu",
    "zmluvnej pokuty",
    "zmluvné pokuty",
    "zmluvne pokuty",

    # contractual sanction (slightly broader, but still on-point)
    "zmluvná sankcia",
    "zmluvnu sankciu",
    "zmluvnú sankciu",
    "zmluvnej sankcie",
    "zmluvné sankcie",
    "zmluvne sankcie",

    # more specific phrases (still directly about contractual penalty)
    "zníženie zmluvnej pokuty",
    "moderácia zmluvnej pokuty",
    "neprimeraná zmluvná pokuta",
    "primeranosť zmluvnej pokuty",
]

# Collegia to keep:
# 1 = civil, 2 = commercial. Tu ponechávame len obchodné.
REQUIRED_KOLEGIUMS = {2}

# Exclude execution cases (where the core is enforcement, not the merits of the penalty)
EXCLUDE_EXECUTION = True

# Exclude typical "procedural only" dockets (e.g. Ndob – proposals, transfer of cases)
EXCLUDE_NDOB = True

# Exclude bankruptcy / insolvency cases (ZKR, konkurz, oddlženie)
EXCLUDE_ZKR = True

ZKR_KEYWORDS = [
    "zákon č. 7/2005", "zákona č. 7/2005", "zkr",
    "konkurz", "konkurze",
    "oddlžen", "oddĺžen",  # odlíži aj "oddlženie", "oddĺženie"
]

# Exclude clearly procedural merito (transfer of case, venue etc.)
EXCLUDE_PROCEDURAL_ONLY = True

PROCEDURAL_MERITO_KEYWORDS = [
    "prikázanie veci", "prikazanie veci",
    "prikázanie sporu", "prikazanie sporu",
    "návrh na prikázanie", "navrh na prikazanie",
    "určenie príslušnosti", "urcenie prislusnosti",
    "určenie miestnej príslušnosti", "urcenie miestnej prislusnosti",
    "nesúhlas s postúpením", "nesuhlas s postupenim",
]

# Output locations
OUT_DIR_PDFS = "data/nsud_pdfs3"
OUT_DIR_JSON = "data/nsud_json3"
OUT_CSV_PATH = "data/nsud_metadata3.csv"

# Network tuning
MAX_WORKERS = 16  # viac workerov – I/O bound úloha
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30
PER_REQUEST_SLEEP_SEARCH = 0.02  # rate limit len pri searchDecision
SESSION_HEADERS = {
    "User-Agent": "NSUD-OpenData-Downloader/2.0",
    "Accept": "application/json,text/*;q=0.9,*/*;q=0.1",
}

# Thread-local Session pre workerov
_thread_local = threading.local()

# =====================================
# Helper functions
# =====================================


def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def _month_ranges(start_dt: datetime, end_dt: datetime):
    """
    Yield (start_str, end_str) pre každý mesiac v tvare YYYY-MM-DD,
    pretože NS SR API očakáva tento formát v parametroch art_datum_od / art_datum_do.
    """
    cur = datetime(start_dt.year, start_dt.month, 1)
    while cur <= end_dt:
        if cur.month == 12:
            next_month = datetime(cur.year + 1, 1, 1)
        else:
            next_month = datetime(cur.year, cur.month + 1, 1)

        month_end = next_month - timedelta(days=1)

        real_start = max(cur, start_dt)
        real_end = min(month_end, end_dt)
        if real_start <= real_end:
            # kľúčová zmena je TU:
            yield real_start.strftime("%Y-%m-%d"), real_end.strftime("%Y-%m-%d")

        cur = next_month




def _robust_get_json(session: requests.Session, params: dict):
    """GET JSON s retry a základným error handlingom."""
    retries = 3
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(
                BASE_API,
                params=params,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"GET failed for params={params}: {last_exc}")


def _extract_ids(payload) -> list[int]:
    """Extract decision IDs from various shapes of API payload."""
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "id" in payload[0]:
            return [int(x["id"]) for x in payload if "id" in x]
        return [int(x) for x in payload]
    if isinstance(payload, dict):
        for k in ("ids", "data", "result"):
            if k in payload and isinstance(payload[k], list):
                return _extract_ids(payload[k])
    return []


def _search_ids_range(session: requests.Session, date_from_str: str, date_to_str: str, obsah_term: str) -> list[int]:
    """
    Search decision IDs pre dátumový rozsah [date_from_str, date_to_str] (formát DD.MM.RRRR)
    a jeden výraz v art_obsah.
    """
    params = {
        "searchDecision": "",
        "art_datum_od": date_from_str,   # napr. "01.01.2010"
        "art_datum_do": date_to_str,     # napr. "31.01.2010"
        "art_obsah": obsah_term,
    }
    data = _robust_get_json(session, params)
    ids = _extract_ids(data)
    if ids:
        print(f"[INFO] {date_from_str}..{date_to_str} term='{obsah_term}' -> {len(ids)} ids")
    return ids



def _get_worker_session() -> requests.Session:
    """
    Thread-local Session pre getDecision + PDF download.
    V každom threade sa vytvorí raz a recykluje sa.
    """
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(SESSION_HEADERS)
        _thread_local.session = s
    return _thread_local.session


def _get_decision(session: requests.Session, dec_id: int) -> dict | None:
    """Fetch full decision metadata for a single ID."""
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
    """Construct candidate PDF URLs from the 'subor' field."""
    if not subor_value:
        return []
    s = subor_value.strip()
    if s.startswith(("http://", "https://")):
        return [s]
    if s.startswith("/data/att/"):
        return [urljoin(BASE_SITE, s)]
    if s.startswith("data/att/"):
        return [urljoin(BASE_SITE + "/", s)]
    # Fallback guesses
    return [
        urljoin(BASE_SITE + "/", "data/att/" + s.lstrip("/")),
        urljoin(BASE_SITE + "/", s.lstrip("/")),
    ]


def _safe_filename(s: str) -> str:
    """Make a filesystem-safe filename."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s or "").strip("_") or "file"


def _download_pdf(session: requests.Session, urls: list[str], out_path: str) -> tuple[str, int, str]:
    """
    Try to download a PDF from candidate URLs.

    Returns:
        (status_label, http_status, used_url)
        status_label ∈ {"ok", "http_404", "http_other"}
    """
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
# Main logic
# =====================================


def main():
    # 1) Validate date window
    start_dt = _parse_date(DATE_START)
    end_dt = _parse_date(DATE_END)
    if end_dt < start_dt:
        raise ValueError("DATE_END must be >= DATE_START")

    # Session použité len na searchDecision (single-threaded, safe)
    search_session = requests.Session()
    search_session.headers.update(SESSION_HEADERS)

    # 2) Prepare CSV header and output dirs
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

    # 3) Collect all IDs by MONTHS and terms (art_obsah)
    all_ids: set[int] = set()
    for month_start_iso, month_end_iso in _month_ranges(start_dt, end_dt):
        for term in OBSAH_TERMS:
            try:
                ids = _search_ids_range(search_session, month_start_iso, month_end_iso, term)
                if ids:
                    all_ids.update(ids)
            except Exception as e:
                print(f"[WARN] searchDecision failed for {month_start_iso}..{month_end_iso} term='{term}': {e}")
            time.sleep(PER_REQUEST_SLEEP_SEARCH)

    all_ids = sorted(all_ids)
    print(f"[INFO] Collected {len(all_ids)} unique candidate IDs.")

    # 4) Process each ID: filter collegium, optionally exclude executions/procedural,
    #    download PDF, store JSON + CSV row

    def _process_one(dec_id: int) -> dict | None:
        """
        Process one decision:
        - fetch metadata via getDecision
        - filter by collegium, docket type and execution/procedural status
        - download PDF (if not already present)
        - save JSON metadata
        - return CSV row dict
        """
        session = _get_worker_session()

        meta = _get_decision(session, dec_id)
        if not meta:
            print(f"[WARN] getDecision returned nothing for ID={dec_id}")
            return None

        # Collegium filter
        try:
            kol = int(meta.get("kolegium", -1))
        except Exception:
            return None

        if kol not in REQUIRED_KOLEGIUMS:
            return None

        cislo = str(meta.get("cislo") or "")

        # Exclude Ndob dockets – typical proposals / venue / transfer decisions
        if EXCLUDE_NDOB and "Ndob" in cislo:
            return None

        # Combine merito + obsah for generic keyword screening
        preview = (
            str(meta.get("merito", "")) + " " +
            str(meta.get("obsah", ""))
        ).lower()

        # 1) Exclude execution cases
        if EXCLUDE_EXECUTION:
            if any(x in preview for x in ["exekuč", "exekúcia", "exekučné konanie"]):
                return None

        # 2) Exclude bankruptcy / insolvency (ZKR) cases
        if EXCLUDE_ZKR:
            if any(kw in preview for kw in ZKR_KEYWORDS):
                return None

        # 3) Exclude clearly procedural merito (transfer, venue, etc.)
        if EXCLUDE_PROCEDURAL_ONLY:
            merito = (meta.get("merito") or "").lower()
            if any(kw in merito for kw in PROCEDURAL_MERITO_KEYWORDS):
                return None

        # Candidate PDF URLs
        candidates = _candidate_pdf_urls(meta.get("subor"))
        saved_path = ""
        status_label, http_status, used_url = ("http_other", 0, "")

        # Build output filename
        fname = (
            f"{meta.get('id', dec_id)}__"
            f"{_safe_filename((meta.get('datum') or '').replace(' ', '_'))}__"
            f"{_safe_filename(meta.get('cislo') or '')}.pdf"
        )
        out_path = os.path.join(OUT_DIR_PDFS, fname)

        # Ak PDF už existuje, neskúšaj ho sťahovať znova
        if os.path.exists(out_path):
            saved_path = out_path
            status_label, http_status, used_url = ("ok", 200, candidates[0] if candidates else "")
        else:
            if candidates:
                status_label, http_status, used_url = _download_pdf(session, candidates, out_path)
                if status_label == "ok":
                    saved_path = out_path
                else:
                    print(
                        f"[WARN] download failed for ID={dec_id} "
                        f"status={http_status} url={used_url} reason={status_label}"
                    )
            else:
                print(f"[WARN] missing 'subor' for ID={dec_id}")

        # Save full JSON metadata for later analysis
        json_path = os.path.join(OUT_DIR_JSON, f"{meta.get('id', dec_id)}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as fj:
                json.dump(meta, fj, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] failed to save JSON for ID={dec_id}: {e}")
            json_path = ""

        # Build CSV row
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

    # 5) Run processing in a thread pool
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
