# Preprocessing Pipeline - Notes and Issues

File: `src/preprocessing/data_cleaner.py`
These are my working notes about what the cleaning pipeline does and what problems i had to fix.

---

## What this script does (overview)

Takes a raw PDF of a court decision and outputs a structured JSON. Steps are numbered to match chapter 3 in spec i uploaded on BP1.

```
PDF -> text per page -> light unicode -> metadata extraction
    -> remove pagination -> layout normalization
    -> advanced unicode -> segmentation -> JSON
```

---

## Steps

### STEP 3.1 - PDF text extraction (`pdfplumber`)

Extract text page by page using `pdfplumber`. For each page i call `page.extract_text()` which returns a raw string. Pages are stored in `raw_pages` list.

Note: pdfplumber sometimes puts multiple paragraphs on one line when the PDF doesnt have explicit paragraph breaks. This cant be fixed in the cleaner, its a limitation of PDF extraction. More in Known Limits section at the bottom.

---

### STEP 3.2 - Light unicode normalization (`clean_light`)

First, safe wave of cleaning. Only things that are 100% safe to do:

- NFC normalization (unify character representation)
- Remove invisible characters: zero-width space `\u200b`, soft hyphen `\u00ad`
- Unify line endings: `\r\n` and `\r` to `\n`

This runs before metadata extraction, because i want to extract metadata from clean text, not from text with invisible garbage in it.

---

### STEP 3.3 - Metadata extraction from header (`extract_meta`)

From first ~2000 characters of first page i extract metadata using regexes:

| Field | Regex / source |
|-------|---------------|
| `date_raw` | `\d{1,2}\.\s*\d{1,2}\.\s*\d{4}` |
| `case_id` | after keyword `Spisová značka:` |
| `ecli` | after prefix `ECLI:` |
| `court` | after keyword `Súd:` |
| `document_type` | looking for word `ROZSUDOK` or `UZNESENIE` |

`court` and `document_type` were added later, originally i only extracted the first 3 fields. I went back to add them because i needed the court name and decision type in the processed JSON for the evaluation later. Easy to add because the header format is standardized across all courts on the justice.sk portal.

---

### STEP 3.4 - Remove pagination and split header (`remove_pagination`, `split_header_body`)

#### `remove_pagination`
Goes through text line by line and removes:
- `Strana X` / `Strana X z Y`
- `- X -` (page numbers in dashes)
- `Pokračovanie` / `Pokračovanie...`

Bug i fixed: Original regex was `\-?\s*\d+\s*\-?`, the dashes were optional (`?`). This means a bare number like `3` or `12` on its own line would also get deleted. This was a problem because court decisions have numbered paragraphs and sometimes the number stands alone on a line after PDF extraction. I fixed this by making dashes required: `^\-\s*\d+\s*\-$`. Now standalone digits are safe.

Also added `Pokračovanie` removal, which i forgot originally even though it was in my thesis spec. Some decisions have "Pokračovanie" printed at the top of continuation pages.

#### `split_header_body`
Looks for regex `\nROZSUDOK\n` or `\nUZNESENIE\n` and splits text there into:
- `header_text`, the metadata block (court name, case id, ECLI...)
- `body_text`, actual decision starting from the word Rozsudok/Uznesenie

Fallback: if those words not found, use position of ECLI line. Last resort is hard cut after 1000 characters.

---

### STEP 3.5 - Layout normalization (`normalize_layout`)

This was the most complex step. PDF extraction typically breaks long sentences across multiple lines based on where the page or column ended. My rules decide when to join lines and when to keep them separate.

#### Decision logic (simplified)

```
for each line:
    if empty line: flush buffer, write empty line (paragraph boundary)

    else compare previous (prev) and current (line):

    A) ALWAYS JOIN:
        - prev ends with hyphen (split word)
        - prev ends with preposition (v, z, zo, k, ku...)
        - prev ends with abbreviation (ods., písm., č., sp. zn., JUDr., ...)
        - prev is a number and line starts with "- number" (number range)

    B) NEVER JOIN (STOP):
        - line looks like spaced header (o d ô v o d n e n i e)
        - prev ends with "SLOVENSKEJ REPUBLIKY"
        - line starts with list item (1., a), bullet)
        - line starts with uppercase letter and prev ends with . ? !

    C) GENERAL RULE (if doesnt match A or B):
        - join if prev doesnt end with . ? !
        - join if line starts with lowercase letter
```

I preserve paragraph breaks `\n\n`. At the end i join with `"\n\n".join(...)`.

Originally i was joining with `"\n"` and it looked fine when reading the .txt files. But then the chunker couldnt find paragraph boundaries and was making bad chunks. Took me a while to figure out why chunking was so bad, turned out it was because the preprocessing removed the paragraph boundaries. Fixed by switching to `"\n\n"`.

---

### Helper function `fix_spaced_words`

Some PDFs have section headings printed with spaces between each letter: `o d ô v o d n e n i e`, `r o z h o d o l`. This function joins them back into normal words.

Bug i fixed: Original regex had `(?!\S)` at the end, meaning "after the word there must not be a non-whitespace character". This worked fine for spaced words in the middle of a sentence, but failed when the spaced word was right before punctuation.

Example of the failure:
```
'I. Dovolanie o d m i e t a.'  became  'I. Dovolanie odmiet a.'   BUG
```

The last `a` is followed by `.` which is `\S`, so `(?!\S)` failed and regex stopped one letter too early. The word "odmieta" (rejects) got split into "odmiet" and "a".

Fixed `(?!\S)` to `(?=[\s.,;:!?()\[\]]|$)`, meaning "after the word there can be whitespace, punctuation, or end of string". After the fix:
```
'I. Dovolanie o d m i e t a.'  became  'I. Dovolanie odmieta.'    correct
```

I kept the threshold at `{2,}` (at least 2 repetitions of letter+space) on purpose. Lowering it to `{1,}` would also fix two-letter spaced words like `m á`, but the risk of accidental false positives on normal text felt too high so i left it conservative.

---

### STEP 3.6 - Advanced unicode normalization (`advanced_unicode`)

Second wave of unicode cleaning, intentional changes not just "removing garbage":

- Different dash types (en-dash, em-dash) all become regular hyphen `-`
- Different quote types (typographic quotes) all become regular `"`

This runs after layout normalization, not before, because during line joining i need to be able to tell the difference between a dash in a sentence vs a word-splitting hyphen at end of line. If i normalized dashes before layout, id lose that information.

---

### STEP 3.7 - Segmentation (`segment_body`)

Splits `body_text` into 3 logical parts using keywords:

| Segment | Keyword | Note |
|---------|---------|------|
| `verdict` | (none) | text before odôvodnenie |
| `reasoning` | `odôvodnenie` / `o d ô v o d n e n i e` | regex handles both spaced and normal version |
| `instruction` | `poučenie` / `p o u č e n i e` | same approach |

Fallback: if `odôvodnenie` is not found (e.g. shortened decisions), entire text goes into `verdict`. I log when this happens so i can track how often it occurs.

---

## Output format (JSON)

```json
{
    "filename": "NS_SR_1Cdo_85_2023_00_dokument.pdf",
    "metadata": {
        "date_raw": "28.05.2025",
        "case_id": "1Cdo/85/2023",
        "ecli": "ECLI:SK:NSSR:2025:3819205466.1",
        "court": "Najvyšší súd",
        "document_type": "Uznesenie"
    },
    "header": "Súd: Najvyšší súd\nSpisová značka: ...",
    "segments": {
        "verdict": "Uznesenie Najvyšší súd...",
        "reasoning": "odôvodnenie :\n\n1. Okresný súd...",
        "instruction": "Poučenie: Proti tomuto..."
    }
}
```

---

## Fixed bugs (summary)

| # | Where | Problem | Fix |
|---|-------|---------|-----|
| 1 | `fix_spaced_words` | `(?!\S)` failed before punctuation, "odmiet a." | Changed to `(?=[\s.,;:!?()\[\]]|$)` |
| 2 | `remove_pagination` | Optional dashes `\-?` were deleting bare numbers like `3`, `12` | Made dashes required: `^\-\s*\d+\s*\-$` |
| 3 | `normalize_layout` | Paragraphs joined with `\n` instead of `\n\n`, chunker couldnt find boundaries | Changed to `"\n\n".join(...)` |
| 4 | `extract_meta` | Missing fields `court` and `document_type` | Added two new regexes |
| 5 | `remove_pagination` | `Pokračovanie` was not removed despite being in spec | Added new regex condition |

---

## Running the script

Originally the script used relative path `../../data` which only worked from the `src/preprocessing/` folder. Changed to `data` so it works when running from project root:

```bash
.venv/bin/python src/preprocessing/data_cleaner.py
```

This processes all PDFs in `data/01_raw_pdfs/` and writes JSONs to `data/02_processed_json/`.

---

## Known limits

pdfplumber merges paragraphs onto one line in some documents. Older or scan-to-PDF documents dont have paragraph structure embedded in the PDF. In that case pdfplumber returns multiple paragraphs on a single line. The cleaner cant do anything about this because it doesnt have access to pixel-level layout information. In practice i saw this as merged paragraphs in one decision (paragraphs 35 and 36 ended up joined). I will monitor how often this happens in the corpus and if needed switch to a more robust extractor.
