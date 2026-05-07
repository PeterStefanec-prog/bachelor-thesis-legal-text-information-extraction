# Retrieval Development Set — `golden_dataset_template.csv`

## What this file is

This CSV is the **RETRIEVAL DEVELOPMENT SET** (aka "retrieval dev set", "retrieval tuning set"). It contains 20 Slovak court decisions with manually annotated ground-truth quotes for 3 query types (Q1 contract context, Q2 penalty details, Q3 moderation analysis).

**Purpose:** used to iteratively tune the hybrid retrieval pipeline:
- `alpha` for RRF dense/BM25 fusion (final value: 0.7)
- Reranker pattern weights (breach/rate/outcome/factors/noise)
- Keyword pattern set (added during v2 fixes)
- `top_k` selection (final: 7 + adaptive +3/+5 for large docs)
- `RERANKER_OVERSAMPLE` (final: 5)

All decisions about retrieval configuration were made based on metrics computed against this 20-doc set.

## What this file is NOT

This is **NOT** the same set used for lawyer evaluation of LLM extractions.

The **EXTRACTION TEST SET** for lawyer evaluation is defined in:
- `src/evaluation_extraction/generate_evaluation_sheets.py` (variable `EXTRACTION_TEST_SET`, aliased as `EVAL_DOCS`)
- Documented in `data/07_extractions/evaluation_20_selection.md`

**Overlap between the 2 sets:** 13 documents (shared)
**Retrieval-only docs (in this CSV but not in extraction eval):** 7
**Extraction-only docs (in extraction eval but not in this CSV):** 7

## Why two different sets?

To prevent the criticism that retrieval is evaluated on the same docs it was tuned on. The 7 held-out docs in `EXTRACTION_TEST_SET` were NEVER seen by the retrieval tuning loop. When lawyers evaluate extraction quality on those 7 docs, it tests whether the tuned retrieval config generalizes to unseen documents.

This is standard ML methodology: dev set for tuning, test set for final evaluation (with partial overlap acceptable here because we're not re-training a neural model, only tuning hyperparameters).

## Docs that are in this CSV but NOT in EXTRACTION_TEST_SET

These 7 docs were part of original retrieval tuning. After legal audit (April 2026), they were dropped from lawyer evaluation — typically because they weren't obchodnoprávne (commercial law) or lacked relevant legal richness:

- `KS_Banská_Bystrica_43CoPv_10_2023` — civil industrial-property (not commercial)
- `KS_Bratislava_1Cob_130_2019`
- `KS_Bratislava_1Cob_40_2018`
- `KS_Bratislava_2Co_380_2011`
- `KS_Košice_4Cob_116_2023`
- `KS_Trnava_32Cob_3_2021`
- `NS_SR_1Cdo_85_2023` — civil code NS SR (not obchodnoprávne dovolanie)

## Scripts that read this file

- `src/evaluation_retrieval/evaluate_retrieval.py` — main retrieval evaluation
- `src/evaluation_retrieval/diagnose_retrieval_mE5_failures.py` — debug analysis
- `src/chunking/chunk_processor.py` — uses for chunking analysis
- `notebooks/04_retrieval_experiments.ipynb`
- `notebooks/05_gold_chunk_analysis.ipynb`
- `notebooks/06_retrieval_comparison.ipynb`

**Scripts that previously read this file but shouldn't (now fixed):**
- `src/extraction/run_extraction.py` — was reading this for `GOLDEN_ONLY=True`, fixed on 2026-04-16 to use `EXTRACTION_TEST_SET` instead.

## Renaming

The file is still called `golden_dataset_template.csv` (not renamed to `retrieval_dev_set.csv`) because 8 scripts/notebooks reference this path. Renaming would cascade through notebooks that have cached run outputs tied to this path. The README here clarifies the semantic purpose.
