# INPUT
#           data/07_extractions/gpt-4o_fulldoc/ - can be changes
# OUTPUT
#           data/10_penalty_index/penalty_index.jsonl  - one JSON line per penalty
#           data/10_penalty_index/embeddings.npy       - numpy array of penalty card embeddings [N x 1536]

# Run from project root:    - run only when extraction was changed
#   export OPENAI_API_KEY="sk-..."
#   .venv/bin/python src/multi_doc/build_penalty_index.py

"""
Build the penalty-level index for multi-document precedent search.

This is one-time preprocessing script.
I read all extraction JSONs (one per  decision) and create flat index where each row = one contractual penalty (not one document!)

Why penalty-level and not document-level?
  One court decision can have multiple penalties with diffrent breach types, rates, outcomes.
  For example KS_Trencin_8Cob_38_2012 has pokuta_1 (dismissed, non_monetary_performance) and pokuta_2 (moderated_301, early_termination).
  If i filter at document level, i could get false matches - document matching "late_payment + awarded_full" when  those are 2 different penalties, neither of which is late_payment + awarded_full.
  I checked - 8.5% of documents have multiple penalties with mixed attributes.

"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

# project root setup (same pattern as run_extraction.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


# #########################################
# CONFIGURATION
# #########################################

# reading extraction jsons from (gpt-4o with RAG mode)
# EXTRACTION_DIR = "data/07_extractions/gpt-4o_rag"
EXTRACTION_DIR = "data/07_extractions/gpt-4o_fulldoc"

#  saving penalty index
OUTPUT_DIR = "data/10_penalty_index"

# embedding model - same as in my retrieval pipeline (src/retrieval/vector_store.py)
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions


# #########################################
# AUTHORITY LEVEL
# #########################################
# for ranking later - Supreme Court decision has more legal weight than regional court.
# in my dataset i have: 8 "Krajsky sud" (regional) + 1 "Najvyssi sud" (supreme)

def get_authority_level(court_name):
    """Parse court name and return authority level number.
    3 = Supreme Court, 2 = Regional Court, 1 = District or unknown.
    """
    if not court_name:
        return 1

    name_lower = court_name.lower()

    # "Najvyssi" handles diacritics too (najvy matches both najvyssi and najvyšší)
    if "najvy" in name_lower:
        return 3                    # the most valuable from najvyssi sud
    elif "krajsk" in name_lower:
        return 2
    elif "okresn" in name_lower:        # lthe least valuable from okresny sud
        return 1

    return 1


# #########################################
# PENALTY CARD TEXT
# #########################################
# i generate short text description for each penalty - its "signature"
# this text gets embedded and used for semantic similarity search in ranking (Stage 3)
# when lawyer searches "zmluva o dielo, omeskanie, 50k EUR", the embedding of
# their query should be close to embeddings of penalty cards from similar cases

def build_penalty_card_text(rec):
    """Build short text summary of  penalty for embedding.

    I put the most important attributes here: contract type, breach, rate, amounts, decision, factors, and reasoning summary.
    Text is ~60-100 tokens - good size for embedding

    Input - Penalty record (flat)
    """
    parts = []

    # contract basics
    ct = rec.get("contract_type", "neznamy")
    rt = rec.get("relationship_type", "")
    if rt:
        parts.append(f"Zmluva: {ct}, {rt}.")
    else:
        parts.append(f"Zmluva: {ct}.")

    # breach type
    bt = rec.get("breach_type", "")
    if bt:
        parts.append(f"Porusenie: {bt}.")

    # rate
    rate_raw = rec.get("rate_value_raw", "")
    if rate_raw:
        parts.append(f"Sadzba: {rate_raw}.")    # for example Sadzba: 0,05 % denne z dlžnej sumy.

    # amounts (only if we have them)
    principal = rec.get("secured_principal")
    currency = rec.get("currency", "EUR")
    if principal is not None:
        # format number with spaces as thousand separator (e.g. 50 000)
        principal_str = f"{principal:,.0f}".replace(",", " ")
        parts.append(f"Istina: {principal_str} {currency}.")

    claimed = rec.get("original_claimed")
    awarded = rec.get("final_awarded")
    if claimed is not None and awarded is not None:
        claimed_str = f"{claimed:,.0f}".replace(",", " ")
        awarded_str = f"{awarded:,.0f}".replace(",", " ")
        parts.append(f"Pozadovane: {claimed_str}, priznane: {awarded_str} {currency}.")

    # decision
    decision = rec.get("decision", "")
    decision_labels = {
        "awarded_full": "Potvrdena v plnej vyske",
        "moderated_301": "Znizena podla §301",
        "dismissed": "Zamietnuta",
        "returned": "Vratena na doplnenie",
        "unclear": "Nejasne",
    }
    if decision:
        label = decision_labels.get(decision, decision)
        parts.append(f"Rozhodnutie: {label}.")

    # key factors (only those that court actually mentioned)
    factors = rec.get("factors", [])
    factor_strs = []
    for f in factors:
        sentiment = f.get("sentiment")
        if sentiment is not None and sentiment != "not_mentioned":
            factor_strs.append(f"{f['label']} ({sentiment})")
    if factor_strs:
        parts.append(f"Faktory: {', '.join(factor_strs)}.")

    # reasoning summary - the most semantically rich part
    # i trim to 200 chars so the card doesnt get too long
    reasoning = rec.get("legal_reasoning_summary", "")
    if reasoning:
        if len(reasoning) > 200:
            # cut at last space before 200 chars so i dont break a word
            trimmed = reasoning[:200]
            last_space = trimmed.rfind(" ")
            if last_space > 0:
                trimmed = trimmed[:last_space]
            parts.append(trimmed)
        else:
            parts.append(reasoning)

    return " ".join(parts)


# #########################################
# FLATTEN ONE PENALTY
# #########################################
# take one penalty object from extraction JSON and create flat dict with everything
#  multi-doc pipeline needs -  so i dont need to look up original document later

def flatten_penalty(doc_data, penalty_obj):
    """Create flat penalty record from extraction JSON.

    doc_data - json from courst decision
    penalty_obj is one item from doc_data["result"]["contractual_penalties"]

    So yeah it takes  full document extraction and one penalty from contractual_penalties array.
    Returns dict with all fields needed for filtering, ranking, analytics, display.
    """
    meta = doc_data["result"]["meta"]
    ctx = doc_data["result"]["case_context"]    # case context
    qc = doc_data["result"].get("quality_control", {})  # quality control

    # basic document info
    rec = {
        "doc_id": meta.get("doc_id", ""),
        "court_name": meta.get("court_name", ""),
        "case_number": meta.get("case_number", ""),
        "decision_date": meta.get("decision_date", ""),
        "ecli": meta.get("ecli"),
        "document_type": meta.get("document_type", ""),
        "authority_level": get_authority_level(meta.get("court_name", "")),
    }

    # source file - i can reconstruct it from doc_id (might need later for chunk retrieval)
    rec["source_file"] = rec["doc_id"] + ".pdf"

    # case context (same for all penalties in one document)
    rec["contract_type"] = ctx.get("contract_type", "nezname")
    rec["relationship_type"] = ctx.get("relationship_type", "unknown")
    rec["dispute_summary"] = ctx.get("dispute_summary", "")
    rec["verdict_summary"] = ctx.get("verdict_summary", "")

    # penalty-specific fields
    rec["penalty_id"] = penalty_obj.get("penalty_id", "pokuta_1")
    rec["related_claim_ref"] = penalty_obj.get("related_claim_ref")

    # breach type
    bt = penalty_obj.get("breach_type", {})
    rec["breach_type"] = bt.get("value", "other")

    # rate definition
    rd = penalty_obj.get("rate_definition", {})
    rec["rate_type"] = rd.get("type", "ine")
    rec["rate_value_raw"] = rd.get("value_raw", "")

    # amounts - extract numeric values, keep None if missing
    amounts = penalty_obj.get("amounts", {})
    rec["currency"] = amounts.get("currency", "unknown")

    principal_obj = amounts.get("secured_principal", {})
    rec["secured_principal"] = _safe_number(principal_obj.get("value"))

    claimed_obj = amounts.get("original_claimed", {})
    rec["original_claimed"] = _safe_number(claimed_obj.get("value"))

    awarded_obj = amounts.get("final_awarded", {})
    rec["final_awarded"] = _safe_number(awarded_obj.get("value"))

    # associated interest
    interest = penalty_obj.get("associated_interest", {})
    rec["interest_awarded"] = interest.get("awarded", "unclear")
    rec["interest_rate"] = interest.get("rate_value")       # awarded also kumulacia_s_urokom ?

    # moderation analysis
    ma = penalty_obj.get("moderation_analysis", {})
    decision = ma.get("decision", {})
    rec["decision"] = decision.get("value", "unclear")
    rec["dismissal_reason"] = decision.get("dismissal_reason")
    rec["legal_reasoning_summary"] = ma.get("legal_reasoning_summary", "")
    rec["key_quotes"] = ma.get("key_quotes", [])
    rec["factors"] = ma.get("factors", [])

    # quality control flags
    # flags look like "ANONYMIZED_AMOUNT_original_claimed_pokuta_1" or
    # "CURRENCY_MISMATCH_POSSIBLE_SKK_pokuta_1"
    # i only keep flags that are about THIS penalty (contain its penalty_id)
    # plus flags that are global (dont mention any specific pokuta_)
    all_flags = qc.get("flags", [])
    pid = rec["penalty_id"]
    relevant_flags = []
    for flag in all_flags:
        if pid in flag:
            # this flag is about this specific penalty
            relevant_flags.append(flag)
        elif "pokuta_" not in flag:
            # this is a global flag (not penalty-specific)
            relevant_flags.append(flag)
    rec["flags"] = relevant_flags

    return rec


def _safe_number(val):
    """Convert value to float, return None if not valid number.
    Extraction sometimes returns weird values, this handles it safely.
    """
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# #########################################
# COMPUTE EMBEDDINGS
# #########################################
# same embedding model as in my retrieval pipeline (text-embedding-3-small)
# i embed all penalty cards in one batch API call (much faster than one-by-one)

def compute_embeddings(texts):
    """Compute embeddings for list of texts using OpenAI API.

    Same approach as in src/retrieval/vector_store.py - send all texts in one batch.
    OpenAI supports up to 2048 texts per request, so far i hve 202 now
    Returns numpy array [len(texts), 1536].
    """
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    print(f"  Computing embeddings for {len(texts)} texts...")

    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )

    # sort by index to be safe (API should return in order but just in case)
    sorted_data = sorted(response.data, key=lambda x: x.index)

    # extract embedding vectors into list
    embeddings = []
    for item in sorted_data:
        embeddings.append(item.embedding)

    tokens_used = response.usage.total_tokens
    print(f"  Done. Used {tokens_used} tokens for embedding.")

    return np.array(embeddings, dtype=np.float32)


# #########################################
# MAIN
# #########################################

def main():
    print("-" * 60)
    print("BUILDING PENALTY INDEX")
    print("-" * 60)

    # load all extraction JSONs
    all_files = os.listdir(EXTRACTION_DIR)
    json_files = []
    for f in sorted(all_files):
        if f.endswith(".json"):
            json_files.append(f)
    print(f"\nFound {len(json_files)} extraction files in {EXTRACTION_DIR}")

    # flatten all penalties - one record per penalty, not per document (so if does court have 4 penalties - i have 4 records)
    records = []
    docs_with_multiple = 0

    for filename in tqdm(json_files, desc="Flattening penalties"):
        filepath = os.path.join(EXTRACTION_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        penalties = doc_data["result"]["contractual_penalties"]
        if len(penalties) > 1:
            docs_with_multiple += 1

        for penalty_obj in penalties:
            rec = flatten_penalty(doc_data, penalty_obj)    # call function above - flattening penalty (choosing attributes)
            records.append(rec)

    print(f"\nFlattened {len(records)} penalties from {len(json_files)} documents")
    print(f"  Documents with multiple penalties: {docs_with_multiple}")

    # generate penalty card texts for embedding
    print("\nGenerating penalty card texts...")
    card_texts = []
    for rec in records:
        card_text = build_penalty_card_text(rec)    # calling function above
        rec["penalty_card_text"] = card_text
        card_texts.append(card_text)

    # show sample to check it looks ok
    print(f"\n  Sample penalty card (record 0):")
    print(f"  {card_texts[0][:200]}...")

    # compute embeddings via OpenAI API
    print(f"\nComputing embeddings with {EMBEDDING_MODEL}...")
    embeddings = compute_embeddings(card_texts)
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # save penalty records as JSONL (one JSON per line)
    jsonl_path = os.path.join(OUTPUT_DIR, "penalty_index.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False)
            f.write(line + "\n")
    print(f"\nSaved {len(records)} records to {jsonl_path}")

    # save embeddings as numpy array
    npy_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    np.save(npy_path, embeddings)
    print(f"Saved embeddings to {npy_path} (shape: {embeddings.shape})")

    # print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    from collections import Counter

    # count values for each important field
    ct_counts = Counter()
    bt_counts = Counter()
    dec_counts = Counter()
    auth_counts = Counter()
    curr_counts = Counter()
    for r in records:
        ct_counts[r["contract_type"]] += 1
        bt_counts[r["breach_type"]] += 1
        dec_counts[r["decision"]] += 1
        auth_counts[r["authority_level"]] += 1
        curr_counts[r["currency"]] += 1

    print(f"\nTotal penalties: {len(records)}")
    print(f"\nContract types:  {dict(ct_counts.most_common())}")
    print(f"Breach types:    {dict(bt_counts.most_common())}")
    print(f"Decisions:       {dict(dec_counts.most_common())}")
    print(f"Authority:       {dict(auth_counts.most_common())}")
    print(f"Currencies:      {dict(curr_counts.most_common())}")

    # count how many have usable amounts
    has_principal = 0
    has_claimed = 0
    has_awarded = 0
    for r in records:
        if r["secured_principal"] is not None:
            has_principal += 1
        if r["original_claimed"] is not None:
            has_claimed += 1
        if r["final_awarded"] is not None:
            has_awarded += 1

    total = len(records)
    print(f"\nAmount coverage:")
    print(f"  secured_principal: {has_principal}/{total} ({100*has_principal/total:.1f}%)")
    print(f"  original_claimed:  {has_claimed}/{total} ({100*has_claimed/total:.1f}%)")
    print(f"  final_awarded:     {has_awarded}/{total} ({100*has_awarded/total:.1f}%)")

    # count quality flags - i use any() so each penalty counts at most once per category
    flagged = 0
    anon = 0
    currency_mm = 0
    for r in records:
        if r["flags"]:
            flagged += 1
        if any("ANONYMIZED" in flag for flag in r["flags"]):
            anon += 1
        if any("CURRENCY_MISMATCH" in flag for flag in r["flags"]):
            currency_mm += 1
    print(f"\nQuality flags:")
    print(f"  Total flagged:       {flagged}")
    print(f"  ANONYMIZED_AMOUNT:   {anon}")
    print(f"  CURRENCY_MISMATCH:   {currency_mm}")

    print("\nDone!")


if __name__ == "__main__":
    main()
