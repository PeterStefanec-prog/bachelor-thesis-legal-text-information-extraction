"""
Stage 3: Ranking - rank filtered penalties by how similar they are to query.
It narrows the filter.

After Stage 2 filters to subset (e.g. 49 penalties matching "dielo + late_payment"),
this module ranks them by similarity to the lawyer's specific situation.

I use 4 dimensions:
  1. Embedding similarity (40%) - semantic match between query and penalty card text
  2. Amount proximity (25%) - how close contract amounts are
  3. Bonus attributes (20%) - extra points for matching decision, factors
  4. Authority (15%) - Supreme Court ranks higher than Regional court

Why these weights? After Stage 2 already filtered by contract_type and breach_type,
those fields are same for all filtered records.
So ranking needs to differentiate on OTHER things - embedding captures semantic
similarity, amount captures financial proximity, bonus captures specific legal
attributes, authority captures legal weight.
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# #########################################
# EMBEDDING MODEL
# #########################################
# same model as in build_penalty_index.py and in my retrieval pipeline
EMBEDDING_MODEL = "text-embedding-3-small"


# #########################################
# RANKING WEIGHTS
# #########################################
# these are my starting weights based on domain reasoning, but mainly on experiments
# embedding gets most weight because it captures "situational similarity"
# that simple field matching cant - e.g. "argumentacia funkciou pokuty"
# matches "zabezpecovacia funkcia" through embedding but not through fields.

W_EMBEDDING = 0.40      # 40%
W_AMOUNT = 0.25         # 25%
W_BONUS = 0.20
W_AUTHORITY = 0.15


# #########################################
# COMPUTE QUERY EMBEDDING
# #########################################

def _compute_query_embedding(semantic_query):
    """Embed semantic query using same model as penalty cards.

    I embed the semantic_query (reformulated by LLM in Stage 1) not the raw user query
    - LLM rewrites it to be closer in form to penalty card descriptions, which should
    give better cosine similarity.
    """
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env

    response = client.embeddings.create(
        input=[semantic_query],
        model=EMBEDDING_MODEL,
    )

    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)    # return embedding as numpy array


# #########################################
# SCORE FUNCTIONS
# #########################################

def _embedding_scores(query_embedding, penalty_embeddings):
    """Cosine similarity between query and all penalty embeddings (penalty cards).

    text-embedding-3-small returns normalized vectors so dot product = cosine similarity
    I do it all at once with numpy.
    Returns array of scores, one per penalty.
    """
    # dot product of query vector with every penalty vector
    sims = np.dot(penalty_embeddings, query_embedding)

    # cosine can be slightly negative for very different texts, clip to [0, 1]
    return np.clip(sims, 0.0, 1.0)


def _amount_score(query_amount, penalty_record):
    """How close is the penalty's amount to what lawyer specified.

    I use ratio: min(a,b) / max(a,b) which gives:
      50k vs 48k = 0.96 (very similar)
      50k vs 100k = 0.50 (moderately similar)
      50k vs 500k = 0.10 (not similar)

    I prefer secured_principal but fall back to original_claimed if missing
    (only 39% of penalties have secured_principal).
    """
    # if lawyer didnt specify amount, give neutral score (dont penalize)
    if query_amount is None or query_amount <= 0:
        return 0.5

    # try secured_principal first, then original_claimed as fallback
    penalty_amount = penalty_record.get("secured_principal")
    if penalty_amount is None:
        penalty_amount = penalty_record.get("original_claimed")

    # if penalty has no amount data, give neutral score
    if penalty_amount is None or penalty_amount <= 0:
        return 0.5

    # ratio of smaller to larger - always between 0 and 1
    smaller = min(query_amount, penalty_amount)
    larger = max(query_amount, penalty_amount)
    return smaller / larger


def _bonus_score(intent, penalty_record):
    """Bonus points for decision_interest and factor_interest.

    These used to be hard filters in Stage 2, but that caused tautological statistics
    (filtering on decision=awarded_full -> 100% awarded, duh).
    Now they are SOFT ranking signals - matching cases rank higher but non-matching
    cases stay in pool for meaningful analytics.

    Returns score between 0 and 1.
    """
    score = 0.0
    max_possible = 0.0

    # decision match - if lawyer interested in moderation, moderated cases rank higher
    # but upheld cases stay in pool so we can compute factor lift
    dec_interest = intent.get("decision_interest")
    if dec_interest:
        max_possible += 2.0
        if penalty_record.get("decision") == dec_interest:
            score += 2.0

    # factor overlap - if lawyer mentioned specific factors, penalties where
    # those factors were mentioned get bonus points
    factors_interest = intent.get("factor_interest", [])
    if factors_interest:
        max_possible += len(factors_interest) * 1.0

        # collect which factors are mentioned in this penalty (not "not_mentioned")
        mentioned_factors = set()
        for f in penalty_record.get("factors", []):
            sentiment = f.get("sentiment")
            if sentiment is not None and sentiment != "not_mentioned":
                mentioned_factors.add(f["label"])

        # give point for each matching factor
        for factor_label in factors_interest:
            if factor_label in mentioned_factors:
                score += 1.0

    # if lawyer didnt specify any decision or factors, return neutral
    if max_possible == 0:
        return 0.5

    return score / max_possible


def _authority_score(penalty_record):
    """Score based on court authority level.

    Supreme Court (level 3) > Regional Court (level 2) > District (level 1).
    I used to also add recency (year of decision), but removed it - with only
    ~20 years of data the recency bonus was basically noise (~0.01 impact on
    final score) and added complexity without real effect on ranking.
    """
    # authority_level is 1-3, i normalize to 0-1 range
    auth = penalty_record.get("authority_level", 1)
    return auth / 3.0


# #########################################
# MAIN RANKING FUNCTION
# #########################################

def rank_penalties(filtered_records, filtered_embeddings, intent, top_n=5):
    """Rank filtered penalties by multi-dimensional similarity to query.

    Returns list of (record, score) tuples sorted by score descending, top N.
    Default top_n=5 - pipeline shows top 5 PENALTIES (not 5 court decisions -
    one decision can have multiple penalties, each becomes its own precedent card).
    """
    if not filtered_records:
        return []

    start = time.time()

    # compute query embedding
    semantic_query = intent.get("semantic_query", "")
    if not semantic_query:
        semantic_query = "zmluvna pokuta moderacia"  # fallback

    # calling openAI API
    query_embedding = _compute_query_embedding(semantic_query)

    # compute embedding similarity for all penalties at once (numpy vectorized)
    emb_scores = _embedding_scores(query_embedding, filtered_embeddings)

    # compute score for each penalty individually
    query_amount = intent.get("amount_hint")
    results = []

    for i in range(len(filtered_records)):
        rec = filtered_records[i]
        # for each record compute
        emb = float(emb_scores[i])
        amt = _amount_score(query_amount, rec)
        bonus = _bonus_score(intent, rec)
        auth = _authority_score(rec)

        # weighted combination of all 4 dimensions
        final = (W_EMBEDDING * emb +
                 W_AMOUNT * amt +
                 W_BONUS * bonus +
                 W_AUTHORITY * auth)

        results.append((rec, final))

    # sort by final score (second element from tuple) - best match first
    results.sort(key=lambda x: x[1], reverse=True)

    latency = round(time.time() - start, 2)

    # take top N
    top_results = results[:top_n]

    # print ranking summary
    n_returned = min(top_n, len(results))
    print(f"  Ranked {len(filtered_records)} penalties in {latency}s, returning top {n_returned}")
    if top_results:
        best_rec = top_results[0][0]
        best_score = top_results[0][1]
        print(f"  Best match: {best_rec['case_number']} {best_rec['penalty_id']} "
              f"(score={best_score:.3f})")

    return top_results
