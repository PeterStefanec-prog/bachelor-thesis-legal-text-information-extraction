"""
Reranker for Slovak legal court decisions about contractual penalties (zmluvna pokuta).

This is post-processing step that runs AFTER retrieval and BEFORE the LLM.
The problem is that retrieval sometimes returns chunks that are semantically similar to  query
but dont actually contain the useful information.
For example, a chunk about an appeal ("odvolanie") mentions "zmluvna pokuta" many times but doesnt have
 actual penalty amount or the court's reasoning.

So what this does:
  1. Retrieval returns more chunks than we need (e.g. 10 instead of 5)
  2. This reranker scores each chunk using regex patterns for legal keywords
  3. It combines the retrieval score with the keyword score
  4. Returns only the best top_k chunks

I figured out the keywords by reading through 15+ court decisions and noting which
phrases appear in the "useful" chunks vs the "noise" chunks.
The patterns themselves are in legal_patterns.py - centralized there for easier maintenance.
"""

import re

# all patterns are in legal_patterns.py so i dont maintain two copies
from src.reranking.legal_patterns import (
    PATTERNS_FOR_QUERY,
    FIRST_CHUNK_BONUS,      # bonus is applied only if first chunks already has some positive keyword signal
)


# ##############################
# SCORING FUNCTION
# ##############################
# Goes through all patterns for the given query type and adds up the weights for each pattern that matches in the chunk text

def get_keyword_score(text, query_key):
    """
    Score chunk based on how many legal keywords it contains.

    Returns (score, matched_patterns) where matched_patterns is dict showing which patterns matched and how many times.
    I use matched_patterns for debugging - it helps me understand why  chunk got  high or low score.
    """
    pattern_groups = PATTERNS_FOR_QUERY.get(query_key, [])  # get patterns for the query
    score = 0.0
    matched_patterns = {}

    for group in pattern_groups:    # going through groups
        for pattern, weight in group:   # going through exact patterns (pattern - compoiled regex, weight - number like +3, -2)
            matches = pattern.findall(text)
            if matches:
                if weight < 0:
                    # FIX: noise penalties should only count once!
                    # I originally multiplied by match count, but that was wrong.
                    # Chunk mentioning "trovy konania" 3 times is the same procedural section - it shouldnt get 3x the penalty
                    # This was causing chunk_0 (which mentions "nahradu trov" twice in the verdict summary) to get
                    # an unfairly low score and drop out of the top-5 for q_breach.
                    points = weight  # just apply the penalty once
                else:
                    # positive signals: more matches = more relevant, but cap at 3
                    # so a chunk with 20x "zmluvna pokuta" doesnt dominate everything
                    how_many = min(len(matches), 3)
                    points = weight * how_many

                score += points
                matched_patterns[pattern.pattern[:50]] = len(matches)   # just for debug

    # FIX: dampen noise penalties for chunks that have strong positive signal.
    # Problem:  chunk can contain both useful legal content (e.g. "zmluvná pokut vo výške 300 EUR" -> +3) AND noise patterns (e.g. "žalovaný namietal" -> -1).
    # Before this fix, both penalties applied at full strenght regardless of positive signal.
    # But  chunk with score 8 that mentions  party argument is not the same as chunk with score 1 that mentions  party argument.
    # Now: if chunk has positive score >= 5, i halve the noise penalty that was already applied
    # This keeps the noise penalty for weak chunks (where it  correctly pushes procedural text down) but reduces it for strong chunks (where the positive content outweighs the noise).
    #
    # IMPORTANT: i import NOISE_PATTERNS directly instead of iterating pattern_groups
    # because pattern_groups may contain NOISE_PATTERNS multiple times (its appende to every query's group list).
    # Iterating pattern_groups would undo the penalty 2-3x instead of once, which made the dampening way too agressive.
    if score >= 5:
        from src.reranking.legal_patterns import NOISE_PATTERNS
        for pattern, weight in NOISE_PATTERNS:
            if weight < 0 and pattern.findall(text):
                score -= weight * 0.5  # undo half of the negative penalty

    return score, matched_patterns


def get_chunk_number(chunk_id):     # i use it for first chunk bonus, etc..
    """
    Get the chunk number from  chunk ID like 'document.pdf_chunk_5'.
    Returns None if the ID doesnt match the expected format.
    """
    found = re.search(r'_chunk_(\d+)$', chunk_id)
    if found:
        return int(found.group(1))
    return None


# ########################################
# MAIN RERANKING FUNCTION
# #######################################

def rerank_chunks(query_key, chunk_texts, chunk_scores, chunk_ids,
                  top_k=5, alpha=0.6, debug=False):
    """
    Rerank retrieved chunks using keyword-based scoring

    This takes the output of do_retrieve() (which returned more chunks than needed)
    and re-scores each chunk by combining the original retrieval score with  keyword-based score.
    Then it picks the best top_k chunks.

    The formula is:
        final_score = alpha * retrieval_score_normalized + (1 - alpha) * keyword_score_normalized

    alpha controls the balance:
        alpha = 1.0  means only trust retrieval (reranker does nothing)
        alpha = 0.5  means equal weight for both
        alpha = 0.6  means retrieval is more important but keywords can still fix mistakes (default)
        alpha = 0.0  means only trust keywords (ignore retrieval completely)

    I picked alpha=0.6 because the retrieval is already pretty good (95% recall with hybrid),
    and i just want the keywords to fix the last few edge cases where retrieval puts
    procedural chunks above informative ones.

    Returns (reranked_texts, reranked_scores, reranked_ids) with only top_k items.
    """
    # if retrieval returned nothing
    if not chunk_texts:
        return [], [], []

    # size of reranking pool
    total_chunks = len(chunk_texts)

    # Step 1: score every chunk with keyword patterns
    keyword_scores = []
    all_debug_info = []

    for i in range(total_chunks):
        kw_score, debug_info = get_keyword_score(chunk_texts[i], query_key)

        # add bonus for chunk_0 (first chunk of the document)
        # CONDITIONAL: only apply if chunk_0 already matched at least one positive pattern
        # This prevents giving a free bonus to verdict-summary chunk_0s that contain no breach/contract info (happened in 4/10 golden dataset docs)
        chunk_number = get_chunk_number(chunk_ids[i]) if i < len(chunk_ids) else None
        if chunk_number == 0 and kw_score > 0:
            bonus = FIRST_CHUNK_BONUS.get(query_key, 0)
            kw_score += bonus
            if bonus > 0:
                debug_info["first_chunk_bonus"] = bonus

        # Position bonus for breach/contract queries: court decisions always start with factual description (contract type, parties, breach).
        # Chunks 1-5 are very likely to contain this info.
        # Bonus decays with position.
        # Only applies when chunk already has SOME keyword match (kw_score > 0) to avoid boosting completely irrelevant early chunks.
        if query_key in ("q_breach", "q_contract") and chunk_number is not None and kw_score > 0:
            if 1 <= chunk_number <= 3:
                pos_bonus = 2
            elif 4 <= chunk_number <= 6:
                pos_bonus = 1
            else:
                pos_bonus = 0
            if pos_bonus > 0:
                kw_score += pos_bonus
                debug_info["position_bonus"] = pos_bonus

        # saving keyword scores
        keyword_scores.append(kw_score)
        all_debug_info.append(debug_info)

    # Step 2: normalize both scores to [0, 1] so they are comparable
        # without normalization, retrieval scores are like 0.7-0.9 and keyword
    # scores are like 3-15, so they cant be combined fairly
    def min_max_normalize(scores):
        if not scores:
            return scores
        lowest = min(scores)
        highest = max(scores)
        if highest == lowest:
            # all scores are the same, just give everyone 0.5
            return [0.5] * len(scores)
        return [(s - lowest) / (highest - lowest) for s in scores]  # [10, 20, 30] - > [0.0, 0.5, 1.0]

    # retrieval score 0-1
    retrieval_normalized = min_max_normalize(chunk_scores)
    # keyword score 0-1
    keyword_normalized = min_max_normalize(keyword_scores)

    # Step 3: combine both scores using the alpha weight
    combined_scores = []
    for i in range(total_chunks):
        final = alpha * retrieval_normalized[i] + (1 - alpha) * keyword_normalized[i]       # RERANKING FORMULA
        combined_scores.append(round(final, 6))

    # Step 4: sort by combined score (highest first) and take top_k
    sorted_indices = sorted(range(total_chunks), key=lambda i: combined_scores[i], reverse=True)
    sorted_indices = sorted_indices[:top_k]

    # debug output - helps me see if the reranker is doing what i expect
    if debug:
        print(f"\n  [RERANKER] query={query_key}, alpha={alpha}, pool={total_chunks}, returning top {top_k}")
        for rank, i in enumerate(sorted_indices):
            chunk_num = get_chunk_number(chunk_ids[i]) if i < len(chunk_ids) else "?"
            print(f"    #{rank+1}: chunk_{chunk_num}  "
                  f"retrieval={chunk_scores[i]:.4f} (norm={retrieval_normalized[i]:.3f})  "
                  f"keyword={keyword_scores[i]:.1f} (norm={keyword_normalized[i]:.3f})  "
                  f"combined={combined_scores[i]:.4f}  "
                  f"matched={all_debug_info[i]}")

    # build the output lists
    out_texts = [chunk_texts[i] for i in sorted_indices]
    out_scores = [combined_scores[i] for i in sorted_indices]
    out_ids = [chunk_ids[i] for i in sorted_indices]

    return out_texts, out_scores, out_ids
