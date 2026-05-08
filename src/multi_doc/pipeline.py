"""
Pipeline orchestrator - ties all 4 stages together into single search() call.

This is what you call to use the system:

    from src.multi_doc.pipeline import PrecedentSearchPipeline

    pipeline = PrecedentSearchPipeline()
    result = pipeline.search("Robim zmluvu o dielo za 50k EUR. Aku pokutu za omeskanie?")
    print(result["statistics_text"])
    for card in result["precedent_cards"]:
        print(card)

It loads penalty index once at startup, then for each query runs:
  Stage 1: parse query (LLM Gemini Flash)
  Stage 2: filter penalties (Python)
  Stage 3: rank filtered penalties (numpy + OpenAI embedding)
  Stage 4: compute analytics (Python)

I do NOT generate LLM-synthesized answer at the end.
After lawyer feedback i decided that LLM-synthesis adds risk (hallucination,
smooth-talking generalizations from small samples) without real new information.
Statistics + precedent cards are deterministic and lawyer reads concrete cases
themselves. The system is a precedent SEARCH engine, not a chatbot.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.multi_doc.penalty_index import PenaltyIndex
from src.multi_doc.query_analyzer import analyze_query
from src.multi_doc.ranker import rank_penalties
from src.multi_doc.analytics import compute_analytics, format_for_lawyer


# #########################################
# BUILD PRECEDENT CARD (display-friendly dict for one ranked penalty)
# #########################################
# i used to build these in synthesizer.py but synthesizer is gone now.
# Each card is a flat dict with everything lawyer needs to see for one precedent.
# IMPORTANT: one card = one PENALTY, not one court decision. If a decision has
# multiple penalties (15 docs in our dataset do), each penalty gets its own card.

def _build_precedent_card(rec, score, rank):
    """Build display dict for one precedent card (one penalty, not one decision).

    rec is penalty record from index, score is final ranking score, rank is 1-based position.
    """
    return {
        "rank": rank,
        "penalty_id": rec.get("penalty_id", "pokuta_1"),
        "case_number": rec.get("case_number", ""),
        "court_name": rec.get("court_name", ""),
        "decision_date": rec.get("decision_date", ""),
        "contract_type": rec.get("contract_type", ""),
        "relationship_type": rec.get("relationship_type", ""),
        "breach_type": rec.get("breach_type", ""),
        "rate_value_raw": rec.get("rate_value_raw", ""),
        "currency": rec.get("currency", "EUR"),
        "secured_principal": rec.get("secured_principal"),
        "original_claimed": rec.get("original_claimed"),
        "final_awarded": rec.get("final_awarded"),
        "decision": rec.get("decision", ""),
        "legal_reasoning_summary": rec.get("legal_reasoning_summary", ""),
        "key_quotes": rec.get("key_quotes", []),
        "score": round(score, 3),
        "authority_level": rec.get("authority_level", 1),
    }


class PrecedentSearchPipeline:
    """Main pipeline for multi-document precedent search.
    Loads penalty index once at startup, then runs search() for each query.
    """

    def __init__(self, index_dir="data/10_penalty_index"):
        """Load penalty index at startup."""
        print("Initializing PrecedentSearchPipeline...")
        self.index = PenaltyIndex(index_dir)
        print(f"Ready. {len(self.index.records)} penalties loaded.\n")

    def search(self, query_text, top_n=5):
        """Run full search pipeline for a lawyer's query.

        Returns dict with: query, intent, filter_metadata,
        statistics_text, precedent_cards, timing.
        """
        print(f"{'=' * 60}")
        print(f"QUERY: {query_text}")
        print(f"{'=' * 60}")

        timing = {}
        total_start = time.time()

        # #########################################
        # STAGE 1: Query Understanding (LLM - Gemini Flash)
        # #########################################
        print("\n[Stage 1] Analyzing query...")
        t0 = time.time()

        try:
            intent = analyze_query(query_text)
        except Exception as e:
            # if LLM fails completely, continue without filters
            print(f"  WARNING: Query analysis failed ({e}), using no filters")
            intent = {
                "contract_type": None,
                "breach_type": None,
                "decision_interest": None,
                "factor_interest": [],
                "amount_hint": None,
                "intent": "general_precedent",
                "semantic_query": query_text,
            }

        timing["stage1_query_understanding"] = round(time.time() - t0, 2)

        # show what LLM understood - print the intent
        ct = intent.get("contract_type")
        bt = intent.get("breach_type")
        amt = intent.get("amount_hint")
        intent_type = intent.get("intent")
        print(f"  contract_type={ct}, breach_type={bt}, amount={amt}, intent={intent_type}")

        # #########################################
        # STAGE 2: Penalty-Level Filtering
        # #########################################
        print("\n[Stage 2] Filtering penalties...")
        t0 = time.time()

        # uses penalty index to filter from all decisions
        filtered_records, filtered_embeddings, filter_meta = self.index.filter(intent)

        timing["stage2_filtering"] = round(time.time() - t0, 4)
        print(f"  {filter_meta['filtered_count']}/{filter_meta['total_penalties']} penalties match")
        for f in filter_meta.get("active_filters", []):
            print(f"    filter: {f}")

        # if nothing matched filters, use all penalties (graceful degradation)
        if not filtered_records:
            print("  WARNING: No penalties matched filters, using all penalties")
            filtered_records, filtered_embeddings, filter_meta = self.index.get_all()
            filter_meta["filter_type"] = "none_fallback"

        ##########################################
        # STAGE 3: Ranking
        # #########################################
        print("\n[Stage 3] Ranking penalties...")
        t0 = time.time()

        ranked = rank_penalties(filtered_records, filtered_embeddings, intent, top_n=top_n)

        timing["stage3_ranking"] = round(time.time() - t0, 2)

        # #########################################
        # STAGE 4: Analytics (uses ALL filtered penalties, not just ranked top)
        # #########################################
        print("\n[Stage 4] Computing analytics...")
        t0 = time.time()

        analytics = compute_analytics(filtered_records)
        statistics_text = format_for_lawyer(analytics)

        timing["stage4_analytics"] = round(time.time() - t0, 4)
        print(f"  n={analytics['n']}")

        # #########################################
        # BUILD PRECEDENT CARDS (top N ranked)
        # #########################################
        # i build display cards directly from ranked tuples (rec, score)
        # no LLM synthesis - lawyer reads precedents themselves
        precedent_cards = []
        for i, (rec, score) in enumerate(ranked):
            card = _build_precedent_card(rec, score, rank=i + 1)
            precedent_cards.append(card)

        # #########################################
        # ASSEMBLE RESULT
        # #########################################
        timing["total"] = round(time.time() - total_start, 2)

        # build intent dict without internal metadata
        clean_intent = {k: v for k, v in intent.items() if k != "_metadata"}

        result = {
            "query": query_text,
            "intent": clean_intent,
            "filter_metadata": filter_meta,
            "statistics_text": statistics_text,
            "precedent_cards": precedent_cards,
            "timing": timing,
        }

        # print timing summary
        print(f"\n[Done] Total time: {timing['total']}s")
        print(f"  Stage 1 (query): {timing['stage1_query_understanding']}s")
        print(f"  Stage 2 (filter): {timing['stage2_filtering']}s")
        print(f"  Stage 3 (rank): {timing['stage3_ranking']}s")
        print(f"  Stage 4 (analytics): {timing['stage4_analytics']}s")

        return result
