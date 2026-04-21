"""
Pipeline orchestrator — ties all 5 stages together into single search() call.

This is what you call to use the system:

    from src.multi_doc.pipeline import PrecedentSearchPipeline

    pipeline = PrecedentSearchPipeline()
    result = pipeline.search("Robim zmluvu o dielo za 50k EUR. Aku pokutu za omeskanie?")
    print(result["statistics_text"])
    print(result["llm_answer"])

It loads penalty index once at startup, then for each query runs:
  Stage 1: parse query (LLM) -> Stage 2: filter penalties -> Stage 3: rank
  -> Stage 4: compute analytics -> Stage 5: synthesize answer (LLM)
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
from src.multi_doc.analytics import compute_analytics
from src.multi_doc.synthesizer import synthesize


class PrecedentSearchPipeline:
    """Main pipeline for multi-document precedent search.
    Loads penalty index once at startup, then runs search() for each query.
    """

    def __init__(self, index_dir="data/10_penalty_index"):
        """Load penalty index at startup."""
        print("Initializing PrecedentSearchPipeline...")
        self.index = PenaltyIndex(index_dir)
        print(f"Ready. {len(self.index.records)} penalties loaded.\n")

    def search(self, query_text, top_n_ranking=15, top_n_precedents=5):
        """Run full search pipeline for a lawyer's query.

        Returns dict with: query, intent, filter_metadata, analytics,
        statistics_text, llm_answer, precedent_cards, timing.
        """
        print(f"{'=' * 60}")
        print(f"QUERY: {query_text}")
        print(f"{'=' * 60}")

        timing = {}
        total_start = time.time()

        # #########################################
        # STAGE 1: Query Understanding (LLM — Gemini Flash)
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

        ranked = rank_penalties(
            filtered_records, filtered_embeddings,
            intent, top_n=top_n_ranking,
        )

        timing["stage3_ranking"] = round(time.time() - t0, 2)

        # #########################################
        # STAGE 4: Analytics (uses ALL filtered penalties, not just ranked top)
        # #########################################
        print("\n[Stage 4] Computing analytics...")
        t0 = time.time()

        analytics = compute_analytics(filtered_records, intent)

        timing["stage4_analytics"] = round(time.time() - t0, 4)
        print(f"  n={analytics['n']}")

        # #########################################
        # STAGE 5: LLM Synthesis
        # #########################################
        print("\n[Stage 5] Synthesizing answer...")
        t0 = time.time()

        # take top N penalties for precedent display
        # for safe_rate intent: make sure we include at least 1 moderated case
        # in the top results, so the LLM can discuss what crosses the line
        # (without this, ranking naturally promotes awarded cases only)
        top_for_synthesis = ranked[:top_n_precedents]

        if intent_type == "safe_rate":
            # check if there is any moderated case in top results already
            has_moderated = False
            for entry in top_for_synthesis:
                rec = entry[0]
                if rec.get("decision") == "moderated_301":
                    has_moderated = True
                    break

            # if no moderated case, find the best one from ranked and add it
            if not has_moderated:
                for entry in ranked:
                    rec = entry[0]
                    if rec.get("decision") == "moderated_301":
                        # replace last item in top list with this moderated case
                        top_for_synthesis = list(top_for_synthesis)
                        top_for_synthesis[-1] = entry
                        break

        synthesis = synthesize(
            query_text, intent, analytics, top_for_synthesis,
        )

        timing["stage5_synthesis"] = round(time.time() - t0, 2)

        # #########################################
        # ASSEMBLE RESULT
        # #########################################
        timing["total"] = round(time.time() - total_start, 2)

        # build intent dict without internal metadata
        clean_intent = {}
        for key, value in intent.items():
            if key != "_metadata":
                clean_intent[key] = value

        result = {
            "query": query_text,
            "intent": clean_intent,
            "filter_metadata": filter_meta,
            "analytics": analytics,
            "statistics_text": synthesis["statistics_text"],
            "llm_answer": synthesis["llm_answer"],
            "precedent_cards": synthesis["precedent_cards"],
            "timing": timing,
        }

        # print timing summary
        print(f"\n[Done] Total time: {timing['total']}s")
        print(f"  Stage 1 (query): {timing['stage1_query_understanding']}s")
        print(f"  Stage 2 (filter): {timing['stage2_filtering']}s")
        print(f"  Stage 3 (rank): {timing['stage3_ranking']}s")
        print(f"  Stage 4 (analytics): {timing['stage4_analytics']}s")
        print(f"  Stage 5 (synthesis): {timing['stage5_synthesis']}s")

        return result
