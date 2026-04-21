"""
Demo script for thesis defense — runs queries and prints formatted results.

Shows full pipeline in action: query -> parse -> filter -> rank -> analytics -> synthesis.

Run from project root:
    python src/multi_doc/demo.py

I run 5 demo queries covering main use cases:
1. Contract drafter looking for safe penalty rate
2. Litigation lawyer looking for defense arguments
3. General question about moderation factors
4. Specific contract type (loan)
5. Factor-focused search (cumulation with interest)
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.multi_doc.pipeline import PrecedentSearchPipeline


# ==========================================
# DEMO QUERIES
# ==========================================
# i picked these to cover 3 intents and different contract types
# written in casual Slovak like a real lawyer would type

DEMO_QUERIES = [
    # 1. safe_rate — lawyer writing a contract
    "Idem pisat zmluvu o dielo, hodnota je cca 50 tisic eur. Aku vysoku pokutu mozem dat za omeskanie aby to sud potom neznizil?",

    # 2. defense_args — litigation lawyer
    "Zastupujem klienta na sude, dostal pokutu 0,5 percenta denne z najomnej zmluvy, je to dost vysoke. Cim mozem argumentovat aby sud pokutu znizil?",

    # 3. general — broad question about factors
    "Zaujima ma, co vlastne sud berie do uvahy ked rozhoduje ci pokutu znici alebo necha? Ake su tie faktory?",

    # 4. general — specific contract type
    "Mame zmluvu o uvere a dlznik neplati, chceme uplatnit pokutu. Je tam nejaka sudna prax k pokutam pri uveroch?",

    # 5. general — specific factor (cumulation)
    "Ak mame v zmluve aj urok z omeskania aj pokutu, je to problem? Ako to sudy posudzuju?",
]


# ==========================================
# OUTPUT FORMATTING
# ==========================================

def print_result(result, query_num):
    """Pretty-print one search result (no emojis, clean format)."""
    print("\n" + "=" * 70)
    print(f"  DEMO QUERY {query_num}")
    print("=" * 70)

    # query + what the system understood
    intent = result["intent"]
    print(f"\nQuery: {result['query']}")
    print(f"  Intent: {intent.get('intent', '?')}")

    # show filter values (replace None with - for readability)
    ct = intent.get("contract_type") or "-"
    bt = intent.get("breach_type") or "-"
    amt = intent.get("amount_hint")
    if amt:
        amt_str = f"{amt:,.0f} EUR".replace(",", " ")
    else:
        amt_str = "-"
    print(f"  Filters: contract_type={ct}, breach_type={bt}, amount={amt_str}")

    # filter results
    fm = result["filter_metadata"]
    print(f"\nFilter: {fm['filtered_count']}/{fm['total_penalties']} penalties matched")
    for f in fm.get("active_filters", []):
        print(f"  {f}")

    # statistics
    print(f"\nSTATISTIKY")
    print("-" * 50)
    print(result["statistics_text"])

    # LLM answer
    print(f"\nODPOVED")
    print("-" * 50)
    print(result["llm_answer"])

    # precedent cards
    card_count = len(result["precedent_cards"])
    print(f"\nPRECEDENSY (top {card_count})")
    print("-" * 50)

    # collect case numbers to detect duplicates (same case, multiple penalties)
    all_case_nums = []
    for card in result["precedent_cards"]:
        all_case_nums.append(card["case_number"])

    for card in result["precedent_cards"]:
        # authority marker — just text label for supreme court
        auth_level = card.get("authority_level", 1)
        if auth_level == 3:
            auth_label = "[NS SR] "
        else:
            auth_label = ""

        # decision label
        dec_labels = {
            "awarded_full": "POTVRDENA",
            "moderated_301": "ZNIZENA (s301)",
            "dismissed": "ZAMIETNUTA",
            "returned": "VRATENA",
        }
        dec = card.get("decision", "")
        dec_label = dec_labels.get(dec, dec)

        # if same case number appears multiple times, show penalty_id to distinguish
        pid_label = ""
        if all_case_nums.count(card["case_number"]) > 1:
            pid_label = f" [{card.get('penalty_id', '')}]"

        # rate — show "nespecifikovana" instead of empty string
        rate = card.get("rate_value_raw") or "nespecifikovana"

        # print the card
        print(f"  [{card['rank']}] {auth_label}{card['court_name']} {card['case_number']}"
              f"{pid_label} ({card['decision_date']})")
        print(f"      {card['contract_type']} | {card['breach_type']} | {rate}")
        print(f"      {dec_label} (score: {card['score']})")

    # timing
    t = result["timing"]
    print(f"\nCas: celkovo {t['total']}s "
          f"(query: {t['stage1_query_understanding']}s, "
          f"filter: {t['stage2_filtering']}s, "
          f"rank: {t['stage3_ranking']}s, "
          f"analytics: {t['stage4_analytics']}s, "
          f"synthesis: {t['stage5_synthesis']}s)")


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 70)
    print("MULTI-DOCUMENT PRECEDENT SEARCH — DEMO")
    print("=" * 70)

    # load penalty index once
    pipeline = PrecedentSearchPipeline()

    # run each demo query
    results = []
    for i in range(len(DEMO_QUERIES)):
        query = DEMO_QUERIES[i]
        query_num = i + 1

        print(f"\n\n{'#' * 70}")
        print(f"# RUNNING DEMO QUERY {query_num}/{len(DEMO_QUERIES)}")
        print(f"{'#' * 70}")

        result = pipeline.search(query)
        results.append(result)

        # print formatted output
        print_result(result, query_num)

    # summary table
    print("\n\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    for i in range(len(results)):
        result = results[i]
        query_num = i + 1
        t = result["timing"]
        n = result["filter_metadata"]["filtered_count"]
        intent_type = result["intent"].get("intent", "?")
        print(f"  Query {query_num}: {intent_type:.<20s} {n:>3d} matches, {t['total']:.1f}s total")

    print("\nDone!")


if __name__ == "__main__":
    main()
