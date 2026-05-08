"""
Stage 4: Analytics - compute statistics across filtered penalties.

This is what makes system DIFFERENT from naive RAG.
Regular RAG can only find similar text chunks. It CANNOT compute things like:
  "V meritornych pripadov sud znizil pokutu v 37.5%"
  "Najcastejsi dovod znizenia bol pomer pokuty k istine (8 z 9 pripadov)"

All computation here is pure Python - no ML, no API calls. Just counters.
The output text is shown directly to the lawyer (no LLM synthesis on top).

What i show (after talking to lawyers i kept only what's defensible):

1. Meritorny pomer - kolko % potvrdenych vs znizenych (len tieto 2 kategorie)
   Why only merit: dismissed/returned failed for formal reasons (invalid clause,
   procedural issues), not because of rate. Including them would dilute the signal.

2. Factor frequencies in moderated cases - which of 7 factors most often appeared
   as "negative" when court actually reduced. Plain frequencies, no jargon.

What i REMOVED (after lawyer feedback):
- amount_analysis (min/max/median sum, avg reduction %) - sound-bites that dont
  generalize, each case has diffrent context
- rate_analysis per type (percent_denne vs fixna_suma moderation rates) -
  at small sample sizes its anecdotal (66.7% from 3 cases = 2 cases)
- plna decision distribution (potvrdene/znizene/zamietnute/vratene %) -
  dismissed/returned percentages confuse lawyers, only merit ratio matters
- factor LIFT computation (P(neg|mod) / P(neg|awd)) - too statistically heavy,
  lawyer just wants to know which factors come up most often when court reduces
- LLM synthesis at the end - risk of hallucination/smooth-talking generalizations
  without new information. Lawyer reads concrete precedent cards themselves.
"""

import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.multi_doc.schemas import FACTOR_LABELS


# #########################################
# SAMPLE SIZE THRESHOLD
# #########################################
# at n<5 any statistic is basically anecdotal - we dont compute stats then.
# i used to have 4 levels (insufficient/low/medium/higher) but decided its
# overcomplicated - lawyer just needs to know "is there enough data or not"
# the actual n is shown so lawyer can judge himself.

MIN_SAMPLE_SIZE = 5


# #########################################
# MERIT RATIO (only awarded_full + moderated_301)
# #########################################
# dismissed and returned cases failed for formal reasons (invalid clause,
# procedural issues) - not because of the penalty rate. Including them would
# dilute the signal. I only count cases where court ACTUALLY evaluated the rate.

def _merit_ratio(records):
    """Compute moderation rate among meritorne decisions (awarded + moderated only).

    Returns dict with 3 numbers: how many awarded, how many moderated, and %.
    Or None if there are less than 3 meritorne cases (too few to say anything).
    """
    # count only awarded_full and moderated_301
    counts = Counter()
    for r in records:
        counts[r["decision"]] += 1

    n_awarded = counts.get("awarded_full", 0)
    n_moderated = counts.get("moderated_301", 0)
    n_merit = n_awarded + n_moderated

    # need at least 3 meritorne cases for this to say anything
    if n_merit < 3:
        return None

    # % of meritornych cases that were moderated
    moderation_rate_pct = round(100 * n_moderated / n_merit, 1)

    return {
        "awarded_full": n_awarded,
        "moderated_301": n_moderated,
        "total": n_merit,
        "moderation_rate_pct": moderation_rate_pct,
    }


# #########################################
# FACTOR FREQUENCIES (in moderated cases)
# #########################################
# for each of 7 factors (dobre_mravy, pomer_k_istine, ...) i count how many times
# it appeared as "negative" in moderated cases. Lawyer sees these as plain
# frequencies sorted from most common to least common.

def _factor_frequencies(records):
    """Count negative-sentiment occurrences per factor in moderated cases only."""
    # take only moderated cases - those are the ones where court reduced
    moderated = [r for r in records if r["decision"] == "moderated_301"]

    counts = {}
    for label in FACTOR_LABELS:
        c = 0
        for r in moderated:
            for f in r.get("factors", []):
                if f["label"] == label and f.get("sentiment") == "negative":
                    c += 1
        counts[label] = c

    return {
        "n_moderated": len(moderated),
        "factor_counts": counts,  # {factor_label: count_of_negative_in_moderated}
    }


# #########################################
# MAIN ANALYTICS FUNCTION
# #########################################

def compute_analytics(filtered_records):
    """Compute analytics over filtered penalty records.

    Simple 2-state logic based on sample size:
      n < MIN_SAMPLE_SIZE (5):  nothing, just note that its too few
      n >= MIN_SAMPLE_SIZE:     merit_ratio + factor_frequencies

    Lawyer sees actual n and can judge how much to trust it.
    """
    n = len(filtered_records)

    analytics = {
        "n": n,
        # caveat is always present - dataset is not representative of all Slovak courts
        "dataset_caveat": (
            "Statistiky su zalozene na korpuse sudnych rozhodnuti "
            "vybranych podla klucovych slov 'zmluvna pokuta' a §301 "
            "Obchodneho zakonnika. Nezastupuju vsetky slovenske sudy."
        ),
    }

    # not enough data for any meaningful statistics
    if n < MIN_SAMPLE_SIZE:
        analytics["note"] = (
            f"Pre zadane kriteria sme nasli len {n} pripadov. "
            f"Statistiku pri tomto pocte nepocitame - zobrazujeme jednotlive rozhodnutia."
        )
        return analytics

    # merit ratio (might be None if <3 meritorne cases)
    analytics["merit_ratio"] = _merit_ratio(filtered_records)

    # how often each factor appeared as negative in moderated cases
    analytics["factor_frequencies"] = _factor_frequencies(filtered_records)

    return analytics


# #########################################
# FRIENDLY LABELS for lawyer display
# #########################################
# factor labels use snake_case internally but lawyer sees plain slovak

FACTOR_LABELS_FRIENDLY = {
    "dobre_mravy": "dobre mravy",
    "zabezpecovacia_funkcia": "zabezpecovacia funkcia pokuty",
    "vyska_skody": "vyska skutocnej skody",
    "pomer_k_istine": "pomer pokuty k istine",
    "spravanie_dlznika": "spravanie dlznika",
    "kumulacia_s_urokom": "kumulacia s urokom z omeskania",
    "spravanie_veritela": "spravanie veritela",
}


# #########################################
# FORMAT FOR LAWYER (user display)
# #########################################
# plain language, no "lift" jargon, no "inf" values.
# lawyer sees just: N cases, % moderated, top reasons as simple frequencies.

def format_for_lawyer(analytics):
    """Plain-language summary for lawyer display.

    Just counts and percentages. No statistical jargon. Lawyer doesnt want to
    read "lift 26.4x" - they want "pomer k istine bol problem v 8 z 9 pripadov".
    """
    n = analytics["n"]
    lines = []

    # simple header
    lines.append(f"ZHRNUTIE ({n} relevantnych pripadov)")
    lines.append("=" * 60)

    # too few cases - just show the note and stop
    if n < MIN_SAMPLE_SIZE:
        lines.append(analytics.get("note", ""))
        lines.append(f"\nPoznamka: {analytics['dataset_caveat']}")
        return "\n".join(lines)

    # merit ratio in natural language
    merit = analytics.get("merit_ratio")
    if merit is not None:
        pct = merit["moderation_rate_pct"]
        mod = merit["moderated_301"]
        total = merit["total"]
        lines.append(
            f"\nV {pct}% meritornych pripadov sud znizil pokutu "
            f"({mod} z {total})."
        )

    # factor frequencies - how many times each factor was "negative" in moderated cases
    ff = analytics.get("factor_frequencies", {})
    if ff and ff.get("n_moderated", 0) > 0:
        n_mod = ff["n_moderated"]
        lines.append(f"\nNajcastejsie dovody znizeni (v {n_mod} znizenych pripadoch):")

        # sort factors by count descending (most common reason first)
        sorted_factors = sorted(ff["factor_counts"].items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_factors:
            if count > 0:  # skip factors never negative in moderated
                friendly = FACTOR_LABELS_FRIENDLY.get(label, label)
                lines.append(f"  - {friendly:.<35s} {count} z {n_mod} pripadov")

    lines.append(f"\nPoznamka: {analytics['dataset_caveat']}")
    return "\n".join(lines)


