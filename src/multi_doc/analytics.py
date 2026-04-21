"""
Stage 4: Analytics — compute statistics across filtered penalties.

This is what makes  system DIFFERENT from naive RAG.
Regular RAG can only find similar text chunks.
It CANNOT compute things like:
  "Strongest predictor of moderation is penalty-to-principal ratio (lift 26x)"
  "In meritornych pripadov sud znizil pokutu v 37.5%"

All computation here is pure Python — no ML, no API calls. Just counters and math.
LLM in Stage 5 then explains these numbers in natural language.

What i show (after talking to lawyers i removed a lot):

1. Meritorny pomer — kolko % potvrdenych vs znizenych (len tieto 2 kategorie)
   Why only merit: dismissed/returned failed for formal reasons (invalid clause,
   procedural issues), not because of  rate. Including them would dilute the signal.

2. Factor LIFT — ktory zo 7 faktorov najsilnejsie predpoveda moderaciu
   Lift = P(negative | moderated) / P(negative | awarded)
   This is contrastive statistic — much more useful than raw frequency.

What i REMOVED (after lawyer feedback):
- amount_analysis (min/max/median sum, avg reduction %) — sound-bites that dont
  generalize, each case has diffrent context
- rate_analysis per type (percent_denne vs fixna_suma moderation rates) —
  at small sample sizes its anecdotal (66.7% from 3 cases = 2 cases)
- plna decision distribution (potvrdene/znizene/zamietnute/vratene %) —
  dismissed/returned percentages confuse lawyers, only merit ratio matters
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
# FACTOR ANALYSIS (with LIFT)
# #########################################
# this is the strongest analysis in the system.
# For each of 7 factors (dobre_mravy, pomer_k_istine, ...) i compute:
#   lift = P(factor negative | moderated) / P(factor negative | awarded)
#
# Example:
#   pomer_k_istine negative in 23/34 moderated (68%), 2/78 awarded (3%)
#   lift = 68% / 3% = 22.7x -> strong predictor of moderation
#
# If factor appears in 83% moderated AND 70% awarded, lift is only 1.2x -
# means it appears everywhere, not actually predictive of moderation

def _factor_analysis(records):
    """For each of 7 factors compute moderation lift (with raw counts)."""
    # split records into 2 groups - i only care about awarded vs moderated
    moderated = []
    awarded = []
    for r in records:
        if r["decision"] == "moderated_301":
            moderated.append(r)
        elif r["decision"] == "awarded_full":
            awarded.append(r)

    n_mod = len(moderated)
    n_awd = len(awarded)

    factor_stats = {}

    for label in FACTOR_LABELS:
        # count how many penalties in each group have this factor as "negative"
        mod_negative = 0
        for r in moderated:
            for f in r.get("factors", []):
                if f["label"] == label and f.get("sentiment") == "negative":
                    mod_negative += 1

        awd_negative = 0
        for r in awarded:
            for f in r.get("factors", []):
                if f["label"] == label and f.get("sentiment") == "negative":
                    awd_negative += 1

        # basic counts — raw numbers so lawyer can judge sample size
        stat = {
            "label": label,
            "moderated_negative": mod_negative,
            "moderated_total": n_mod,
            "awarded_negative": awd_negative,
            "awarded_total": n_awd,
        }

        # compute lift — need at least 3 cases in each group for it to be meaningful
        if n_mod >= 3 and n_awd >= 3 and awd_negative > 0:
            # normal lift: how many times more likely is negative in moderated vs awarded
            mod_rate = mod_negative / n_mod
            awd_rate = awd_negative / n_awd
            stat["lift"] = round(mod_rate / awd_rate, 2)
            stat["lift_note"] = None
        elif n_mod >= 3 and awd_negative == 0 and mod_negative > 0:
            # factor appears in moderated but NEVER in awarded — infinite lift
            # this is actually the strongest signal
            stat["lift"] = None
            stat["lift_note"] = "only_in_moderated"
        else:
            # not enough data to compute lift
            stat["lift"] = None
            stat["lift_note"] = "insufficient_data"

        factor_stats[label] = stat

    # sort factors by lift (strongest predictors first):
    # 1. only_in_moderated (infinite lift) first, sorted by how often they appear
    # 2. numeric lift after, sorted descending
    # 3. insufficient_data last (wont be displayed anyway)
    def sort_key(x):
        if x.get("lift_note") == "only_in_moderated":
            # sort inf-lift factors by mod_negative count (more frequent first)
            return (2, x["moderated_negative"])
        elif x.get("lift") is not None:
            return (1, x["lift"])
        else:
            return (0, 0)

    sorted_factors = sorted(factor_stats.values(), key=sort_key, reverse=True)

    # only keep factors that have displayable lift (skip "insufficient_data")
    displayable = []
    for f in sorted_factors:
        has_lift = f.get("lift") is not None
        is_inf = f.get("lift_note") == "only_in_moderated"
        if has_lift or is_inf:
            displayable.append(f["label"])

    return {
        "factors": {f["label"]: f for f in sorted_factors},
        "sorted_by_lift": displayable,
        "n_moderated": n_mod,
        "n_awarded": n_awd,
    }


# #########################################
# MAIN ANALYTICS FUNCTION
# #########################################

def compute_analytics(filtered_records, intent):
    """Compute analytics over filtered penalty records.

    Simple 2-state logic based on sample size:
      n < MIN_SAMPLE_SIZE (5):  nothing, just note that its too few
      n >= MIN_SAMPLE_SIZE:     merit_ratio + factor_analysis

    I dont have "low/medium/higher confidence" labels anymore - lawyer sees
    the actual n count and can judge for themselves how much to trust it.
    """
    n = len(filtered_records)

    analytics = {
        "n": n,
        # caveat is always present — dataset is not representative of all Slovak courts
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
            f"Statistiku pri tomto pocte nepocitame — zobrazujeme jednotlive rozhodnutia."
        )
        return analytics

    # merit ratio (might be None if <3 meritorne cases)
    analytics["merit_ratio"] = _merit_ratio(filtered_records)

    # factor lift — the strongest analysis in the system
    analytics["factor_analysis"] = _factor_analysis(filtered_records)

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
# the contrast with awarded cases stays in LLM prompt (format_for_llm), not here.

def format_for_lawyer(analytics):
    """Plain-language summary for lawyer display.

    Just counts and percentages. No statistical jargon. lawyer doesnt want to
    read "lift 26.4x" — they want "pomer k istine bol problem v 8 z 9 pripadov".
    """
    n = analytics["n"]
    lines = []

    # simple header
    lines.append(f"ZHRNUTIE ({n} relevantnych pripadov)")
    lines.append("=" * 60)

    # too few cases — just show the note and stop
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

    # factor frequencies — only counts from moderated cases, sorted by count
    # i dont show awarded contrast here — thats for LLM grounding only
    fa = analytics.get("factor_analysis", {})
    if fa and fa.get("n_moderated", 0) > 0:
        n_mod = fa["n_moderated"]
        lines.append(f"\nNajcastejsie dovody znizeni (v {n_mod} znizenych pripadoch):")

        # collect (friendly_label, count) pairs, sort by count descending
        factor_counts = []
        for label, f_data in fa["factors"].items():
            mn = f_data["moderated_negative"]
            if mn > 0:  # skip factors never negative in moderated
                friendly = FACTOR_LABELS_FRIENDLY.get(label, label)
                factor_counts.append((friendly, mn))
        factor_counts.sort(key=lambda x: x[1], reverse=True)

        for friendly, count in factor_counts:
            lines.append(f"  - {friendly:.<35s} {count} z {n_mod} pripadov")

    lines.append(f"\nPoznamka: {analytics['dataset_caveat']}")
    return "\n".join(lines)


# #########################################
# FORMAT FOR LLM (prompt grounding)
# #########################################
# full lift text with raw counts and contrast between moderated vs awarded.
# LLM uses this to write accurate sentences like "pomer k istine bol problem
# v 8 z 9 znizenych pripadov, zatial co v potvrdenych len v 1 z 15."
# this format is NOT shown to the lawyer — only goes into Stage 5 prompt.

def format_for_llm(analytics):
    """Full statistical detail for LLM grounding.

    LLM needs the contrast (moderated vs awarded counts) so it can write
    proper comparative sentences. Also includes lift values — LLM can decide
    itself whether to mention them in the answer or translate to plain words.
    """
    n = analytics["n"]
    lines = []

    # simple header with just n (no more confidence labels)
    lines.append(f"ANALYZA PRECEDENSOV ({n} relevantnych pripadov)")
    lines.append("=" * 60)

    # too few cases
    if n < MIN_SAMPLE_SIZE:
        lines.append(analytics.get("note", ""))
        lines.append(f"\nPoznamka: {analytics['dataset_caveat']}")
        return "\n".join(lines)

    # merit ratio
    merit = analytics.get("merit_ratio")
    if merit is not None:
        lines.append(f"\nMeritorny pomer (len potvrdene + znizene):")
        lines.append(f"  Potvrdene: {merit['awarded_full']}, Znizene: {merit['moderated_301']}")
        lines.append(f"  Miera moderacie: {merit['moderation_rate_pct']}%")

    # factor analysis with full lift + counts
    fa = analytics.get("factor_analysis", {})
    if fa and fa.get("sorted_by_lift"):
        lines.append(
            f"\nFaktory moderacie (lift = kolkokrat castejsie negativny "
            f"v znizenych vs potvrdench):"
        )
        lines.append(f"  Znizene pripady: {fa['n_moderated']}, Potvrdene: {fa['n_awarded']}")

        for label in fa["sorted_by_lift"]:
            f_data = fa["factors"][label]

            # format lift — "inf" for only_in_moderated, number otherwise
            if f_data.get("lift_note") == "only_in_moderated":
                lift_str = "  inf"
            elif f_data.get("lift") is not None:
                lift_str = f"{f_data['lift']:.1f}x"
            else:
                continue  # skip factors with insufficient data

            # raw counts — LLM uses these to write precise sentences
            mn = f_data["moderated_negative"]
            mt = f_data["moderated_total"]
            an = f_data["awarded_negative"]
            at = f_data["awarded_total"]
            lines.append(
                f"  {label:.<30s} lift={lift_str:>5s}  "
                f"(zniz={mn}/{mt}, potv={an}/{at})"
            )

    lines.append(f"\nPoznamka: {analytics['dataset_caveat']}")
    return "\n".join(lines)
