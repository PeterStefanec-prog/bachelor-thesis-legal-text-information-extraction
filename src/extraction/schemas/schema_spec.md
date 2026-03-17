# Extraction Schema v4.3 – Changes and Overview

> **File:** `src/extraction/schemas/extraction_schema_v4.3.json`

---

## What the schema extracts

The goal is to extract structured info about contractual penalties (zmluva pokuta) from Slovak court decisions, specifically to help lawyers calibrate penalty clauses in new contracts. The key question is always: was the penalty moderated, why, and what rate/amount was involved.

---

## Structure overview

```
meta                          - court name, case number, date, ecli
case_context
  dispute_summary             - 2-3 sentence summary of what the dispute was about
  verdict_summary             - 1-2 sentences on overall outcome
  contract_type               - dielo / uver / najom / ...
  relationship_type           - B2B / B2C / C2C

contractual_penalties[]       - list because one decision can have multiple penalties
  penalty_internal_id         - pokuta_1, pokuta_2 ...
  related_claim_ref           - which invoice/claim this penalty relates to (if multiple)
  breach_type                 - late_payment / non_monetary_performance / ...  + evidence
  rate_definition             - type + raw value (napr. "0,05% denne")          + evidence
  amounts
    currency
    secured_principal         - base amount the penalty is calculated from       + evidence (strict null if not explicit)
    original_claimed          - what plaintiff asked for                         + evidence
    final_awarded             - what court actually gave                         + evidence
  associated_interest         - just enough to not confuse interest with penalty rate
  moderation_analysis
    decision_on_penalty       - awarded_full / awarded_reduced / dismissed       + evidence
    moderation_applied        - yes / no / not_applicable                        + evidence
    legal_reasoning_summary   - how numbers were calculated + why court moderated + key_quotes
    factors[]                 - 5 specific legal arguments + sentiment            + evidence each

quality_control
  flags, missing_fields
```

---

## Changes from v4.1 to v4.3

### removed fields

**`verdict_summary` evidence**
verdict_summary is a model-generated synthesis from multiple paragraphs. there is no single sentence that "proves" it. added evidence would just be a random cherry-picked quote, which is misleading. kept the field itself (as a plain string) because it is still useful - sometimes the court never even reaches the penalty question (zamietol pre neplatnost zmluvy) and without verdict_summary you wouldnt know that from decision_on_penalty alone.

**`dispute_summary` evidence**
same reason - it is a synthesis not an extraction. plain string, no evidence.

**`parties.plaintiff` and `parties.defendant` names**
Slovak decisions are anonymized (O. R., XX. B. XXXX). model would hallucinate real names. names are useless for penalty calibration anyway. kept `relationship_type` (B2B/B2C) because that one actually affects how strictly courts apply §301.

**`breach_duration` (trvanie porusenia)**
almost never stated explicitly in decisions. model would derive it from the calculation context and write it as a fact - that is a hallucination. when duration is relevant it shows up naturally in `legal_reasoning_summary`.

**`calculation_logic_summary`**
merged into `legal_reasoning_summary`. both fields were pulling quotes from the same paragraphs - court describes calculation and immediately explains why it accepted or rejected it. two separate evidence lists for overlapping content confused the model and wasted tokens. now `legal_reasoning_summary` covers both.

**`associated_interest.applies_to`**
hard to determine reliably. field exists only to prevent confusing 9% p.a. interest with 0.05% daily penalty rate. for that you only need `awarded` + `rate_value`.

---

### added / changed

**`breach_type` now has evidence**
in v4.1 it had no evidence. this was a mistake - breach_type determines comparability of cases (late_payment vs non_monetary_performance behave very differently in courts). should be verifiable.

**`factors` got `not_mentioned` as 4th sentiment option**
before: positive / negative / neutral
now: positive / negative / neutral / not_mentioned

this matters for thesis statistics. if the court never mentioned a factor, it should not be counted as `neutral`. if it were, the chapter 9 analysis would show artificially high neutral counts for factors the court simply did not discuss.

**`legal_reasoning_summary` uses `key_quotes` instead of `evidence`**
it is a synthesis from multiple paragraphs, so a single evidence list would be misleading. replaced with `key_quotes` (1-2 anchor quotes) which signals to both the model and the reader that this is not a single extracted fact but a summarized interpretation.

**all examples in Slovak**
previous version had English examples (e.g. "Plaintiff calculated..."). the model reads Slovak legal texts and should produce Slovak output. English examples would cause the model to mimic them and mix languages in output.

---

## Where evidence is and where it is not

| field | evidence? | reason |
|-------|-----------|--------|
| dispute_summary | no | synthesis from multiple chunks |
| verdict_summary | no | synthesis from multiple chunks |
| contract_type | no | inferred from overall context, no single sentence proves it |
| relationship_type | no | same as above |
| breach_type | yes | verifiable claim, affects case comparability |
| rate_definition | yes | specific number, must be checkable |
| secured_principal | yes (nullable) | specific number, strict null if not explicit |
| original_claimed | yes | specific number |
| final_awarded | yes | specific number |
| associated_interest | yes (nullable) | rate value must be verifiable |
| decision_on_penalty | yes | legal conclusion, must be anchored |
| moderation_applied | yes | critical distinction yes/no/not_applicable |
| legal_reasoning_summary | key_quotes | synthesis but needs 1-2 anchors for auditability |
| factors (each) | yes (nullable) | interpretive claim about court argument |