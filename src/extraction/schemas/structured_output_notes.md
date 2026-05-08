# Structured output schemas, notes

File: `src/extraction/schemas/output_schemas.py`

## Why structured output and not just "give me JSON please"

I started with `response_format={"type": "json_object"}` (OpenAI JSON Mode).
This guaranteed the response is valid JSON. But the model could still:
- skip fields i asked for (no breach_type in output)
- add extra fields i didnt ask for
- use wrong types (string "180" instead of number 180)

After switching to `response_format={"type": "json_schema", "json_schema": schema}`
(Structured Outputs), the model is forced to match my schema exactly. OpenAI
claims 100% adherence. Gemini and Ollama have similar features through their
own APIs (`response_schema` for Gemini, `format` for Ollama, both grammar-based
constrained decoding).

This was the single biggest improvement in JSON quality during the whole
extraction work, much bigger than any prompt change.

## Three schemas for three call types

I have 3 schemas because each call returns different fields.

### `get_call1_schema()`, contract facts

Returns: `case_context` and `contractual_penalties` (without
`moderation_analysis`).

This is what Call 1 extracts. Who signed what contract, what penalty was
agreed, how much was claimed, what was finally awarded.

It does NOT return `meta`. Metadata fields like court name, case number,
date, ECLI are added later in `run_extraction.py` from regex output of
preprocessing. They are more reliable from regex than from the LLM, and
asking the LLM to extract them again wastes tokens.

### `get_call2_schema()`, moderation analysis

Returns: `penalties_moderation` array, each item has `penalty_id` plus
`moderation_analysis`.

This is what Call 2 extracts. Did the court moderate or not, why, what
factors did it consider. The matching to Call 1 happens in
`response_parser.merge_call1_call2()` by `penalty_id`.

### `get_fulldoc_schema()`, everything in one shot

Returns: `case_context`, `contractual_penalties` (WITH `moderation_analysis`
this time), and `quality_control`.

This is for full-document mode where the LLM gets the entire text and
extracts everything in one call. Same as RAG mode, `meta` is added later
from deterministic regex metadata.

## How the schemas are built

I use small helper functions to avoid copy-pasting the same nested objects.
The schema files would be 1500+ lines without them.

| Helper | What it returns | Used in |
|---|---|---|
| `_nullable_string()` | `anyOf: [string, null]` | date, ecli, value_raw, etc |
| `_nullable_number()` | `anyOf: [number, null]` | amounts (secured_principal, etc) |
| `_evidence_obj()` | `{quote, chunk_id}` both nullable | every evidence field |
| `_factor_obj()` | `{label (enum), sentiment (enum), evidence}` | factors array |
| `_meta_obj()` | doc_id, court, case_number, date, ecli | legacy helper, not used in current LLM schemas |
| `_case_context_obj()` | summary, verdict_summary, contract_type, relationship | all 3 schemas |
| `_amounts_obj()` | currency + 3 amount fields with evidence | penalty object |
| `_moderation_analysis_obj()` | decision + reasoning + quotes + factors | call2, fulldoc |
| `_penalty_call1_obj()` | penalty without moderation | call1 schema |
| `_penalty_fulldoc_obj()` | penalty with moderation | fulldoc schema |

## OpenAI strict mode requirements

These are the rules i had to follow for the schemas to work with OpenAI
`strict: true`:

1. Every object must have `"additionalProperties": false`. Model cant add extra fields.
2. ALL properties must be in the `"required"` array. No optional fields. If a field can be missing, make it nullable in the type.
3. Nullable fields use `"anyOf": [{"type": "string"}, {"type": "null"}]`. Cant use shorthand `"type": ["string", "null"]`.
4. Enum values must be explicit. `"enum": ["late_payment", "non_monetary_performance", ...]`.
5. Arrays need `"items"` with full schema. `"type": "array", "items": {...}`.

The same schemas work for Gemini, but i pass only the inner `"schema"` part
without the OpenAI wrapper. I also strip `additionalProperties` because
Gemini does not support that field and returns 400 INVALID_ARGUMENT if it
sees one.

For Ollama i pass the same schema to the `format` parameter and Ollama uses
grammar-based constrained decoding to enforce it.

## Enums in the schemas

All enum values match the extraction schema v5.0 spec:

- contract_type: uver, pozicka, najom, dielo, kupna, dodavka_sluzieb, sprostredkovatelska, telekom, preprava, mandatna, ine, nezname
- relationship_type: B2B, B2C, C2C, unknown
- breach_type: late_payment, non_monetary_performance, early_termination, breach_of_confidentiality, other
- rate_definition.type: percent_denne, percent_mesacne, percent_rocne, percent_jednorazovo, percent_z_ceny, percent_z_dlznej_sumy, fixna_suma_denne, fixna_suma_mesacne, fixna_suma_jednorazovo, ine
- currency: EUR, SKK, CZK, unknown
- decision.value: awarded_full, moderated_301, dismissed, returned, unclear
- dismissal_reason: contract_invalidity, clause_invalidity, unproven_breach, procedural, other (or null)
- factor.label: dobre_mravy, zabezpecovacia_funkcia, vyska_skody, pomer_k_istine, spravanie_dlznika, kumulacia_s_urokom, spravanie_veritela
- factor.sentiment: positive, negative, neutral, not_mentioned

The `spravanie_veritela` factor was added after reading NS SR decisions where
courts considered whether the creditor was passive, just waiting for the
penalty to grow instead of enforcing performance. This is not in the standard
list of "factors of proportionality" you find in textbooks, but it appears in
real Slovak court reasoning often enough that i wanted a slot for it.
