# Extraction pipeline notes

These are my notes about how the LLM extraction pipeline works. The goal of this
part is simple: take retrieved chunks from the RAG step and turn them into a
structured JSON about the contractual penalty. Then a lawyer can read the JSON
or i can compute statistics from it.

## Where this fits

```
01_raw_pdfs -> 02_processed_json -> 03_chunked_docs -> 04_vectorstore -> 05_retrieval_evaluation
              (data_cleaner.py)   (chunk_processor)  (vector_store)    (evaluate_retrieval)
                                                                              ↓
                                  07_extractions  ←  06_retrieval_results  ←  best config
                                  (run_extraction)   (precompute_retrieval)   (hybrid_a7_reranked_top7)
```

The extraction is the second half of the RAG. Retrieval finds the right chunks,
extraction asks the LLM to read them and fill the JSON. Without retrieval the
LLM would have to read 30k+ tokens for some documents which is slow and
expensive.

## Step 1: precompute retrieval results

Script: `src/retrieval/precompute_retrieval.py`
Output: `data/06_retrieval_results/{document}.json`

Before extraction i save which chunks each document gets. Reasons:
- reproducibility: same chunks every time, no random variation
- debugging: i can open the JSON and see what the LLM will get
- decoupling: if i change the prompt, i dont have to rerun retrieval

The script imports `do_retrieve()` from `evaluate_retrieval.py`. Same function
that produced 90.5% recall in evaluation. I set env vars before importing to
configure the winning config (hybrid, alpha=0.7, reranker, etc). No code
duplication.

What the JSON has:
- `call1_chunks` for contract facts (queries q_breach, q_contract, q_rate, q_principal)
- `call2_chunks` for moderation analysis (q_outcome, q_reasoning, q_factors)
- `header`, `verdict`, `reasoning_full` as full text segments
- `retrieval_config` snapshot
- `stats` with chunk counts and char counts

Different queries often retrieve the same chunk. For example a paragraph about
"zmluva o dielo" matches both q_breach and q_contract. I deduplicate within
each call group: keep the chunk once, save the max score, and remember which
queries hit it (`source_queries` list, useful for debugging).

About scores: the reranker uses min-max normalization within each query's
candidate pool. So 0.98 from q_rate and 0.72 from q_breach are not comparable.
The max() across queries is just rough debug info. The LLM never sees these
scores anyway, chunks go into the prompt sorted by document position
(chunk_index).

Adaptive top_k: for big documents (more than 30 chunks) the call1 queries
q_breach and q_contract get extra chunks, +1 per 20 paragraphs above 30, capped
at +3. Reason: contract context spreads across more numbered points in long
decisions, top_k=7 covers only 8% of an 84-chunk doc.

## Step 2: build prompts

Module: `src/extraction/prompt_builder.py`
Templates: `src/extraction/templates/`

### System prompt

Same for all calls and all models. 9 strict rules:

1. evidence-based: every evidence quote must be exact substring, character by character. No capitalization changes, no adding periods.
2. quote length: 5 to 50 words, complete thoughts, do not cut mid-clause.
3. no math: never compute values, only extract explicit numbers from the text.
4. chunk_id: always include chunk_id from the chunk header for auditability. Also "verdict" and "header" are valid chunk_ids.
5. language: all output in Slovak (same as the source text).
6. multiple penalties: separate entry per distinct penalty with its own rate. Do not split same penalty across invoices.
7. penalty vs interest: urok z omeskania is not zmluvna pokuta.
8. null handling: null value means null evidence (`{"quote": null, "chunk_id": null}`).
9. output: only valid JSON, nothing else.

The system prompt is in English. The reason is that LLMs follow instructions
better in English, they are trained mostly on English instruction data. The
output content (summaries, quotes) stays in Slovak because the source text is
Slovak and the lawyer reads Slovak.

### Call 1 prompt

Extracts: `case_context` and `contractual_penalties` (without
moderation_analysis).

I do not ask the LLM to extract `meta` here. The metadata fields like court
name, case number, date and ECLI are extracted by regex in preprocessing,
which is more reliable and cheaper.

The prompt gets: header (with chunk_id "header"), verdict (with chunk_id
"verdict"), call1_chunks (each labeled with their chunk_id).

I include the JSON schema as an example in the prompt so the LLM sees the
exact structure i want. Enum values are listed inline like
`contract_type: uver|pozicka|najom|...`.

### Call 2 prompt

Extracts: `moderation_analysis` for each penalty. Decision, reasoning, key
quotes, and 7 factors with sentiment.

The prompt gets: cleaned Call 1 JSON output (so the LLM knows what penalty
it is analyzing), verdict, call2_chunks.

Originally i sent only a short summary of Call 1 to save tokens (penalty_id +
breach_type + amounts). Then i changed it to send the full Call 1 values
because:
- the LLM needs `secured_principal` for the `pomer_k_istine` factor
- the LLM needs `associated_interest` for the `kumulacia_s_urokom` factor
- the LLM should keep the same `penalty_id` and analyze the same penalty
- the extra cost is around $0.001 per document, basically nothing

Before sending Call 1 output to Call 2 i strip all `evidence` and `chunk_id`
fields. This prevents what i call "cross-call contamination": Call 2 should
cite from its own moderation chunks, not copy chunk IDs from Call 1.

I also added the verdict text to Call 2 after finding that without it the LLM
sometimes returned `unclear` for the decision field, even when the verdict
clearly said "zamietol" (dismissed). The verdict gets chunk_id "verdict" so
the LLM can use it as evidence.

### Full-doc prompt

Extracts: everything in one call. Complete schema including
moderation_analysis and quality_control.

Gets: header, verdict, full reasoning text. No chunk selection. The LLM reads
the whole document. This is the baseline experiment, to test whether RAG adds
value compared to giving the LLM everything.

All chunk_ids are "full_doc" (or "header" / "verdict" for those sections)
because there are no individual chunks.

## Step 3: call the LLM

Module: `src/extraction/llm_client.py`

Unified interface: `call_llm(model_key, system_prompt, user_prompt, call_type)`.

### Models / providers

| Model | Provider | JSON guarantee | Cost |
|---|---|---|---|
| GPT-4o | OpenAI API | Structured Outputs (`json_schema`) | around $0.08 per doc |
| Gemini 2.5 Flash / Flash-Lite | Google GenAI API | `response_schema` | free tier, estimated $ if paid |
| Qwen 3.5 397B-A17B | OpenRouter | OpenAI-compatible `json_schema` | cheap paid API |

I also tried a local Slovak model (`Qwen3-14B-sk` via Ollama) for the privacy
argument. It did not work for this task. The model often failed to produce
valid JSON or repeated itself in a loop. I keep the Ollama wrapper code as
documentation and as a possible privacy direction. The working open-source
comparison in the current pipeline is Qwen 3.5 397B through OpenRouter
(Apache 2.0 license, anyone can self-host the weights).

### Why structured output matters

I originally used `response_format={"type": "json_object"}` (the old JSON
Mode). It only guaranteed valid JSON syntax. The model could still skip
fields, add extra fields, or use wrong types. I noticed it sometimes forgot
the evidence field.

After switching to Structured Outputs (`json_schema` with strict: true), the
model is forced to return exactly the schema. OpenAI claims 100% adherence.
This was the single biggest improvement in JSON quality.

For Gemini i use `response_schema` with the same schema (i strip
`additionalProperties` because Gemini does not support that field). For Qwen
through OpenRouter i use the OpenAI-compatible `json_schema` format. For the
old Ollama experiments i used the native `/api/chat` endpoint with the
`format` parameter (grammar-based constrained decoding).

The schemas are defined in `src/extraction/schemas/output_schemas.py` using
helper functions (`_evidence_obj()`, `_factor_obj()`, etc) so i do not copy
paste the same structures. Three schemas: `get_call1_schema()`,
`get_call2_schema()`, `get_fulldoc_schema()`.

### Parameters

| Parameter | Value | Why |
|---|---|---|
| temperature | 0.0 | Deterministic output. Same document = same JSON every time. |
| top_p | not set | At temperature=0 the model picks the most probable token, top_p has no effect. |
| max_output_tokens (call1/call2) | 8192 | Multi-penalty cases produce long JSON with many evidence quotes. |
| max_output_tokens (fulldoc) | 16384 | Everything in one response, needs more headroom. |

With structured outputs, max_output_tokens is just a safety limit. The model
cannot be verbose because it has to fill the schema fields and stop. It cant
add explanations or padding outside the JSON. The free-text fields
(`dispute_summary`, `legal_reasoning_summary`) are constrained by the prompt
itself ("2-3 sentences", "3-5 sentences").

### Where i tested prompts

- OpenAI Playground (https://platform.openai.com/playground), paste system + user prompt, select GPT-4o, set response format. Got $5 free credits when i signed up.
- Google AI Studio (https://aistudio.google.com), completely free, no credit card. Great for students, free tier 25 RPM and 1500 RPD is enough for 176 documents.
- Ollama, i tested local structured output with `Qwen3-14B-sk`, but the result was not reliable enough for the final experiments.

## Step 4: parse and validate

Module: `src/extraction/response_parser.py`

### JSON parsing

Three things to try if direct parse fails:
1. Direct `json.loads()` works for OpenAI Structured Outputs (always valid JSON).
2. Strip markdown fences. Some models wrap JSON in ` ```json ... ``` `, the regex catches that.
3. Find first `{` and last `}` and try to parse what is between. Last resort for messy output.

### Merge Call 1 + Call 2

For RAG mode i merge the two call results by matching on `penalty_id`. Call 1
gives the penalty object, Call 2 gives `moderation_analysis` for each penalty.
If Call 2 fails completely i still keep Call 1 result and fill the moderation
with empty placeholders.

### Evidence grounding check

This is my hallucination detector. After the first test on one document i was
surprised that 5 out of 13 evidence quotes were NOT exact substrings of their
claimed chunk. The LLM was doing things like:
- changing capitalization, "že zmluvná" became "Že zmluvná"
- truncating quotes with periods where the original continued with parentheses

I wrote `validate_evidence_grounding()` that goes through every evidence
field and tries to match the quote against the chunk text. There are 6
levels of matching, from strictest to most lenient:

1. EXACT, the quote is exact substring in the chunk, no flag
2. WHITESPACE normalized, catches "potvrdzuje ." vs "potvrdzuje." (preprocessing artifact), flag `EVIDENCE_WHITESPACE_NORMALIZED`
3. CASE-INSENSITIVE, the LLM capitalized the first letter, flag `EVIDENCE_CASE_MISMATCH`
4. LEMMATIZED, the LLM changed grammatical case, "zmluvnej pokuty" vs "zmluvnú pokutu" (Slovak inflection), flag `EVIDENCE_INFLECTION_MISMATCH`
5. FUZZY word overlap (>= 65% of quote words in a sliding window over the chunk), flag `EVIDENCE_QUOTE_REPAIRED` and the quote in the result is replaced with the actual chunk text
6. Nothing matched, flag `EVIDENCE_NOT_FOUND` (most likely hallucination) or `EVIDENCE_TRUNCATED` (first 40 chars match)

For a legal system this is critical. A lawyer would ctrl+F the quote to verify
it. If the quote does not match exactly, ctrl+F fails and the lawyer loses
trust in the whole tool. I added the auto-repair (level 5) so the output JSON
ends up with valid ctrl+F-able quotes even when the LLM paraphrased a little.

### Business logic validation

Same module also runs deterministic checks (no LLM involved):
- `final_awarded > original_claimed` -> flag `MATH_ERROR_AWARDED_EXCEEDS_CLAIMED` (court cant award more than claimed)
- `decision = moderated_301` but `claimed == awarded` -> flag `LOGIC_ERROR_MODERATION_BUT_SAME_AMOUNT` (moderation means reduction)
- `decision = dismissed` without `dismissal_reason` -> flag `MISSING_DISMISSAL_REASON`
- Missing key fields go to `missing_fields` list
- Anonymized amount detection (see below)
- Currency mismatch detection (see below)

## Step 5: run extraction

Script: `src/extraction/run_extraction.py`
Output: `data/07_extractions/{model}_{mode}/{document}.json`

Config at the top of the file:
```python
MODEL_KEY = "qwen3.5-397b"  # or "gpt-4o", "gemini-2.5-flash", "gemini-2.5-flash-lite"
MODE = "rag"             # or "fulldoc"
SINGLE_DOC_TEST = "..."  # comment out for full run
```

Each model x mode combination gets its own output folder. So 3 models x 2
modes = 6 folders. With 176 docs each thats up to 1056 extractions for the
full corpus comparison, but for the 20-document test set its 6 x 20 = 120.

The output JSON has: extraction config, API metadata (tokens, cost, latency),
and the full extraction result with quality_control flags.

## Things i fixed after testing

### Evidence accuracy progression

| Version | Exact match | Problems | What changed |
|---|---|---|---|
| V1 (initial) | 8/13 (61.5%) | 5 | baseline |
| V2 (fixes 1-5) | 9/12 (75.0%) | 3 | grounding check, verdict in call2, null handling |
| V3 (structured outputs) | 11/12 (91.7%) | 1 | json_schema, full call1 JSON to call2 |

### Fix 1: LLM changed capitalization of quotes

The LLM was turning "že zmluvná pokuta" into "Že zmluvná pokuta" when it pulled
the quote out of context. I strengthened the system prompt rule with an
explicit example and added "copy character by character" instruction.

### Fix 2: no evidence verification existed

The system claimed to be "evidence-based" but never checked if the quotes are
real. I added `validate_evidence_grounding()` with the 6-level matching above.

### Fix 3: Call 2 didnt get the verdict

Call 2 was analyzing moderation without seeing the verdict. The verdict
contains "zamietol žalobu" (dismissed) or "zaviazal zaplatiť" (awarded), which
is critical for the decision field. After adding it, the LLM stopped returning
"unclear" for clear cases.

### Fix 4: Call 2 got too little context from Call 1

Originally i only sent penalty_id + breach_type + amounts summary. I changed
it to send the cleaned Call 1 JSON so the LLM has all extracted values for
factor analysis. I strip `evidence` and `chunk_id` before Call 2, otherwise
the model copies old chunk IDs from Call 1 instead of citing from Call 2
chunks.

### Fix 5: full-doc mode had no validation

Full-doc returns everything in one call. If it forgot quality_control, it was
never added. Validation (math check, evidence grounding) never ran. I added
the same validation pipeline as RAG mode, and i also force `quality_control`
to be present even if the LLM forgot it.

### Upgrade: JSON Mode -> Structured Outputs

I switched from `json_object` (only valid JSON) to `json_schema` (schema
compliance). This was the biggest single improvement. The model can no longer
skip fields or use wrong types. Applied to all 3 providers (OpenAI, Gemini,
OpenRouter). Ollama also has an equivalent through the `format` parameter.

## File overview

| File | What it does |
|---|---|
| `src/retrieval/precompute_retrieval.py` | Runs retrieval on all docs, saves chunks to JSON |
| `src/extraction/prompt_builder.py` | Loads templates, formats chunks into prompts |
| `src/extraction/llm_client.py` | Calls OpenAI / Gemini / OpenRouter / Ollama with structured output |
| `src/extraction/response_parser.py` | Parses JSON, merges calls, validates evidence |
| `src/extraction/run_extraction.py` | Main orchestrator, loops over docs, runs pipeline |
| `src/extraction/templates/system_prompt.txt` | Extraction rules (evidence, no-math, language) |
| `src/extraction/templates/user_prompt_call1.txt` | Call 1 prompt (contract facts + penalty) |
| `src/extraction/templates/user_prompt_call2.txt` | Call 2 prompt (moderation analysis) |
| `src/extraction/templates/user_prompt_fulldoc.txt` | Full-doc prompt (everything in one call) |
| `src/extraction/schemas/extraction_schema.json` | Target schema v5.0 (what the final JSON looks like) |
| `src/extraction/schemas/schema_spec.md` | Human-readable schema specification |
| `src/extraction/schemas/output_schemas.py` | JSON schemas for structured output API |
| `data/06_retrieval_results/` | Precomputed chunks per document |
| `data/07_extractions/{model}_{mode}/` | Extraction results per model and mode |

## First run results: GPT-4o RAG on 20 documents

### Overall stats

| Metric | Value |
|---|---|
| Documents | 20 |
| Penalties extracted | 23 (3 docs had multiple) |
| Evidence quotes | 263 |
| Exact match | 216 (82.1%) |
| Usable (exact + case) | 227 (86.3%) |
| Wrong chunk_id | 5 (1.9%) |
| Not found | 20 (7.6%) |
| Cost | $1.59 total, around $0.08 per doc |

5 documents scored 100% on evidence accuracy. 6 more were above 90%.

### Cross-call contamination (WRONG_CHUNK errors)

When Call 2 received the full Call 1 JSON including evidence chunk_ids, it
copied those chunk_ids for its own evidence instead of referencing its own
chunks. This was most visible in 2Cob/69/2020 (3 penalties, 5 wrong_chunk
flags).

Fix: `_strip_evidence()` function in `prompt_builder.py` removes all
`evidence` and `chunk_id` keys from the Call 1 JSON before sending it to
Call 2. The LLM still sees all extracted VALUES but no chunk_ids that could
confuse it.

### secured_principal confusion (3 types of errors)

After analyzing all 23 penalties i found 3 distinct error patterns:

C1 fixed penalty confused with principal (2 cases). In 1Cob/130/2019 and
14Cob/202/2019, the penalties were fixed amounts (2000 EUR and 20000 EUR).
The LLM set `secured_principal` equal to the penalty amount, but for fixed
penalties there IS no base amount. Worse, the LLM picked up the number from
an `úrok z omeškania` clause instead of the penalty clause. Fix: explicit
instruction in the prompt: "For FIXED penalties set secured_principal to
null. Do NOT use amounts from úrok z omeškania clauses."

C2 wrong currency (1 case). In 2Co/380/2011, the text says "4.323.809,- Sk"
(Slovak koruna) but the LLM output `currency: "EUR"`. This is a pre-2009
decision. Fix: explicit instruction "If text says Sk or SKK, use SKK. Older
decisions before 2009 use SKK." Plus a validation flag `CURRENCY_MISMATCH_POSSIBLE_SKK`
that scans chunks for "Sk" patterns when the LLM said EUR.

C3 grammatical form in quote (1 case). In 43Cob/75/2024 the LLM wrote
"nárok na zaplatenie" but the text has "nároku na zaplatenie" (different
grammatical case in Slovak). This caused NOT_FOUND flag. The value (756) was
correct. Fix: i added LEMMATIZED match (level 4) in the grounding check, so
this kind of small inflection difference is detected and flagged with
`EVIDENCE_INFLECTION_MISMATCH` instead of NOT_FOUND.

### Preprocessing artifacts (TRUNCATED errors)

Several TRUNCATED flags (especially in 2Co/380/2011) were caused by spaces
before punctuation in the preprocessed text (e.g. "potvrdzuje ." instead of
"potvrdzuje."). This is an OCR / PDF extraction artifact from `data_cleaner.py`,
not an extraction bug. Fix: i updated `data_cleaner.py` to remove spaces
before punctuation, AND added the WHITESPACE level (level 2) in the grounding
check.

### Decision field accuracy

Out of 23 penalties:
- 12x awarded_full
- 9x moderated_301
- 1x dismissed
- 1x returned

This distribution looks reasonable for our corpus, which was selected
specifically for §301 moderation cases.

### Fix: returned cases had fake factors

When i manually checked 16 random documents against their source text, i
found that documents where decision=returned (case sent back for further
proceedings) had factors filled in, sometimes all positive or all neutral.
But this makes no sense, because when a case is returned the court has not
analyzed the penalty on its merits yet. The factors should all be
not_mentioned.

This happened in 3 out of 16 checked documents (2CoKR/71/2013, 1Cob/287/2014,
5Cob/51/2023). I added a rule to both call2 and fulldoc prompts: "When
decision is returned, set ALL factors to not_mentioned. Do not guess what
the court might decide."

### Fix: anonymized documents hallucinate amounts

Some court decisions have redacted numbers (like "X XXX,XX eur" instead of
real amounts). I found this when checking 4Cob/122/2011, which has 81
occurrences of "XXX" markers in the text. The LLM saw "X XXX,XX Sk za tonu
za XXX ton" and invented numbers like 600000 and 70000 that do not exist
anywhere in the original text.

I added a validation check in `response_parser.py`. If any evidence quote for
an amount field contains "XX" (anonymization marker, regex `r'X{2,}'`), it
gets flagged as `ANONYMIZED_AMOUNT`. These documents should be excluded from
statistical analysis because their numbers are unreliable. Around 20 of 176
documents (11.4%) have significant anonymization.

### Things that did not work, or are still open

1. Multi-penalty unclear decisions. 2Cob/69/2020 has 4 penalties but all got decision=unclear. The court DID decide each one but the relevant text didnt make it into Call 2 chunks. This is a retrieval limitation for very complex cases with many penalties discussed across many paragraphs.

2. Zmenka cases. 31CoZm/4/2022 has penalties embedded inside a zmenka (bill of exchange) dispute. The court did not analyze them separately, so the LLM correctly returns decision=unclear with empty reasoning. This is not a bug, this is the actual court output.

### Stuff i would still try if i had more time

- Cross-validation with regex. Compare LLM-extracted amounts against regex-detected amounts (from `legal_patterns.py`) as an independent check. If the LLM says 15000 but the regex finds only 12500 in the chunks, flag it.
- Repair loop. If grounding check finds NOT_FOUND, re-run that field with the specific chunk text and ask the LLM to fix the quote.
- Lawyer feedback loop. Currently lawyers see the final JSON and rate the fields. If they reject a field i should be able to feed that back into prompt iteration.
