# Multi-doc precedent search, design notes

This is the design doc for the second deliverable of my thesis: a system that
takes a free-text question from a lawyer and returns relevant slovak court
decisions about contractual penalties + a few statistics. It uses everything the
extraction pipeline produced, so without that part this whole module would not
exist.

## 1. Why i built this

### The problem
A lawyer drafting a "zmluva o dielo" for 50000 EUR wants to know: how big a
contractual penalty for late payment can i set so the court does not knock it
down? Today they have to read dozens of decisions by hand. Hours of work.

The other side, a lawyer defending a company that already got hit with a
penalty, wants to know: what factors do courts most often use when they reduce
a penalty under §301? Are there decisions where the penalty was reduced under a
similar contract? Same problem, same hours of reading.

### What this system does
It uses the structured data my extraction pipeline produced and combines that
with embedding similarity. Naive multi-document RAG would just embed query
text and retrieve similar chunks across all documents. My system does this
instead:

1. Filter penalties by structured attributes (contract type, breach type). Only
   possible because the extraction pipeline already gave me these fields.
2. Rank what passed the filter using a few similarity signals.
3. Compute a few statistics over the matched penalties (merit ratio, how often
   each factor appears in moderation cases).
4. Return statistics + top 5 penalties as "precedent cards". One court decision
   can have multiple penalties so each penalty gets its own card. The lawyer
   does the interpretation. The system is a deterministic search tool, not a
   chatbot.

### What kind of system this is, technically
This is NOT RAG in the strict sense. RAG = Retrieval-Augmented GENERATION. There
is no LLM-generated answer at the end here. I deliberately removed it after
talking to lawyers, more on that later.

If i had to put a label on it, the closest fit is Case-Based Reasoning + faceted
search + vector ranking. The "case base" is the penalty index extracted from 176
decisions. Filtering happens over structured facets. Ranking uses cosine
similarity over OpenAI embeddings. The only place an LLM is used is at the very
beginning, to parse the natural language query into filter values + a semantic
query string. That part is inspired by Multi-Meta-RAG (Zareieh et al., 2024),
but i borrowed only the query-parsing idea, not the generation step.

For the thesis i describe it like this: a precedent search system based on
Case-Based Reasoning (Aamodt & Plaza, 1994) where the case base comes from the
extraction pipeline. The query is parsed by an LLM (Gemini Flash). Filtering is
faceted search (Tunkelang, 2009) over structured attributes. Ranking is cosine
over OpenAI embeddings. The output is deterministic statistics + 5 precedent
cards. NOT a Retrieval-Augmented Generation system, because the LLM is not used
to generate the answer. That decision was on purpose, to avoid hallucinated
generalizations from small samples.

## 2. What data i have to work with

### Extraction JSONs in `data/07_extractions/gpt-4o_rag/`
176 documents, each with the full extraction:

```
result.case_context:
  contract_type    - 12 enums (dielo, najom, uver, kupna, ...)
  relationship_type - B2B | B2C | C2C | unknown

result.contractual_penalties[] (1-5 per document):
  breach_type.value           - 5 enums
  rate_definition.type        - 10 enums (percent_denne, fixna_suma_jednorazovo, ...)
  rate_definition.value_raw   - "0,05% denne z dlznej sumy"
  amounts.currency            - EUR | SKK | CZK | unknown
  amounts.secured_principal   - number or null
  amounts.original_claimed    - number or null
  amounts.final_awarded       - number or null
  associated_interest.awarded - yes | no | unclear
  moderation_analysis:
    decision.value              - awarded_full | moderated_301 | dismissed | returned
    legal_reasoning_summary     - 2-5 sentences in slovak
    key_quotes[]                - [{quote, chunk_id}]
    factors[] (always 7):
      label     - dobre_mravy | zabezpecovacia_funkcia | vyska_skody |
                  pomer_k_istine | spravanie_dlznika | kumulacia_s_urokom |
                  spravanie_veritela
      sentiment - positive | negative | neutral | not_mentioned

result.quality_control.flags - ANONYMIZED_AMOUNT, MATH_ERROR, EVIDENCE_NOT_FOUND, ...
result.meta - doc_id, court_name, case_number, decision_date, ecli
```

### Penalty-level index in `data/10_penalty_index/`
The `build_penalty_index.py` script takes all 176 JSONs and produces:
- `penalty_index.jsonl` - one row per penalty (around 202 records)
- `embeddings.npy` - numpy matrix [N, 1536] of penalty card embeddings

Why penalty-level and not document-level: 8.5% of documents have multiple
penalties with different attributes. Filtering at the document level would
create false matches. Example: a filter on "late_payment AND awarded_full"
would match a document where pokuta_1 is "late_payment + moderated" and
pokuta_2 is "non_monetary + awarded". That combination does not actually exist
in the document. By indexing each penalty separately i avoid this.

## 3. The pipeline, 4 stages

```
QUERY: "Robim zmluvu o dielo za 50k EUR. Aku pokutu za omeskanie dat?"
  |
  v
STAGE 1: query understanding ─────────────────────────────────────────
  Gemini 2.5 Flash with structured output schema.
  Input: free text query.
  Output: {
    contract_type: "dielo",
    breach_type: "late_payment",
    decision_interest: null,
    factor_interest: [],
    amount_hint: 50000,
    intent: "safe_rate",
    semantic_query: "zmluvna pokuta za omeskanie zmluva o dielo"
  }
  |
  v
STAGE 2: penalty-level filtering ─────────────────────────────────────
  In-memory python filtering over 202 penalty records.
  Hard filters: contract_type (with a near-miss map), breach_type.
  Input: 202 penalties.
  Output: ~49 penalties matching contract_type IN {dielo, dodavka_sluzieb}
          AND breach_type = late_payment.
  |
  v
STAGE 3: ranking ─────────────────────────────────────────────────────
  4-dim scoring:
  - embedding similarity (40%)  cosine(query_emb, penalty_card_emb)
  - amount proximity      (25%) ratio min/max of amounts
  - bonus attributes      (20%) decision_interest + factor_interest match
  - authority             (15%) NS SR > KS > OS
  Input: ~49 penalties.
  Output: top 5 penalties by score, used as precedent cards.
          (5 penalties, not 5 decisions. one decision can be represented
          by multiple penalties, each gets its own card.)
  |
  +-------------------------------+
  |                               |
  v                               v
STAGE 4: analytics              [TOP 5 penalties]
  n<5: just precedents            metadata, amounts, decision, 7 factors,
  n>=5: merit ratio + factors     reasoning_summary, key_quotes.

  Computes only 2 things:
  - merit ratio (moderation
    among awarded + moderated)
  - factor frequencies in
    moderated cases
  |                               |
  +-------------------------------+
  |
  v
OUTPUT:
  1. STATISTICS - python computed, deterministic, slovak plain language
  2. PRECEDENTS - top 5 penalty cards with metadata, amounts, citations
     (one decision may be represented by multiple penalties)
```

Why no LLM synthesis at the end: after talking to lawyers i decided that an
LLM-generated interpretation adds risk (small-sample generalizations,
hallucinated trends) without adding much. Statistics and precedent cards are
deterministic. The lawyer does the interpretation. The system is a precedent
search tool with analytics, not a chatbot.

## 4. Each stage in detail

### Stage 1: query understanding (`query_analyzer.py`, ~170 lines)

Gemini 2.5 Flash with structured output (`response_schema`).

Why an LLM and not regex: slovak has 6 cases x 2 numbers, so
"zmluva/zmluvy/zmluvou/zmluve o dielo" alone would need many patterns. And
regex would not catch "IT zakazka" = "zmluva o dielo" anyway. Gemini Flash
costs around $0.003 per call and one code path is simpler than two.

The output schema (`schemas.py:get_query_understanding_schema()`) enforces:
- `contract_type`, nullable enum (12 values)
- `breach_type`, nullable enum (5 values)
- `decision_interest`, nullable enum (5 values)
- `factor_interest`, array of factor labels (0-7)
- `amount_hint`, nullable number
- `intent`, enum (safe_rate | defense_args | general_precedent)
- `semantic_query`, string (rewritten query for embedding)

Retry logic: on 503/UNAVAILABLE wait 10s, 20s, 30s, 40s. If the LLM keeps
failing, fallback to empty intent (pipeline still runs, just without filters).

### Stage 2: penalty-level filtering (`penalty_index.py`, ~150 lines)

Class `PenaltyIndex`. `__init__` loads the JSONL + numpy embeddings (once at
startup). `filter(intent)` applies hard filters.

Hard filters apply only to:
- `contract_type` (with a `SIMILAR_CONTRACT_TYPES` map for near-miss: "dielo"
  also matches "dodavka_sluzieb", "uver" matches "pozicka")
- `breach_type` (exact match)

`decision_interest` and `factor_interest` are NOT hard filters. They feed into
Stage 3 as soft ranking signals. Reason: a hard filter on `decision = awarded_full`
would make the analytics meaningless ("100% awarded", duh, i filtered for it).
A hard filter on specific factors would distort factor frequencies (it would
only show factors the lawyer asked about, not the actually most common ones).

Why in-memory and not a database: 202 records (in the future maybe 1500 with
1000 documents) is a tiny dataset. Python dict filtering is faster and simpler
than SQLite. SQLite would make sense from around 10k records.

### Stage 3: ranking (`ranker.py`, ~244 lines)

Four scoring dimensions with weights:

a) Embedding similarity (40%):
   - Query embedding via OpenAI `text-embedding-3-small` on `semantic_query`
   - Cosine similarity against all filtered penalty card embeddings at once
   - numpy `np.dot()`, vectorized, around 1ms for 49 penalties

b) Amount proximity (25%):
   - Ratio `min(query_amount, penalty_amount) / max(...)`
   - 50k vs 48k => 0.96, 50k vs 500k => 0.10
   - Prefer `secured_principal`, fallback to `original_claimed`
   - If data is missing: neutral 0.5 (do not penalize)

c) Bonus attributes (20%):
   - +2.0 if decision matches (when decision_interest is specified)
   - +1.0 per matching factor (when factor_interest is specified)
   - Normalized to [0, 1]
   - Neutral 0.5 if nothing was specified

d) Authority (15%):
   - NS SR (level 3) = 1.0, KS (level 2) = 0.67, OS (level 1) = 0.33
   - Simple: `auth_level / 3.0`

Final score: `0.4 * emb + 0.25 * amt + 0.2 * bonus + 0.15 * auth`

Output: list of `(record, score)` tuples, sorted by score desc.

### Stage 4: analytics (`analytics.py`, ~230 lines)

Pure python aggregation. No ML, no API calls. Just `Counter` and arithmetic.

Two things i compute, both with a story i can defend in front of a lawyer:

1. Merit ratio. Moderation rate counted ONLY among awarded + moderated cases.
   Dismissed and returned cases failed for formal reasons (invalid contract,
   procedural errors), not because the rate was too high. If 49 penalties
   split into 15 awarded / 9 moderated / 11 dismissed / 14 returned,
   "moderation rate 18.4% (9/49)" would be misleading. The real rate among
   merit decisions is 9/(15+9) = 37.5%. I compute it only if there are at
   least 3 merit cases (otherwise None).

2. Factor frequencies in moderated cases. For each of the 7 factors i count
   how often it appears as "negative" in cases where the court reduced the
   penalty. Example output: "pomer pokuty k istine: 8 z 9 znizenych pripadov",
   "kumulacia s urokom: 3 z 9 znizenych". Sorted by frequency.

Just simple counts. No lift, no contrasts with awarded cases. I had a more
complex lift analysis (P(neg|mod) / P(neg|awd)) earlier but after lawyer
feedback i dropped it. Lawyers care about "what does the court most often cite
when reducing the penalty", not "X-times more often than in awarded cases".
Simpler computation, simpler explanation, no statistical jargon to defend.

Sample size threshold: `MIN_SAMPLE_SIZE = 5`. Below n=5 the analytics returns
no statistics, just a note that the dataset is small.

Dataset bias caveat: always shown along with statistics, that the corpus is
not representative (collected via "zmluvna pokuta" + §301 keywords).

Output formatting: `format_for_lawyer(analytics)` builds a plain-language
slovak text with the merit ratio (`V 37.5% pripadov sud znizil`) and factor
frequencies (`pomer pokuty k istine: 8 z 9 pripadov`). This text goes into
`result["statistics_text"]` which the lawyer sees directly.

### Orchestrator (`pipeline.py`, ~200 lines)

Class `PrecedentSearchPipeline`. `__init__()` loads the PenaltyIndex.
`search(query)` runs stages 1-4 and returns the full output (statistics +
precedent cards).

Helper `_build_precedent_card(rec, score, rank)` in pipeline.py: from a
`(rec, score)` tuple it builds a flat dict with 18 fields for display, like
case_number, court_name, decision, amounts, legal_reasoning_summary,
key_quotes, etc.

Error handling:
- Stage 1 fails => empty intent, pipeline continues without filters.
- Stage 2 matches nothing => fallback to all records.

I time each stage just for debug. Total around 3-4s: query 1s, filter 0ms,
rank 1s, analytics 5ms.

## 5. What the system returns

For each query the system returns 2 parts: deterministic statistics + concrete
precedents. No LLM-generated answer. The lawyer gets facts and precedents and
does the interpretation themselves.

### Part 1: statistics (python computed)
```
ZHRNUTIE (49 relevantnych pripadov)
============================================================

V 37.5% meritornych pripadov sud znizil pokutu (9 z 24).

Najcastejsie dovody znizeni (v 9 znizenych pripadoch):
  - vyska skutocnej skody.............. 8 z 9 pripadov
  - pomer pokuty k istine.............. 8 z 9 pripadov
  - zabezpecovacia funkcia pokuty...... 5 z 9 pripadov
  - dobre mravy........................ 4 z 9 pripadov
  - kumulacia s urokom z omeskania..... 3 z 9 pripadov
  - spravanie dlznika.................. 2 z 9 pripadov
  - spravanie veritela................. 2 z 9 pripadov

Poznamka: Statistiky su zalozene na korpuse... Nezastupuju vsetky slovenske sudy.
```

### Part 2: precedents, top 5 penalties from Stage 3
(remember: 5 penalties, not 5 decisions. one decision can be represented by
multiple penalties, each gets its own card)

```
  [1] [NS SR] Najvyssi sud 4Obdo/96/2021 (30.09.2022)
      dielo | late_payment | 25,- eur za kazdy den omeskania
      POTVRDENA (score: 0.722)
  [2] Krajsky sud Bratislava 4Cob/216/2015 (03.11.2016)
      dielo | late_payment | 0,05 % denne zo sumy 79.107,11 eur
      POTVRDENA (score: 0.705)
  ...
  [5] Krajsky sud Trnava 32Cob/3/2021 [pokuta_2] (28.06.2022)
      dodavka_sluzieb | late_payment | 50% z mesacneho pausalu
      ZNIZENA (s301) (score: 0.461)
```

Each precedent card has: rank, case_number, court_name, decision_date,
contract_type, breach_type, rate_value_raw, amounts (principal/claimed/awarded),
decision, legal_reasoning_summary, key_quotes, score, authority_level.

## 6. Comparison with alternatives

### vs. naive multi-doc RAG (with LLM generation)

|                        | Naive multi-doc RAG           | This system (CBR + faceted search)    |
|------------------------|-------------------------------|---------------------------------------|
| Query                  | embedding of full text        | parsed into filters + semantic part   |
| Search space           | 5300+ chunks at once          | 202 penalty records => ~49 filtered   |
| Statistics             | not possible                  | merit ratio, factor frequencies       |
| Explainability         | "score 0.82"                  | "matched: same type + similar amount" |
| Output                 | chunks + LLM-generated answer | statistics + precedent cards          |
| Hallucination risk     | yes (LLM generates)           | no (everything deterministic)         |
| Speed                  | slow (global search)          | fast (pre-filtered space)             |
| Precision              | noisy chunks in top           | only relevant contract types          |
| Depends on extraction  | no                            | yes, this is the key argument         |

### vs. classic keyword search (slov-lex.sk, judikaty.sk)

|                        | Keyword search                | This system                          |
|------------------------|-------------------------------|--------------------------------------|
| Query                  | exact text match              | LLM-parsed structured query          |
| Structured filters     | manual (advanced search)      | automatic from natural language      |
| Sorting                | by date / alphabetical        | multi-dim similarity scoring         |
| Stats over filtered    | none                          | merit ratio, factor frequencies      |
| Slovak morphology      | very limited                  | LLM handles it natively              |

The biggest difference is that this whole system EXISTS only because the
extraction pipeline produced clean structured data. Without it i could not do
CBR-style search (no case base) and could not do faceted filtering (no facets).

## 7. Files and size

| File                                   | Lines | What it does                          |
|----------------------------------------|-------|---------------------------------------|
| src/multi_doc/__init__.py              | 0     | empty package marker                  |
| src/multi_doc/schemas.py               | ~174  | enum values + JSON schema + prompt    |
| src/multi_doc/build_penalty_index.py   | ~448  | one-time preprocessing                |
| src/multi_doc/query_analyzer.py        | ~171  | Stage 1, LLM query understanding      |
| src/multi_doc/penalty_index.py         | ~152  | Stage 2, filtering with 1-pass mask   |
| src/multi_doc/ranker.py                | ~245  | Stage 3, 4-dim ranking                |
| src/multi_doc/analytics.py             | ~230  | Stage 4, merit ratio + factor freq    |
| src/multi_doc/pipeline.py              | ~205  | orchestrator + precedent cards        |
| src/multi_doc/demo.py                  | ~184  | demo script with 5 example queries    |
| total                                  | ~1809 |                                       |

## 8. Key design decisions

### Penalty-level index, not document-level
8.5% of documents have multiple penalties with different attributes. Filtering
at document level would produce false cross-penalty combinations. Verified on
real data: KS_Trencin_8Cob_38_2012 has pokuta_1=dismissed+non_monetary and
pokuta_2=moderated+early_termination. The combination "dismissed +
early_termination" does not actually exist in this document. Penalty-level
indexing avoids this.

### LLM query understanding, not regex
Slovak has 6 cases x 2 numbers, so 12+ forms per noun. Regex for 12 contract
types would need many patterns. Plus "IT zakazka" = "zmluva o dielo" but
regex cant catch it (the word "dielo" is not in the text). Gemini Flash costs
around $0.003 per call and handles morphology natively.

### Hard filter only on contract_type + breach_type
A hard filter on decision would make the analytics tautological (filter on
awarded_full => 100% awarded). So decision_interest and factor_interest are
soft ranking signals in Stage 3, not hard filters in Stage 2.

### Just simple factor frequencies, no lift analysis
I originally had a lift analysis (P(neg|mod) / P(neg|awd)) with 3-state edge
case logic (numeric / only_in_moderated / insufficient_data). After lawyer
feedback i removed it. Raw frequencies from moderated cases ("pomer pokuty
k istine: 8 z 9 pripadov") are what lawyers actually want. They want to know
"what the court most often mentions when reducing the penalty", not a "X-times
more often than in awarded cases" stat. Simpler computation, simpler
explanation, no statistical jargon to defend.

### Merit ratio (not full distribution)
Dismissed and returned cases failed for other reasons (invalid contract,
procedural errors), not because of the rate. The moderation rate among merit
decisions (awarded + moderated only) is more useful for the lawyer than a
total distribution that includes dismissed/returned percentages.

### numpy, not ChromaDB for multi-doc
At 202 records (in the future ~1500) brute-force numpy dot product is faster
than ANN search in ChromaDB. No database dependency. ChromaDB makes sense
from millions of vectors.

### No deep retrieval (chunk-level)
The extraction pipeline already gives me around 300 tokens of text context per
penalty (reasoning_summary + key_quotes + factor evidence). Deep retrieval
would just return chunks from the same paragraphs. Pure redundancy. Without
it: simpler code, around 1.5s less latency, easier to defend.

### No LLM synthesis at the end of the pipeline
After talking to lawyers i decided that an LLM-generated interpretation of
precedents adds risk without adding value:
- on small samples (n=3, n=5) the LLM wrote authoritative-sounding paragraphs
  ("the courts in the analyzed corpus tend to...") from 1-2 cases
- most of what the LLM wrote was just paraphrasing the statistics and
  precedents the lawyer already sees directly in the cards
- every court case is in practice different (different contracts, amounts,
  context), and a "trend" from 5-10 cases does not transfer to a specific
  new situation
- hallucination risk (even with grounding) was higher than the added value

So the system returns deterministic statistics + 5 precedent cards with
reasoning_summary and key_quotes. The lawyer does the interpretation
themselves. That matches legal practice (precedents are read and
interpreted by a jurist, not by a chatbot).
