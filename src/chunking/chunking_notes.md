# Chunking notes

i wrote this file to keep track of what works and what didnt for the chunking
part of the pipeline. main code is in `src/chunking/chunk_processor.py`.
evaluation lives in `src/evaluation_retrieval/evaluate_retrieval.py`.

## What this part does

i take cleaned reasoning text from the JSON output of preprocessing and split
it into smaller pieces -  chunks. 
embedding models have token limits, mE5 base only handles 512 tokens, OpenAI text-embedding-3-small handles 8192.
so i cant just feed a whole 10-page decision to the model.

i only chunk the `reasoning` segment because verdict and instruction are short
enough to send to the LLM directly without retrieval.


### Phase 1: fixed size chunks

i started with `RecursiveCharacterTextSplitter` from langchain. simple idea, just
split every N tokens and embed.

i had 5 strategies, later added a 6th (OPENAI_380) for fair comparison:

| name | model | chunk_size | overlap | use |
|---|---|---|---|---|
| ME5_200 | mE5-base (free, local) | 200 tok | 25 | very small, precise |
| ME5_380 | mE5-base | 380 tok | 50 | mE5 baseline |
| OPENAI_200 | text-embedding-3-small | 200 tok | 25 | fair model comparison |
| OPENAI_380 | text-embedding-3-small | 380 tok | 50 | fair vs mE5 |
| OPENAI_500 | text-embedding-3-small | 500 tok | 75 | OpenAI baseline |
| OPENAI_PARA | text-embedding-3-small | paragraph | 0 | natural boundaries |

mE5-base has a hard 512 token limit and silently truncates anything bigger.
slovak legal paragraphs are typically 300-800 tokens, so paragraph chunks
would not work for mE5 anyway.

results at top_k=5, dense retrieval, no window expansion:
- ME5_200: 46.2% recall, 26% coverage. way too small, arguments split mid sentence
- ME5_380: 58.4% recall, 36% coverage. better but still bad

both numbers were too low. needed to do something else.

### Phase 2: window expansion

idea was to keep small chunks for precise embeddings, but after retrieval also
pull in the neighboring chunks (±1 in the same document) to give LLM more context.

result: ME5_380 with window=1 went from 58.4% to 79.7% recall at top_k=5. nice
jump but kind of cheating. i was solving the small chunk problem by reading
3x more text per hit.

the real issue was that fixed size chunks cut through legal arguments. a
judge writes a 700 token argument as one numbered point ("12. Súd dospel k
záveru..."), and the 380 splitter chops it in half. each half alone is useless.
window expansion glues them back, but what if i just dont break them?

### Phase 3: paragraph chunks

slovak court decisions have a clear structure. the judge writes reasoning as
numbered points:

```
1. Súd prvej inštancie rozhodol...
2. Žalovaný vo vyjadrení uviedol...
3. Okresný súd dospel k záveru...
```

each point is one complete legal argument. it can be 50 tokens (procedural
note) or 2500 tokens (detailed analysis), but it is one thought, one topic.

so for OPENAI_PARA i started splitting at numbered point boundaries instead of
fixed token sizes. no overlap because paragraphs are already complete units.

initial implementation used `RecursiveCharacterTextSplitter` with chunk_size=1500
and paragraph separators. the regex `(?=\n\s*\d+\.\s+)` (zero-width lookahead)
splits before each numbered item. 1500 was a safety net for very long points.

result at top_k=5 with dense OpenAI: 73.3% recall. already better than fixed-size
without window expansion.

### Phase 4: comparing embedding models

with paragraph chunks working, i tested:
- OpenAI text-embedding-3-small (1536 dims, $0.02 per 1M tokens)
- OpenAI text-embedding-3-large (3072 dims, $0.13 per 1M tokens)
- mE5-base (768 dims, free, only fixed-size because of 512 limit)

OpenAI-small and OpenAI-large were almost the same on slovak legal text (73.3
vs 72.6 at top_k=5). small is 6.5x cheaper so i went with small.

### Phase 5: retrieval methods

i compared 3 retrieval methods on paragraph chunks at top_k=5:
- dense (cosine on embeddings): 73.3%
- BM25 (keyword TF-IDF style): 77.3%
- hybrid via RRF (combines both rankings): 80.6%

hybrid wins. dense catches paraphrases, BM25 catches exact legal terms like
"§301". both signals together is strictly better than either alone.

### Phase 6: keyword reranker

problem: hybrid retrieval sometimes ranks procedural text ("trovy konania")
above relevant passages ("§301 moderačné právo"). the initial ranking is not
perfect.

solution: 2-stage retrieval:
1. fetch 5x more chunks than needed (oversample)
2. rescore them using domain specific legal keyword patterns
3. keep top_k best after the rescore

the reranker scores each chunk on terms specific to each query type. for q3
(moderation) those are "neprimerane vysoká", "znížiť", "§301", "moderačné
právo". final score is `α * retrieval_score + (1-α) * keyword_score` with α=0.6.

i went with keyword reranking instead of a cross encoder (like ms-marco) because:
- cross encoders are slow, you have to run a transformer for every chunk-query pair
- keyword reranker is instant and domain specific
- it lets me bake in expert knowledge about slovak legal terminology

result: hybrid α=0.7 + reranker at top_k=7 = 90.5% recall. at top_k=10 it goes
up to 96.5% but coverage also rises substantially (more text for the LLM to read).

### Phase 7: hierarchical chunking (parents and children)

while optimizing flat paragraph chunks i also tried a different architecture:

1. split each numbered point into ~220 token children
2. embed the children, small means precise embeddings
3. when a child is retrieved, return the whole parent as context
4. aggregate scores across children to rank parents

idea was precise embeddings (small children) plus rich context (large parents).

i tested 90 hierarchical configs: mE5 vs OpenAI embeddings, dense vs BM25 vs
hybrid retrieval, different fetch_k and parent selection percentiles.

results:
- best hier config (mE5 + hybrid, fetch_k=16, c1p25/c2p27): 95.97% recall
- best flat para + reranker at k=10: 96.53% recall
- best flat para + reranker at k=7: 90.5% recall (this is the paper number)

so hier ended up basically tied with flat, the difference at the top is 0.56
percent points. not big enough to recommend the more complex pipeline. i
kept flat as the main strategy because:
- fewer parameters to tune (no fetch_k, no parent percentiles)
- simpler retrieval code, less moving parts
- matches the paper-reported Pareto point at top_k=7

i thought that hier was
"clearly worse". it was worse in the early configs i tested but later runs
with bigger fetch_k and tuned percentiles caught up.

### Phase 8: the merged chunks bug

while looking at the actual chunk content i found my OPENAI_PARA was not
doing what i thought. the recursive splitter with chunk_size=1500 was
splitting at numbered points, but then merging small points back together
until they reached 1500 tokens. so points 18-25 (eight separate arguments)
ended up in one 1384 token chunk.

how bad was it: 409 out of 2298 chunks (17.8%) had 2 or more numbered points
with 300+ tokens each. so basically multiple legal arguments fused into one
chunk.

example case:
- point 12 (567 tok): penalty calculation
- point 13 (489 tok): penalty amount for specific period
- point 14 (303 tok): §301 moderation analysis

all three got merged into one 1359-token chunk. searching for "§301
moderation" had to compete with the other two topics in the same embedding.

fix: rewrote the PARA chunker to use my own `split_reasoning_to_paragraphs()`:
1. split at each numbered point boundary using regex, each point = 1 chunk
2. no merging, no minimum size, even a 50 token note stays separate
3. for documents without numbered points (34% of the corpus) fall back to `\n\n`
4. safety net: split points exceeding 8000 tokens (only 8 of 3581 paragraphs)

later i added a second threshold `MAX_PARA_TOKENS = 4000` for sub splitting,
because even a 4000 token chunk has somewhat diluted embedding when it
covers multiple sub-topics. 4000 affects ~44 paragraphs (0.8%), 8000 affects
8 (0.2%). i tried 2000 first but that was too aggressive, it split 137
chunks and i lost 1.58% recall at top_k=7 because info from one paragraph
was spread over 2-3 chunks.

### Phase 9: tuning top_k after the chunking fix

after switching to true paragraph chunks, top_k=5 didnt mean what it used to
mean. old merged chunks were ~1200 tokens each, so k=5 effectively meant
reading ~15 numbered points worth. new chunks average ~500 tokens so k=5
means 5 points.

i extended the top_k grid from [2,3,4,5] to [2,3,4,5,6,7,10] and found that
top_k=7 is the Pareto sweet spot, 90.5% recall at 62% coverage.

if i go to k=10 i get 96.5% recall but 71% coverage, more text for the LLM
to read = more cost and more latency. paper reports k=7 as the chosen point.

### Phase 10: fixing the remaining failures

at ~89% recall i looked at which specific golden quotes the retriever was
still missing.

big documents suffer the most. in an 84-chunk document, k=7 per sub-query
covers only ~8% of the text. background facts in the middle of the document
get missed.

fix: adaptive q1 top_k. for documents with more than 30 chunks, add up to
3 extra chunks for the q1 (contract context) queries. those queries need to
find factual background that is spread across many numbered points in big
decisions.

reranker also needs more candidates. with oversample=3x the reranker only
saw 21 candidates for k=7. in big documents, relevant chunks may rank 25th
in the initial retrieval and never enter the reranker pool.

fix: bumped oversample from 3x to 5x. reranker now sees 35 candidates without
changing the final top_k that goes to the LLM.

i also found 2 broken golden CSV quotes (the character `ľ` got corrupted to
`U+FFFD`). these could never match. fixed in the CSV.

i tried adding a 3rd sub-query for q1 to capture "skutkový stav" (factual
background). result was only +0.42% recall at +1.6% coverage. the hybrid
retriever already finds those chunks via q_breach and q_contract. not worth
the extra coverage cost. removed.

## Bugs i had to fix

### A) characters vs tokens, hidden truncation

original code used `length_function=len` which counts CHARACTERS not tokens.
mE5 has hard 512 token limit and silently truncates anything bigger, no
warning. if i sent 600 tokens worth of text it just dropped everything past
512 internally. the embedding represented only the first part.

fix: real tokenizers. tiktoken for OpenAI, AutoTokenizer for mE5.

### B) chunks too big, semantic dilution

i originally set chunk_size=2000 thinking "more context is better". but a
2000 token embedding is a weighted average of everything in it. specific
facts like "penalty reduced to 500 EUR" get washed out by the surrounding
procedural text.

fix: dropped to 500 for fixed-size, then switched to paragraph chunking.

### C) slovak abbreviations breaking sentence splitter

i tried splitting at sentence boundaries: `(?<=[a-zA-Z])\.\s+(?=[A-Z])`.
slovak legal text is full of abbreviations: JUDr., ods., písm., sp., Z.z..
all of them matched the pattern and created useless micro chunks like
["JUDr", "Martin..."].

fix: removed the sentence splitter entirely. `\n\n`, `\n` and `;` are good
enough.

### D) comma splitter destroying sentences

i had `,\s+` as a separator. it split "Súd dospel k záveru, že pokuta je
neprimeraná" into two useless halves. comma was separator #5 of 6 so it only
fired when everything else failed, basically never useful.

fix: removed.

### E) numbered points detaching from their text

the splitter sometimes put the number "12." at the end of the previous
chunk, separated from its actual text content.

fix: zero-width lookahead `(?=\n\s*\d+\.\s+)`. matches a position without
consuming characters, so "12. Súd prvej inštancie..." stays intact at the
start of the new chunk.

### F) CSP. abbreviation bug in data_cleaner.py

`normalize_layout()` in `data_cleaner.py` joins broken PDF lines and uses
abbreviation detection (`sp.`, `zn.`, `napr.`) to know when not to split.

bug: lines ending in "CSP." (Civilný sporový poriadok) matched the `sp.`
pattern because the regex matched the END of the word. so lines ending in
"CSP." were joined with the next numbered point:

```
35. ...kritériám podľa § 421 CSP. 36. Pokiaľ dovolateľka...
```

points 35 and 36 ended up on one line, then in one chunk.

fix: added `\b` (word boundary) so `sp.` only matches standalone "sp.",
not the end of "CSP.":

```python
# old: matches end of "CSP."
ABBREVIATIONS = r"(?:ods\.|písm\.|sp\.|...)"
# new: only matches standalone "sp."
ABBREVIATIONS = r"(?:\bods\.|písm\.|\bsp\.|...)"
```

## Final separator lists

`SEPARATORS_STANDARD` for fixed size strategies (ME5_200, ME5_380, OPENAI_200,
OPENAI_380, OPENAI_500):

```python
SEPARATORS_STANDARD = [
    r"\n\n+",               # 1. paragraph break
    r"(?=\n\s*\d+\.\s+)",   # 2. before numbered items (zero-width)
    r"\n",                  # 3. any newline
    r";\s+",                # 4. semicolons
    r"\s+",                 # 5. whitespace, last resort
]
```

`SEPARATORS_PARAGRAPH` is the safety net for OPENAI_PARA when a single
point exceeds the 4000 or 8000 token thresholds:

```python
SEPARATORS_PARAGRAPH = [
    r"\n\n+",
    r"\n",
    r";\s+",
    r"\s+",
]
```

## How the final paragraph chunker works

`split_reasoning_to_paragraphs()` is the function that does the actual work:

1. check if reasoning text has numbered points (1., 2., 3., ...)
2. if yes (66% of corpus): split at each point boundary, each point = one chunk
3. if no (34% of corpus): fall back to splitting at `\n\n` paragraph breaks
4. if a point is bigger than 4000 tokens, sub-split using `_para_subsplitter`
5. if it is bigger than 8000 tokens, also runs through `_para_safety_splitter`
   (mostly to stay under OpenAI's 8192 token embedding limit)

no overlap, no merging, no minimum size.

result: chunks range from ~14 tokens (a short reference like "5. Áno.") to
~7500 tokens (detailed analysis). median is around 450 tokens, mean around
550. the wide range is correct because chunk size follows document structure,
not an arbitrary token budget.

## Metadata per chunk

every chunk gets these fields for traceability and window expansion:

| field | meaning |
|---|---|
| `source_file` | original PDF filename |
| `chunk_index` | position in document (used by window expansion ±1) |
| `strategy` | which model and tokenizer was used |
| `char_len` | character count |
| `token_len` | actual token count from real tokenizer |
| `case_id` | spisová značka (from preprocessing metadata) |
| `court` | court name (from preprocessing metadata) |

## Final retrieval config

after 268 experiments the winner is:

| param | value | why |
|---|---|---|
| chunking | true paragraph (1 numbered point = 1 chunk) | respects structure |
| embedding | OpenAI text-embedding-3-small (1536 dims) | cost vs quality |
| retrieval | hybrid α=0.7 (Dense + BM25 via RRF) | combines two signals |
| reranker | keyword reranker (α=0.6, 5x oversample) | domain specific rescore |
| top_k | 7 (adaptive +3 for q1 on big docs) | Pareto point |
| window | 0 | paragraphs are already complete |

results at this config:
- recall 90.5% at top_k=7, 96.5% at top_k=10
- coverage 62% at k=7, 71% at k=10
- MRR 0.510
- efficiency 1.45

## Things that didnt work

- hierarchical child-to-parent chunking. ended up basically tied with flat
  (95.97% vs 96.53% at k=10) but with more complex code. not worth the extra
  parameters to tune.
- window expansion w=1 after paragraph chunking was in place. doubles
  coverage for a small recall gain.
- q_facts sub-query for q1, only +0.42% recall at +1.6% coverage.
- OpenAI text-embedding-3-large, about the same as small but more expensive.
- chunk_size=2000, semantic dilution, worse retrieval precision.
- sentence dot splitter, breaks at slovak abbreviations.
- comma splitter, destroys sentence meaning.
- `RecursiveCharacterTextSplitter` with 1500 limit for PARA, merges small
  points back together (the bug from Phase 8).
