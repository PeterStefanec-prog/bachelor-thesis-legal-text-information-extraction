# Document Chunking – Notes and Problems

[//]: # (> **File:** `src/chunking/chunk_processor.py`)

---

## 1. What this step does

Take cleaned reasoning segments from Slovak court decisions and split them into smaller pieces called "chunks". AI embedding models have strict input token limits, so i can't feed them a whole 10-page document at once.

I only chunk the `reasoning` segment. The `verdict` is short enough to always pass to LLM directly without RAG.

---

## 2. Chunking strategies

I'm testing five different strategies for my thesis. The key variables are which embedding model, how big each chunk is, and how much overlap between chunks:

| Strategy | Model | chunk_size | overlap | Notes |
|---|---|---|---|---|
| ME5_200 | mE5-small (local) | 200 tokens | 25 (12.5%) | very fine-grained |
| ME5_380 | mE5-small (local) | 380 tokens | 50 (13.2%) | baseline |
| OPENAI_200 | text-embedding-3-small | 200 tokens | 25 (12.5%) | for model comparison |
| OPENAI_500 | text-embedding-3-small | 500 tokens | 75 (15%) | baseline |
| OPENAI_PARA | text-embedding-3-small | paragraph-based | 0 | natural boundaries |

mE5 strategies are capped at 380 tokens max because mE5-small has a hard 512 token limit and silently truncates. Typical Slovak legal paragraph is 300–800 tokens so OPENAI_PARA is only for OpenAI.

---

## 3. Problems i had to fix

### A) Characters vs. tokens (hidden truncation)

My original script used `length_function=len` (Python default) which counts **characters** not tokens. i just assumed 1 token ≈ 4 chars as a rough estimate.

**The problem:** mE5 has a hard limit of 512 tokens and **silently truncates** the input without any warning or error. If i sent 600 tokens worth of text, the model just dropped everything after token 512. No exception. No log. The embedding would then represent only the first part of the chunk and i'd have no idea.

**The fix:** switched to exact tokenizers for each model. `tiktoken` for OpenAI and `AutoTokenizer` from HuggingFace for mE5. Now chunk sizes are measured in how the actual model reads the text.

```python
def me5_token_len(text: str) -> int:
    return len(me5_tokenizer.encode(text, add_special_tokens=False))
```

---

### B) Semantic dilution (chunk size too big)

Because OpenAI supports up to 8191 tokens i originally set `chunk_size=2000` thinking "more context = better".

**The problem:** an embedding of a 2000-token chunk is basically a weighted average of everything inside it. A specific detail like "penalty was reduced to 500 EUR" gets completely diluted by surrounding text about procedural history and costs. Cosine similarity between a precise query ("penalty amount") and a huge chunk is lower than against a small focused chunk containing exactly that sentence.

We want "sharp spears not wide nets" for retrieval.

**The fix:** reduced OpenAI chunk to **500 tokens**. This is fine because when the retriever finds the right small chunk, `evaluate_retrieval.py` uses Window Expansion (±1 chunk) to pull surrounding context anyway. So the LLM still gets the full picture, and retrieval is more precise.

---

### C) Slovak legal abbreviations breaking the sentence splitter

i tried to split at sentence boundaries using this regex:
```
r"(?<=[a-zA-Zá-žÁ-Ž])\.\s+(?=[A-ZÁ-Ž])"
```
The idea: cut after a period when preceded by a letter and followed by uppercase. Looked smart.

**The problem:** Slovak legal text is full of abbreviations that exactly match this pattern:
- `JUDr. Martin Vladik` → split into `["JUDr", "Martin Vladik decided..."]`
- `v zmysle Z.z. Obchodného` → split into `["v zmysle Z.z", "Obchodného zákonníka..."]`
- `Mgr.`, `Ing.`, `MUDr.`, `ods.`, `písm.` → all broken

This created micro-chunks that are completely useless for retrieval.

**The fix:** removed the sentence dot splitter entirely. After `data_cleaner.py` the reasoning text already has `\n\n` between paragraphs, so `\n\n`, `\n` and `;` handle everything without this risky regex.

---

### D) Comma splitter destroying sentence meaning

i also had `r",\s+"` in the separator list.

**The problem:** commas split sentences mid-thought. For example:
```
"Súd dospel k záveru, že pokuta je neprimeraná."
```
becomes:
```
["Súd dospel k záveru", "že pokuta je neprimeraná."]
```
Both chunks lose their meaning without the other half.

On top of that, comma was separator #5 out of 6, meaning it only ever triggered after `\n\n`, `\n` and `;` all failed on a single chunk - which basically never happens in cleaned legal text. So it brought no benefit and only risk.

**The fix:** removed.

---

### E) Numbered paragraphs getting split from their text

Legal decisions use numbered paragraphs (`1.`, `2.`, `12.`). The standard splitter was taking the number and leaving it at the **end** of the previous chunk, separated from its actual text.

**The fix:** used a zero-width lookahead regex:
```python
r"(?=\n\s*\d+\.\s+)"
```
Zero-width means the regex matches a position but consumes no characters. So the splitter cuts *exactly before* the number starts, and `"1. Súd prvej inštancie..."` stays intact at the beginning of the new chunk. `keep_separator` has no effect on this (nothing to keep) - the behavior comes purely from the lookahead being zero-width.

---

## 4. Two separator lists

**SEPARATORS_STANDARD** — used for all fixed-size token strategies (ME5_200, ME5_380, OPENAI_200, OPENAI_500):

```python
SEPARATORS_STANDARD = [
    r"\n\n+",               # 1. paragraph break - strongest boundary
    r"(?=\n\s*\d+\.\s+)",   # 2. before numbered items (zero-width lookahead)
    r"\n",                  # 3. any newline
    r";\s+",                # 4. semicolons - common in long legal enumerations
    r"\s+",                 # 5. whitespace - absolute last resort
]
```

**SEPARATORS_PARAGRAPH** — used only for OPENAI_PARA:

```python
SEPARATORS_PARAGRAPH = [
    r"(?=\n\s*\d+\.\s+)",   # 1. numbered points - PRIMARY boundary
    r"\n\n+",               # 2. paragraph breaks within a point
    r"\n",                  # 3. single newlines - only if still too long
]
```

For OPENAI_PARA the numbered point separator comes first because each numbered point in a Slovak court decision is one complete legal argument. chunk_size=1500 is just a safety net for extremely long points — in most cases the numbered point separator fires long before hitting 1500 tokens.

Also: **overlap=0** for OPENAI_PARA. Overlap is a workaround for artificial token-window boundaries. If i added overlap here i would be copying the end of paragraph 3 into paragraph 4 — but those are separate legal arguments. Natural boundaries don't need overlap.

---

## 5. Metadata stored per chunk

Every chunk gets these metadata fields injected so i have full traceability for analysis and window expansion:

| Field | What it is |
|-------|-----------|
| `source_file` | original PDF filename |
| `chunk_index` | position of this chunk in the document (critical for window expansion ±1) |
| `strategy` | which model/tokenizer was used |
| `char_len` | character count |
| `token_len` | actual token count from the real tokenizer (useful for debugging and thesis stats) |