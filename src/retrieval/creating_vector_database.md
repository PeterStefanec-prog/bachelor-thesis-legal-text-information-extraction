# Vector Store – Notes

> **File:** `src/retrieval/vector_store.py`

---

## Where this fits in the pipeline

```
01_raw_pdfs  →  02_processed_json  →  03_chunked_docs  →  04_vectorstore  →  retrieval
                (data_cleaner.py)     (chunk_processor)    (this script)
```

This script takes chunks from step 3 and turns them into a searchable vector database. Every chunk gets converted into a vector (a list of ~384 numbers) that represents its meaning. Later when i search with a query, i convert the query into a vector too and find the chunks whose vectors are closest to it.

---

## Multiple collections

For the thesis experiments i run this script multiple times — once per chunking strategy. Each strategy gets its own separate ChromaDB collection so the results don't mix:

| CHUNK_SUFFIX | COLLECTION_NAME | Model |
|---|---|---|
| ME5_380 | legal_decisions_me5_380 | mE5-small |
| ME5_200 | legal_decisions_me5_200 | mE5-small |
| OPENAI_500 | legal_decisions_openai_500 | OpenAI (later) |
| OPENAI_200 | legal_decisions_openai_200 | OpenAI (later) |
| OPENAI_PARA | legal_decisions_openai_para | OpenAI (later) |

The script reads `CHUNK_SUFFIX` and `COLLECTION_NAME` from environment variables so i don't have to edit the file each time:

```bash
CHUNK_SUFFIX=ME5_200 COLLECTION_NAME=legal_decisions_me5_200 python src/retrieval/vector_store.py
```

If i run without env variables it uses the defaults (ME5_380).

---

## What ChromaDB is

ChromaDB is a vector database. Basically a normal database but instead of searching by exact text match, it searches by vector similarity. i use `PersistentClient` which means it saves everything to disk so i don't lose my work between runs.

---

## Why i load the model manually

Originally i used `SentenceTransformerEmbeddingFunction` which is built into ChromaDB. Seemed convenient. But it's a black box - i had no way to:
- add the `passage: ` prefix that mE5 needs (more on this below)
- set `normalize_embeddings=True`

So i switched to loading `SentenceTransformer` directly and calling `model.encode()` myself. One more import but full control.

---

## The `passage: ` prefix thing

mE5 is an instruction-tuned model. it was trained to expect a short instruction prefix telling it what it's encoding:

- when indexing documents: `"passage: " + text`
- when searching later: `"query: " + text`

Without the prefix the model doesn't know the difference and embedding quality drops. This is written in the model card on HuggingFace.

The tricky part: i need the prefix only for computing the vector, NOT for storing in the database. if i stored `"passage: Súd rozhodol..."` in the DB, the LLM would later see that prefix in the retrieved chunks which is just noise in the prompt.

So i have two separate lists:

```python
raw_texts = []       # clean text -> stored in DB, LLM reads this
texts_to_embed = []  # "passage: " + text -> only for model.encode(), never stored
```

This way ChromaDB stores the clean text but the embeddings are computed correctly.

---

## `normalize_embeddings=True`

When i encode vectors i always set this to True. It forces every vector to have length exactly 1 (unit vector). This is important because cosine similarity between unit vectors is just a dot product - clean and fast. Without normalization both sides (documents and queries) would need extra division at search time, and more importantly - both sides must be normalized the SAME way. If documents are normalized and queries are not (or vice versa), the similarity scores are wrong.

Note: i also had this bug in `evaluate_retrieval.py` - query vectors were encoded without `normalize_embeddings=True` while document vectors in DB were normalized. Fixed that too.

---

## `hnsw:space=cosine`

This tells ChromaDB to use cosine similarity as the distance metric in its internal search index. The default is L2 (euclidean distance) which is wrong for text embeddings.

Technically: for unit vectors, L2 and cosine give the same ranking of results (mathematically equivalent). But cosine gives scores in range `[-1, 1]` which is much more readable and consistent with how i think about similarity.

Important: this must be set when the collection is CREATED. you cannot change it later. if i already had a collection with the wrong metric i had to delete it and start over.

---

## `upsert()` instead of `add()`

`add()` crashes if a chunk ID already exists in the database. This means re-running the script (which i do all the time during development) would always fail unless i manually deleted the whole DB first.

`upsert()` just overwrites existing entries silently. Much better for iterative work.

---

## Batching

i insert chunks in batches of 100 instead of one giant call. ChromaDB can be slow or run into memory issues with very large single inserts. 100 is a safe and fast batch size.