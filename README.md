# Methods of Information Extraction from Legal Texts

**Single-document RAG for Slovak court decisions on contractual penalties (§301 Obchodného zákonníka)**

Bachelor's thesis at FIIT STU. Pipeline that extracts structured records about
contractual penalties from Slovak court decisions — each field grounded in a
verbatim quote from the source document.

Combines paragraph chunking, hybrid BM25/dense retrieval with weighted RRF,
a domain regex reranker (83 patterns), and two-stage LLM extraction with
five-stage quote validation. Evaluated on 176 decisions / 218 penalties
with GPT-4o, Gemini 2.5 Flash, and Qwen 3.5 (RAG vs full-document).

## Setup

Python 3.11+ (tested on 3.13).

​```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
​```

API keys via environment variables:

​```bash
export OPENAI_API_KEY="sk-..."         # embeddings + GPT-4o
export GEMINI_API_KEY="AIzaSy..."      # Gemini 2.5 Flash + multi-doc query parser
export OPENROUTER_API_KEY="sk-or-..."  # Qwen 3.5
​```

## Quick start

Pre-computed retrieval results and LLM extractions are committed in the repo,
so you can browse everything without re-running the pipeline.

Precedent search demo over 218 penalties (needs `GEMINI_API_KEY`):

​```bash
python src/multi_doc/demo.py
​```

Retrieval grid (268 experiments) and lawyer evaluation results — no API keys
needed, just CSV/xlsx readers:

​```bash
jupyter notebook notebooks/06_retrieval_comparison.ipynb
jupyter notebook notebooks/08_lawyer_evaluation.ipynb
​```

## Project layout

- `src/` — pipeline modules: `preprocessing`, `chunking`, `retrieval`,
  `reranking`, `evaluation_retrieval`, `extraction`, `multi_doc`
- `scripts/` — one-off scrapers for OpenData APIs of Slovak courts
- `data/` — numbered folders by pipeline phase (raw PDFs, processed JSON,
  chunks, ChromaDB, evaluation results, LLM outputs, penalty index)
- `notebooks/` — exploratory analysis and paper figures

`data/04_vectorstore/` is gitignored — rebuild from chunks via
`src/retrieval/vector_store.py` (env vars `MODEL_TYPE`, `CHUNK_SUFFIX`,
`COLLECTION_NAME` — see TECHNICAL_DOCUMENTATION.md §6.4).
