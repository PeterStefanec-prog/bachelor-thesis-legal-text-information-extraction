# INPUT data is vector database data/04_vectorstore and processed jsons from data/02_processed_json
# OUTPUT it creates final chunks and other data of each document - "data/06_retrieval_results"

# cd /Users/stefanec/STU_FIIT/bachelor-thesis-legal-text-information-extraction
# export OPENAI_API_KEY="sk-..."
#
# # 1. Build vector store
'''
MODEL_TYPE=openai \
CHUNK_SUFFIX=OPENAI_PARA \
COLLECTION_NAME=legal_decisions_openai_para \
.venv/bin/python src/retrieval/vector_store.py
'''
#
# # 2. Precompute retrieval pre extraction
# .venv/bin/python src/retrieval/precompute_retrieval.py

# .venv/bin/python src/retrieval/precompute_retrieval.py
# change this for debug:
    # SINGLE_DOC_TEST = "KS_Banská_Bystrica_43CoPv_10_2023_00_dokument.pdf"
    # SINGLE_DOC_TEST = None

"""
Precompute retrieval results for extraction pipeline.

This script runs my winning retrieval config (hybrid_a7_reranked_top7) on every
document in ChromaDB and saves the retrieved chunks to JSON files.
The extraction pipeline then reads these JSONs instead of calling ChromaDB live.
This way the extraction is reproducible - same chunks every time.

Runin also from project root:
    python src/retrieval/precompute_retrieval.py
"""


import os
import sys
import json
import re
import glob
import chromadb
from tqdm import tqdm

# #########################################
# CONFIG - hardcoded winning retrieval config
# #########################################
# this is the exact same config that got 90.5% recall in experiments
# i set env vars BEFORE importing evaluate_retrieval.py because it reads
# them at import time (module-level variables).
# if i set them after import it wont work - the old defaults would already be loaded (lots of time spend looking whats wrong ...)

# ################### HRADCODED WINNING CONFIG (optimal) ######################
os.environ["MODEL_TYPE"]         = "openai"
os.environ["COLLECTION_NAME"]    = "legal_decisions_openai_para"
os.environ["RETRIEVAL_MODE"]     = "hybrid"
os.environ["RRF_ALPHA"]          = "0.7"
os.environ["EXP_TOP_K"]         = "7"
os.environ["EXP_WINDOW"]        = "0"
os.environ["USE_RERANKER"]       = "1"
os.environ["RERANKER_ALPHA"]     = "0.6"
os.environ["RERANKER_OVERSAMPLE"] = "5"
os.environ["RERANKER_DEBUG"]     = "0"
os.environ["CHUNK_MODE"]         = "flat"

# Mac M2 fix - same as in evaluate_retrieval.py - just technical stabilizing setttings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# make sure project root is in path so imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# now import the retrieval functions - they will use the env vars i set above
# NOTE: do_retrieve() and get_doc_total_chunks() take `collection` as parameter,
# so i dont need to import the collection object. i create my own below.
from src.evaluation_retrieval.evaluate_retrieval import (      # do_retrieve - outsource retrieval logic into evaluate_retrieval.py
    do_retrieve, get_doc_total_chunks       # get_doc_total_chunks - just for number of chunks of document
)
from src.retrieval.shared_queries import QUERIES, BM25_QUERIES      # mapping query key - query text
from src.reranking.legal_patterns import CALL_TO_QUERIES   # mapping call1/call2 - query keys

TOP_K = 7
DB_DIR = "data/04_vectorstore"  # where is ChromDB
COLLECTION_NAME = "legal_decisions_openai_para"
OUTPUT_DIR = "data/06_retrieval_results"        # there will precomputed retrieval jsons be saved
PROCESSED_DIR = "data/02_processed_json"    # there are processed json dokcument - i get metadata, header, verdict, resoning from there

##################################################################
##################################################################

# connect to ChromaDB - same settings as vector_store.py and evaluate_retrieval.py
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=DB_DIR)  # opening databse
collection = chroma_client.get_collection(name=COLLECTION_NAME) # opening the collection in the database
print(f"Collection '{COLLECTION_NAME}' loaded.")
# now i can do this do_retrieve(..., collection, ...)

# debug - uncommenting this line will test on single document first:
# SINGLE_DOC_TEST = "KS_Banská_Bystrica_43CoPv_10_2023_00_dokument.pdf"
SINGLE_DOC_TEST = None

os.makedirs(OUTPUT_DIR, exist_ok=True)  # just to be sure that data/06_retrieval_results exists


# ############################################
# EMBEDDING - precompute query vectors once
# ############################################
# i only need to embed the 7 queries once, then reuse for every document.
# this saves API calls (7 total instead of 7 * 176 = 1232)

from openai import OpenAI
openai_client = OpenAI()    # uses OPENAI_API_KEY from env

print("Embedding query vectors (7 queries, one API call)...")
query_texts = list(QUERIES.values())    # texts of queries
query_keys = list(QUERIES.keys())       # keys of those texts

# seinding all 7 queires into openai embedding model (before i send it 7 times and 7 * 176 documents is 1232)
# insted of 1232 embeddings i vectorise 7 (big save)
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=query_texts,
)
# saving vectors in dictionary so i dont have to calculate it again later - map query_key -> embedding vector
precalculated_vectors = {}
for i, key in enumerate(query_keys):
    precalculated_vectors[key] = response.data[i].embedding
# example
# {
#   "q_breach": [...],
#   "q_contract": [...],
#   ...
# }
print(f"Done - {len(precalculated_vectors)} query vectors ready.\n")


# ############################################
# HELPER FUNCTIONS
# ############################################

def parse_chunk_index(chunk_id):
    """Extract chunk index number from chunk_id like 'docname.pdf_chunk_7' -> 7"""
    match = re.search(r'_chunk_(\d+)$', chunk_id)   # finds _chunk_ and one or more digits behind it
    if match:
        return int(match.group(1))
    return -1  # fallback if format is unexpected (as i am implementing whole pipeline - hsould not happen)


def load_document_data(pdf_filename):
    """Loads metadata, header, verdict and reasoning from processed JSON.
    I need all of these for the extraction pipeline:
    - metadata: case_id, court, date for the output JSON header
    - header + verdict: context that always goes to LLM alongside chunks
    - reasoning: for full-document mode where LLM gets  entire text
    """
    clean_name = pdf_filename.replace(".pdf", "").replace(".json", "")  # delete postfix
    pattern = os.path.join(PROCESSED_DIR, f"{clean_name}*.json")
    matches = glob.glob(pattern)
    if not matches:
        print(f"  WARNING: no processed JSON found for {pdf_filename}")
        return None

    with open(matches[0], "r", encoding="utf-8") as f:  # load the json file
        data = json.load(f)

    # returning segments which i need
    return {
        "metadata": data.get("metadata", {}),
        "header": data.get("header", ""),
        "verdict": data.get("segments", {}).get("verdict", ""),
        "reasoning": data.get("segments", {}).get("reasoning", ""),
    }


def retrieve_call_chunks(call_queries, pdf_filename, top_k_override=None):
    """Run retrieval for a group of queries (call1 or call2) and deduplicate.

    Each query returns its own top_k chunks. Some chunks will appear in results from multiple queries
    (e.g. a chunk about 'zmluva o dielo' matches both q_breach and q_contract).
    I merge those duplicates so the LLM doesnt read the same text twice.

    NOTE about scores: the reranker uses min-max normalization WITHIN each query's
    candidate pool.
    So a score of 0.98 from q_rate and 0.72 from q_breach are NOT comparable
        - they are normalized in different pools (different candidates, different keyword patterns).
    When i merge duplicates i keep max(scores) but thats just for rough sorting/debugging, not for any real ranking decision.
    The LLM never sees these scores anyway - chunks are ordered by document position (chunk_index).

    Returns list of chunk dicts sorted by chunk_index (document order).
    """
    use_top_k = top_k_override if top_k_override else TOP_K

    # pool of unique chunks - keyed by chunk_id to catch duplicates
    chunk_pool = {}

    for query_key in call_queries:
        # do_retrieve handles everything: hybrid fusion, reranking, etc.
        texts, scores, ids, _ = do_retrieve(
            collection,
            query_key,
            QUERIES[query_key],
            precalculated_vectors.get(query_key),
            pdf_filename,
            use_top_k,
            0,  # window_size = 0 (winning config)
        )

        for chunk_id, text, score in zip(ids, texts, scores):
            if chunk_id in chunk_pool:
                # same chunk found by another query - keep max score, track all queries
                chunk_pool[chunk_id]["score"] = max(chunk_pool[chunk_id]["score"], score)
                # just for me for debug - which query hit that chunk?
                if query_key not in chunk_pool[chunk_id]["source_queries"]:
                    chunk_pool[chunk_id]["source_queries"].append(query_key)
            else: # chunk is not in pool - add it
                chunk_pool[chunk_id] = {
                    "chunk_id": chunk_id,
                    "chunk_index": parse_chunk_index(chunk_id),
                    "text": text,
                    "score": round(score, 6),
                    "source_queries": [query_key],
                }

    # sort by chunk_index so LLM reads chunks in document order
    # this is important because legal arguments build on each other
    sorted_chunks = sorted(chunk_pool.values(), key=lambda c: c["chunk_index"])
    return sorted_chunks


# ############################################
# MAIN LOOP
# ############################################

# get all unique documents from ChromaDB
print("Getting document list from ChromaDB...")
all_metadata = collection.get(include=["metadatas"])

source_files = []
for metadata in all_metadata["metadatas"]:
    source_files.append(metadata["source_file"])
all_source_files = sorted(set(source_files))

# if test mode activated - it process only one file, otherwise all of them
if SINGLE_DOC_TEST:
    all_source_files = [SINGLE_DOC_TEST]
    print(f"TEST MODE: processing only {SINGLE_DOC_TEST}\n")
else:
    print(f"Found {len(all_source_files)} documents in ChromaDB.\n")

results_summary = []    # how much chunks does document have, how much in call1 and how much in call2

for pdf_filename in tqdm(all_source_files, desc="Precomputing retrieval"):
    # output path - use doc name without .pdf extension
    doc_stem = pdf_filename.replace(".pdf", "")
    output_path = os.path.join(OUTPUT_DIR, f"{doc_stem}.json")

    # skip if already computed (
    if os.path.exists(output_path) and not SINGLE_DOC_TEST:
        continue

    # how many chunks does this doc have? needed for adaptive top_k
    total_doc_chunks = get_doc_total_chunks(collection, pdf_filename)

    # adaptive top_k for call1 - same logic as evaluate_retrieval.py line 1074
    # larger documents (>30 chunks) need more chunks to find contract context
    # because the factual background is spread across more numbered points
    #
    # FIX: raised the cap from TOP_K+3 to TOP_K+5 and changed divisor from 20 to 15.
    # Before this fix, an 86-chunk doc (eval_09, 43Cob/75/2024) got only top_k=9, with oversample pool covering just 52% of the document.
    # After fix, it gets top_k=10 and the oversample covers ~58%. For 120-chunk docs the cap goes from 10 to 12.
    # This is still well below 60% coverage, so single-document RAG retains its selectivity advantage over full-doc.
    if total_doc_chunks > 30:
        # Add 1 extra retrieved chunk for every 15 chunks above 30. But Do not increase TOP_K by more than 5.
        call1_top_k = min(TOP_K + 5, TOP_K + (total_doc_chunks - 30) // 15)
    else:
        call1_top_k = TOP_K

    # ##### CALL 1: contract facts + penalty definition #####
    # q_breach and q_contract get adaptive top_k (more chunks for big docs)
    # q_rate and q_principal use standard top_k
    call1_adaptive_queries = ["q_breach", "q_contract"]
    call1_standard_queries = ["q_rate", "q_principal"]

    call1_chunks_adaptive = retrieve_call_chunks(call1_adaptive_queries, pdf_filename, call1_top_k) # this get adaptive top_k
    call1_chunks_standard = retrieve_call_chunks(call1_standard_queries, pdf_filename, TOP_K)   # basic TOP_K

    # merge the two call1 groups (dedup again in case same chunk appeared in both)
    call1_pool = {}
    for chunk in call1_chunks_adaptive + call1_chunks_standard: # deduplication again
        cid = chunk["chunk_id"]
        if cid in call1_pool:   # save higher score
            call1_pool[cid]["score"] = max(call1_pool[cid]["score"], chunk["score"])
            for q in chunk["source_queries"]:       # adding queries which hit the chunnk
                if q not in call1_pool[cid]["source_queries"]:
                    call1_pool[cid]["source_queries"].append(q)
        else:
            call1_pool[cid] = chunk.copy()  # safely copy dict

    call1_chunks = sorted(call1_pool.values(), key=lambda c: c["chunk_index"])  # sorted based on doc

    # ### CALL 2: moderation analysis ###
    call2_queries = CALL_TO_QUERIES["call2"]
    # FIX: call2 now also uses adaptive top_k for larger documents.
    # Before this fix, call2 always used flat TOP_K=7, even for a 64-chunk NS SR decision
    # where the moderation analysis spans chunks 25-55.
    # With flat 7, only small fraction of the legal reasoning was retrieved for call2.
    # I use a slightly less agressive formula than call1 (divisor 20 instead
    # of 15) because call2 queries (q_outcome, q_reasoning, q_factors) are more
    # focused and dont need as many chunks as the broader call1 queries.
    if total_doc_chunks > 30:
        call2_top_k = min(TOP_K + 3, TOP_K + (total_doc_chunks - 30) // 20)
    else:
        call2_top_k = TOP_K
    call2_chunks = retrieve_call_chunks(call2_queries, pdf_filename, call2_top_k)

    # ### LOAD DOCUMENT DATA ###
    doc_data = load_document_data(pdf_filename)
    if doc_data is None:
        print(f"  Skipping {pdf_filename} - no processed JSON")
        continue

    # ### BUILD OUTPUT - what i will next use in prompts for llm ###
    output = {
        "source_file": pdf_filename,
        "case_id": doc_data["metadata"].get("case_id", ""),
        "retrieval_config": {
            "name": "hybrid_a7_reranked_top7",
            "retrieval_mode": "hybrid",
            "rrf_alpha": 0.7,
            "base_top_k": TOP_K,
            "window_size": 0,
            "reranker": True,
            "reranker_alpha": 0.6,
            "reranker_oversample": 5,
            "adaptive_top_k_call1": call1_top_k,
            "adaptive_top_k_call2": call2_top_k,
        },
        "document_metadata": doc_data["metadata"],
        "header": doc_data["header"],
        "verdict": doc_data["verdict"],
        "reasoning_full": doc_data["reasoning"],
        "call1_chunks": call1_chunks,
        "call2_chunks": call2_chunks,
        "stats": {
            "total_doc_chunks": total_doc_chunks,
            "call1_unique_chunks": len(call1_chunks),
            "call2_unique_chunks": len(call2_chunks),
            "call1_total_chars": sum(len(c["text"]) for c in call1_chunks),
            "call2_total_chars": sum(len(c["text"]) for c in call2_chunks),
        },
    }

    # save output of each decision in json file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    results_summary.append({
        "doc": pdf_filename,
        "chunks": total_doc_chunks,
        "call1": len(call1_chunks),
        "call2": len(call2_chunks),
        "call1_top_k": call1_top_k,
    })


# ##########################################
# SUMMARY
# ########################################
print(f"\n{'='*60}")
print(f"PRECOMPUTE DONE — {len(results_summary)} documents processed")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*60}")

if results_summary:
    avg_call1 = sum(r["call1"] for r in results_summary) / len(results_summary)
    avg_call2 = sum(r["call2"] for r in results_summary) / len(results_summary)
    print(f"Avg call1 chunks: {avg_call1:.1f}")
    print(f"Avg call2 chunks: {avg_call2:.1f}")

    # showing few examples just for me
    print(f"\nFirst 5 documents:")
    for r in results_summary[:5]:
        print(f"  {r['doc'][:50]:50s}  chunks={r['chunks']:3d}  call1={r['call1']:2d}  call2={r['call2']:2d}  call1_top_k={r['call1_top_k']}")
