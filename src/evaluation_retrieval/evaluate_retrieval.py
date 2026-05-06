# INPUT - data/03_chunked_docs
# INPUT data/04_vectorstore
# INPUT data/05_retrieval_evaluation/golden_dataset_template.csv
#
# NOTE on file organization:
# this file holds two different things mixed together:
#
#   1) RETRIEVAL ALGORITHMS  (~600 lines, the bm25 / dense / hybrid / hier
#      functions, plus do_retrieve as the main entry point).
#      these would normally belong in src/retrieval/algorithms.py
#      - they are even imported from here by src/retrieval/precompute_retrieval.py.
#      They ended up in this file because i wrote them while building the evaluation harness
#      and never split them out.
#
#   2) EVALUATION LOGIC  (find_hits, evaluate_query_group, run_hier_evaluation, main).
#       this is what this file is supposed to contain - it compares
#       retrieved chunks against the golden dataset and writes the experiment results CSV.
#
# i added clear section markers below to make the split visible.
# proper refactor (move retrieval out into src/retrieval/) is left as future
# work to avoid breaking pipeline behaviour right before submission.

import os # env variables, paths to files , new directories
import csv  # loading golden dataset, saing results of experiments
import json  # FIX: needed for json.dumps() when saving lists to CSV - see detail_rows below (dict or list convert to string so can be saved into one cell csv)
import chromadb
import datetime
import re
import sys
from colorama import Fore, Style, init      # colorful output in terminal

# shared utilities - extracted to avoid code duplication between flat and hier modes
# Add project root to sys.path so local imports work when this script is run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # go to 2 folders upper
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.shared_queries import QUERIES, BM25_QUERIES  # dictionary with natural queries for dense retrieval and optimalized querie sfor BMT (keywords)
from src.retrieval.shared_utils import clean_text, safe_avg, tokenize_slovak    # tokenize slovak - for BM25 - lemmatization

# these are set by experiment_runner.py via env variables so i dont have to edit this file
# for each experiment.
# if running manually, just change the defaults here
EXPERIMENT_NAME = os.environ.get("EXP_NAME",        "mE5_chunk380_top5_window1")
TOP_K           = int(os.environ.get("EXP_TOP_K",   "5"))   # how much results for one query?
WINDOW_SIZE     = int(os.environ.get("EXP_WINDOW",  "1"))   # how much neighbours?
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "legal_decisions_me5_380")
MODEL_TYPE      = os.environ.get("MODEL_TYPE",      "me5")  # "me5" or "openai"

# ###################################
# CHUNK_MODE: "flat" (default) or "hier" (hierarchical child-to-parent)
# ###################################
# flat: original behavior - retrieve chunks directly, use window expansion
# hier: retrieve small children (~220 tokens), group by parent, return complete parents to the LLM
#          Better for coverage because the LLM gets full legal arguments instead of chopped-up pieces.
CHUNK_MODE = os.environ.get("CHUNK_MODE", "flat")

# hierarchical-specific config (only used when CHUNK_MODE=hier)
FETCH_K_PER_QUERY = int(os.environ.get("FETCH_K_PER_QUERY", "8"))   # how much children for one query?
CALL1_PARENTS = int(os.environ.get("CALL1_PARENTS", "5"))           # how much parents will LLM call 1 get?
CALL2_PARENTS = int(os.environ.get("CALL2_PARENTS", "6"))           # how much parents will LLM call 2 get?

# ###################################
# RETRIEVAL MODE: "dense" (default), "bm25", or "hybrid" (dense + BM25 with RRF)
# - dense:  cosine similarity via embedding model (original behavior)
# - bm25:   keyword matching via BM25 (no embeddings needed)
# - hybrid: both dense + BM25 combined using Reciprocal Rank Fusion (RRF)
# ###################################
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "dense")
RRF_K = int(os.environ.get("RRF_K", "60"))  # RRF constant, standard value = 60

# ###################################
# WEIGHTED RRF: how much dense vs BM25 matters in hybrid
# alpha=0.5 = equal weight (original RRF)
# alpha=0.7 = dense dominates, BM25 is a light boost
# alpha=1.0 = pure dense (BM25 ignored)
#
# Why not 0.5? BM25 with paragraph chunks pulls in noisy chunks (procedural text full of legal keywords) that push out good dense results.
# Dense is better at understanding semantic relevance, but misses introductory chunks (chunk_0) where facts are stated as context, not as main topic.
# alpha=0.7 lets BM25 boost chunks with keyword matches without overriding dense.
# ###################################
RRF_ALPHA = float(os.environ.get("RRF_ALPHA", "0.5"))  # default 0.5 = standard equal-weight RRF

# ###################################
# RERANKER: keyword-based reranking after retrieval
# ###################################
# When turned on, retrieval fetches MORE chunks than usual (e.g. 10 instead of 5) and then the reranker re-scores them using legal keyword patterns and picks  best top_k.
# The idea is that retrieval sometimes puts procedural chunks above informative ones because they are semantically similar, and keyword matching
# can fix that.
#
# USE_RERANKER=1 to turn it on, =0 to keep original behavior (default off)
# RERANKER_ALPHA controls balance: 0.6 means retrieval is more important than keywords
# RERANKER_OVERSAMPLE: how many times more chunks to fetch (2 = fetch 2x, rerank to top_k)
# RERANKER_DEBUG=1 to print detailed scoring info for each chunk
# ==========================================
USE_RERANKER = os.environ.get("USE_RERANKER", "0") == "1"
RERANKER_ALPHA = float(os.environ.get("RERANKER_ALPHA", "0.6"))
# oversample=5 means we fetch 5*top_k candidates and rerank to top_k.
# i tested 3x and 5x — with 5x the reranker sees more chunks from large
# documents (50-84 chunks) and has better chance of finding the relevant
# ones that rank low in pure dense/hybrid similarity.
RERANKER_OVERSAMPLE = int(os.environ.get("RERANKER_OVERSAMPLE", "5"))
RERANKER_DEBUG = os.environ.get("RERANKER_DEBUG", "0") == "1"

if USE_RERANKER:
    from src.reranking.reranker import rerank_chunks       # import only when needed

# ###################################
# FIX: Mac M2 freezing problem!
# These lines stop my Mac M2 from freezing. It turns off parallel processing and threads.
# I also disable Chroma telemetry so it doesn't hang my script randomly.
# ###################################
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# MODEL IMPORTS: loaded conditionally based on MODEL_TYPE to avoid unnecessary dependencies
if RETRIEVAL_MODE in ("dense", "hybrid"):
    if MODEL_TYPE == "me5":
        from sentence_transformers import SentenceTransformer
    elif MODEL_TYPE in ("openai", "openai_large"):
        from openai import OpenAI

if RETRIEVAL_MODE in ("bm25", "hybrid"):
    from rank_bm25 import BM25Okapi
    import simplemma


# ###################################
# STEP 1: Setup paths and basic variables
# ###################################
if CHUNK_MODE == "hier":
    DB_DIR = "data/04_vectorstore_hier" # where is chromaDB
else:
    DB_DIR = "data/04_vectorstore"

EVAL_DIR = "data/05_retrieval_evaluation"       # evaluation saving here
CSV_PATH = os.path.join(EVAL_DIR, "golden_dataset_template.csv")    # here is golden dataset

# same as DB_DIR above - make sure folder exists before we try to write CSVs into it without this the script would crash at  very end with FileNotFoundError after
# doing all the work (already computed everything, just cant save it - very annoying)
os.makedirs(EVAL_DIR, exist_ok=True)

# Two output CSVs:
# - SUMMARY: one row per experiment run (for comparing experiments at high level)
# - DETAILS: one row per question per document (for deep analysis, cosine scores, thesis charts)
RESULTS_SUMMARY_CSV = os.path.join(EVAL_DIR, "experiment_results_summary.csv")
RESULTS_DETAILS_CSV = os.path.join(EVAL_DIR, "experiment_results_details.csv")

# ###################################
# STEP 2: Define  queries
# ###################################
# I now use 4 queries instead of 3. The old q3 was too broad - it asked about
# both the OUTCOME (was penalty reduced?) and the REASONING (why?) in one query.
# This was bad for cases where par.301 was NOT applied -  dense model would retrieve
# chunks about the outcome (neplatna zmluva, vratene konanie) but miss the reasoning.
#
# New split:
#   q_context  -> maps 1:1 to CSV column q1_context_quotes
#   q_penalty  -> maps 1:1 to CSV column q2_penalty_quotes
#   q_outcome  -> BOTH map to q3_moderation_quotes together (see Step 5.2 below)
#   q_reasoning -> (evaluated as a combined pool with q_outcome)
#
# For evaluation: q_outcome and q_reasoning run separately, their super_chunks are
# MERGED into one pool, and we check if q3 golden quotes are found anywhere in it.
# This is FAIRER - two focused queries together cover what one blurry query couldn't.

# NOTE: QUERIES, BM25_QUERIES, QUERY_TO_CSV_COLUMN are imported from shared_queries.py
# NOTE: clean_text, safe_avg, tokenize_slovak, SLOVAK_STOPWORDS are imported from shared_utils.py


# ###################################
# BM25 SUPPORT
# ###################################
# BM25 tokenizer for Slovak legal text. Key design decisions:
#
# 1. PRESERVE NUMBERS & PERCENTAGES: "0,05%" stays as one token, not ["0", "05"]
#    Legal texts are full of specific amounts (15.234,60 EUR) and rates (0,05% ročne)
#    that are critical for matching q_rate and q_principal queries.
#
# 2. PRESERVE LEGAL REFERENCES: "§301" as one token (join § with following number)
#    Dense retrieval doesn't understand "§ 301" semantically, but BM25 can match it exactly.
#
# 3. LEMMATIZATION via simplemma: "pokuty" → "pokuta", "záväzku" → "záväzok"
#    Slovak has rich inflection (6 cases × 2 numbers = 12+ forms per noun).
#    Without lemmatization, BM25 misses obvious matches.
#
# 4. STOPWORD REMOVAL: Drop common words that add noise ("je", "sa", "na", "že", "ako")
#    Legal text is full of procedural language that dilutes keyword signal.
# ###################################

# #################################################################
# ====== RETRIEVAL ALGORITHMS START HERE ==========================
# #################################################################
# everything from here until "EVALUATION ORCHESTRATORS" section
# is retrieval logic that conceptually belongs in src/retrieval/.
#
# group 1 - flat chunks:
#   get_bm25_index, retrieve_bm25, retrieve_hybrid_rrf,
#   retrieve_super_chunks, do_retrieve,
#   get_doc_total_chunks, get_doc_all_texts, get_token_coverage
#
# group 2 - hierarchical (parent-child):
#   _load_parents, _get_all_parent_texts, retrieve_hier_call
#
# the only EVAL function in this block is find_hits() - kept here
# because it is called right after retrieval. marked with [EVAL] tag.
# #################################################################

# Slovak legal stopwords - common function words that appear in nearly every chunk and carry no retrieval signal.
# Kept short to avoid removing anything useful.
# Cache BM25 index per document to avoid rebuilding for each query (do not want to build repeatedly for each query)
_bm25_cache = {}  # {pdf_filename: (bm25_index, chunk_ids, chunk_texts, chunk_metadatas)}

def get_bm25_index(collection, pdf_filename):
    """Build (or retrieve cached) BM25 index for all chunks of one document."""
    if pdf_filename in _bm25_cache:
        return _bm25_cache[pdf_filename]

    # Get ALL chunks for this document from ChromaDB
    result = collection.get(
        where={"source_file": pdf_filename},
        include=["documents", "metadatas"]
    )

    # Sort by chunk_index to ensure consistent ordering
    paired = list(zip(result['ids'], result['documents'], result['metadatas']))     # tuple (id, document_text, metadata)
    # important to have chunks in right order
    if CHUNK_MODE == "hier":
        paired.sort(key=lambda x: (x[2].get('parent_index', 0), x[2].get('child_index', 0)))
    else:
        paired.sort(key=lambda x: x[2]['chunk_index'])

    # dividing back into lists
    chunk_ids = [p[0] for p in paired]
    chunk_texts = [p[1] for p in paired]
    chunk_metadatas = [p[2] for p in paired]

    # Build BM25 index from tokenized chunks
    # k1=1.2: faster saturation (legal chunks are short, term appearing 2-3x is enough signal)  - when word is appearing 10 times, i do not want it to be 10x better
    # b=0.4: reduced length normalization (paragraph chunks are similar in length) - lower b means that less punish longer chunks
    # for hier mode b=0.5 because children can vary more in length
    b_val = 0.5 if CHUNK_MODE == "hier" else 0.4

    tokenized_corpus = [tokenize_slovak(text) for text in chunk_texts]  # each chunks is changed to tokens for BM25
    bm25_index = BM25Okapi(tokenized_corpus, k1=1.2, b=b_val)

    _bm25_cache[pdf_filename] = (bm25_index, chunk_ids, chunk_texts, chunk_metadatas)
    return _bm25_cache[pdf_filename]


##### MAIN FUNCTION FOR BM25 #######
def retrieve_bm25(collection, query_text, pdf_filename, top_k, window_size):
    """BM25 retrieval - returns same format as retrieve_super_chunks for compatibility.
    Returns (super_chunks, scores, retrieved_ids, fetched_chunk_ids).
    """
    # get bm25 index
    bm25_index, all_ids, all_texts, all_metas = get_bm25_index(collection, pdf_filename)

    tokenized_query = tokenize_slovak(query_text) # tokenizing query
    bm25_scores = bm25_index.get_scores(tokenized_query) # calculate bm25 score for each chunk

    # Get top_k indices by BM25 score (descending)
    ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    super_chunks = []
    scores = []
    ret_ids = []
    fetched_chunk_ids = set()

    for idx in ranked_indices:
        score = round(float(bm25_scores[idx]), 4)

        if CHUNK_MODE == "hier":
            # no window expansion for hierarchical children
            super_chunks.append(clean_text(all_texts[idx]))
            fetched_chunk_ids.add(all_ids[idx])
        else:
            c_idx = all_metas[idx]['chunk_index']
            # Window expansion - same logic as dense retrieval
            window_ids = [
                f"{pdf_filename}_chunk_{c_idx + offset}"
                for offset in range(-window_size, window_size + 1)
            ]       # if winodow is 1 and found chunk_10 then : chunk_9, chunk_10, chunk_11

            db_fetch = collection.get(ids=window_ids)   # loading those chunks from ChromaDB
            fetched_chunk_ids.update(db_fetch['ids'])   # uunique ID chunks

            if db_fetch['documents']:
                paired = list(zip(db_fetch['metadatas'], db_fetch['documents']))    # creating pairs like (metadata_chunk_9, text_chunk_9),
                paired.sort(key=lambda x: x[0]['chunk_index'])  # chunk_9 -> chunk_10 -> chunk_11
                ordered_texts = [doc for _, doc in paired]  # same as paired but saving just texts
                combined_text = " ".join(ordered_texts) # super chunk text_chunk_9 + text_chunk_10 + text_chunk_11
                super_chunks.append(clean_text(combined_text))  # normalize it
            else:
                super_chunks.append("")

        scores.append(score)    # add score and id found chunk
        ret_ids.append(all_ids[idx])   # central retrieved chunk (without window)

    return super_chunks, scores, ret_ids, fetched_chunk_ids     # super_chunks (final texts)


def retrieve_hybrid_rrf(collection, query_vector, query_text, pdf_filename, top_k, window_size, rrf_k=60):
    """Hybrid retrieval: Dense + BM25 combined with Weighted Reciprocal Rank Fusion (RRF).

    Weighted RRF formula:
        score(d) = alpha/(k + rank_dense(d)) + (1-alpha)/(k + rank_bm25(d))

    alpha=0.5 = standard equal-weight RRF (original formula from Cormack et al.)
    alpha=0.7 = dense-dominant: dense signal has 70% weight, BM25 acts as light boost
    alpha=1.0 = pure dense (BM25 contribution is zero)

    Why weighted? For single-document RAG with paragraph chunks:
    - Dense understands semantic relevance (what the chunk is ABOUT)
    - BM25 finds keyword matches (chunks that MENTION specific terms)
    - Equal weight causes BM25 noise to displace good dense results
    - Dense-dominant keeps semantic precision while BM25 boosts keyword-rich chunks

    Returns same format as retrieve_super_chunks
    """
    alpha = RRF_ALPHA  # read from env/global

    # ### Dense ranking ###
    results = collection.query(
        query_embeddings=query_vector,
        n_results=min(top_k * 3, 20),  # get more candidates for fusion
        where={"source_file": pdf_filename},    # single document retrieval
        include=["documents", "metadatas", "distances"]
    )
    dense_ids = results['ids'][0] if results['ids'] else []
    dense_ranks = {doc_id: rank for rank, doc_id in enumerate(dense_ids)}   # for each dense result - save his rank

    # ### do BM25 ranking the same ###
    bm25_index, all_ids, all_texts, all_metas = get_bm25_index(collection, pdf_filename)
    tokenized_query = tokenize_slovak(query_text)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_id_ranks = {
        all_ids[idx]: rank for rank,
        idx in enumerate(bm25_ranked_indices[:top_k * 3])
    }

    # ### Weighted RRF fusion ###
    all_candidate_ids = set(dense_ranks.keys()) | set(bm25_id_ranks.keys()) # union of results of Bm25 and dense
    rrf_scores = {}
    for doc_id in all_candidate_ids:
        score = 0.0
        if doc_id in dense_ranks:
            score += alpha / (rrf_k + dense_ranks[doc_id])
        if doc_id in bm25_id_ranks:
            score += (1.0 - alpha) / (rrf_k + bm25_id_ranks[doc_id])
        rrf_scores[doc_id] = score

    # Sort by RRF score and take top_k
    top_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    # ### Build super_chunks with window expansion ###
    super_chunks = []
    scores = []
    ret_ids = []
    fetched_chunk_ids = set()

    if CHUNK_MODE == "hier":
        # hierarchical: no window, just return the children directly
        id_to_text = dict(zip(all_ids, all_texts))
        for doc_id in top_ids:
            super_chunks.append(clean_text(id_to_text.get(doc_id, "")))
            scores.append(round(rrf_scores[doc_id], 6))
            ret_ids.append(doc_id)
            fetched_chunk_ids.add(doc_id)
    else:
        # flat: window expansion as before
        # Need metadata for chunk_index lookup
        if top_ids:
            meta_fetch = collection.get(ids=top_ids, include=["metadatas"])
            id_to_meta = dict(zip(meta_fetch['ids'], meta_fetch['metadatas']))
        else:
            id_to_meta = {}

        for doc_id in top_ids:
            meta = id_to_meta.get(doc_id, {})
            c_idx = meta.get('chunk_index', 0)

            window_ids = [
                f"{pdf_filename}_chunk_{c_idx + offset}"
                for offset in range(-window_size, window_size + 1)
            ]

            db_fetch = collection.get(ids=window_ids)
            fetched_chunk_ids.update(db_fetch['ids'])

            if db_fetch['documents']:
                paired = list(zip(db_fetch['metadatas'], db_fetch['documents']))
                paired.sort(key=lambda x: x[0]['chunk_index'])
                ordered_texts = [doc for _, doc in paired]
                combined_text = " ".join(ordered_texts)
                super_chunks.append(clean_text(combined_text))
            else:
                super_chunks.append("")

            scores.append(round(rrf_scores[doc_id], 6))
            ret_ids.append(doc_id)

    return super_chunks, scores, ret_ids, fetched_chunk_ids


# #### DENSE RETREIVAL #####
def retrieve_super_chunks(collection, query_vector, pdf_filename, top_k, window_size):
    """Run one query, return (super_chunks, cosine_sims, retrieved_ids, fetched_chunk_ids).
    - find top-k chunkov in one document based  - cosine similarity
    - fetched_chunk_ids = set of ALL chunk IDs actually read from DB (TOP_K + window neighbors).
    I need this set to compute coverage - how much of the doc did this query actually look at?
    - so return super_chunks + score + id + coverage info
    """
    results = collection.query(     # chroma with cosine returns distance
        query_embeddings=query_vector,
        n_results=top_k,
        where={"source_file": pdf_filename},
        include=["documents", "metadatas", "distances"]
    )

    raw_distances = results['distances'][0] if results['distances'] else []     # just for the query
    cosine_sims = [round(1 - d, 4) for d in raw_distances]  # cosine similarities = 1 - distance = similarity e.g. 0.82
    ret_ids = results['ids'][0] if results['ids'] else []   # ids of found chunks

    super_chunks = []
    fetched_chunk_ids = set()  # track every chunk we actually read (TOP_K + their neighbors)

    if len(results['ids']) > 0 and len(results['ids'][0]) > 0:
        retrieved_metadatas = results['metadatas'][0]

        for i in range(len(ret_ids)):
            if CHUNK_MODE == "hier":
                # no window expansion for hierarchical children - just return the child text
                super_chunks.append(clean_text(results['documents'][0][i]))
                fetched_chunk_ids.add(ret_ids[i])
            else:
                c_idx = retrieved_metadatas[i]['chunk_index']

                # Build IDs for surrounding chunks based on WINDOW_SIZE
                # (Chroma smartly ignores IDs that don't exist, e.g., chunk_-1 for first chunk)
                window_ids = [
                    f"{pdf_filename}_chunk_{c_idx + offset}"
                    for offset in range(-window_size, window_size + 1)
                ]

                db_fetch = collection.get(ids=window_ids)
                fetched_chunk_ids.update(db_fetch['ids'])  # add all actually returned IDs

                # FIX: ChromaDB .get() does NOT guarantee order of returned documents!
                # If I just join them as-is, chunk 6 text might appear before chunk 5 text.
                # I must sort by chunk_index first so the super_chunk reads in correct order.
                if db_fetch['documents']:
                    paired = list(zip(db_fetch['metadatas'], db_fetch['documents']))
                    paired.sort(key=lambda x: x[0]['chunk_index'])
                    ordered_texts = [doc for _, doc in paired]
                    combined_text = " ".join(ordered_texts)
                    super_chunks.append(clean_text(combined_text))
                else:
                    # keep index alignment even if window fetch returned nothing
                    super_chunks.append("")

    return super_chunks, cosine_sims, ret_ids, fetched_chunk_ids


def find_hits(super_chunks, cosine_sims, quotes_to_find):
    """[EVAL] Check which quotes are found in super_chunks.

    Returns (quotes_found_count, hit_cosine_scores, miss_cosine_scores,
             chunks_that_had_hits, reciprocal_rank, precision_at_k).

    reciprocal_rank (MRR component): 1/rank of the first chunk that contains
        any golden quote.
        Measures how high the first relevant result appears.
        E.g. if the first hit is at position 3, RR = 1/3 = 0.333.

    precision_at_k: fraction of returned chunks that contain at least one golden quote.
        Measures how many of the retrieved chunks are useful.
        E.g. if 4 out of 10 chunks contain quotes, P@k = 0.4.
    """
    chunks_that_had_hits = set()
    quotes_found_count = 0

    # chunk that had hits
    for quote in quotes_to_find:
        for j, super_chunk in enumerate(super_chunks):
            if quote in super_chunk:    # exact substring match
                chunks_that_had_hits.add(j)
                quotes_found_count += 1     # quotes found
                break


    hit_scores = [sim for j, sim in enumerate(cosine_sims) if j in chunks_that_had_hits]    # cosine scores which was hit
    miss_scores = [sim for j, sim in enumerate(cosine_sims) if j not in chunks_that_had_hits]   # cosine score whis was not hit

    # ### MRR: reciprocal rank of first chunk containing any golden quote ###.- how high is first chunk
    # we sort chunks by score (descending) to get  true rank ordering,
    # because merged sub-query results are just concatenated, not interleaved
    if super_chunks and cosine_sims:
        sorted_indices = sorted(range(len(cosine_sims)), key=lambda i: cosine_sims[i], reverse=True)
        reciprocal_rank = 0.0
        for rank, idx in enumerate(sorted_indices, start=1):
            if idx in chunks_that_had_hits:
                reciprocal_rank = 1.0 / rank
                break
    else:
        reciprocal_rank = 0.0

    # ### Precision@k: how many of the returned chunks are relevant ###
    precision_at_k = len(chunks_that_had_hits) / len(super_chunks) if super_chunks else 0.0
    # from 10 returned chunks are 3 hits so precission is 0.3

    return quotes_found_count, hit_scores, miss_scores, chunks_that_had_hits, reciprocal_rank, precision_at_k


def get_doc_total_chunks(collection, pdf_filename):
    # how many chunks does this document have in total?
    # used as denominator for coverage: fetched_chunks / total_chunks
    result = collection.get(where={"source_file": pdf_filename}, include=["metadatas"])
    return len(result['ids'])


def get_doc_all_texts(collection, pdf_filename):
    """Get all chunk texts for a document. Used for token-based coverage."""
    result = collection.get(
        where={"source_file": pdf_filename},
        include=["documents"]
    )
    # returns dict: chunk_id -> text
    return {cid: doc for cid, doc in zip(result['ids'], result['documents'])}


def get_token_coverage(all_texts, fetched_ids):
    """Compute coverage as characters of fetched chunks / total characters.

    Using character count as proxy for tokens -  ratio is roughly constant for Slovak legal text, so the percentage is the same either way.
    This is fairer than chunk-count because paragraphs vary hugely in length
    (some are 50 tokens, others 1500+).
    """
    total_chars = sum(len(t) for t in all_texts.values())
    if total_chars == 0:
        return 0.0
    fetched_chars = sum(len(all_texts[cid]) for cid in fetched_ids if cid in all_texts)
    return round(fetched_chars / total_chars * 100, 1)


def do_retrieve(collection, query_key, query_text, query_vector, pdf_filename, top_k, window_size):
    """Pick the right retrieval method and optionally rerank the results.

    For BM25 and hybrid modes, uses BM25_QUERIES (keyword-optimized) instead of semantic QUERIES used for dense retrieval.
    This is intentional:
    - Dense needs natural language questions for good embeddings
    - BM25 needs domain keywords for exact token matching

    When  reranker is turned on, this fetches more chunks than needed (2x by default),
    then  reranker re-scores them using legal keywords and picks the best top_k.
    """
    # BM25 uses keyword queries; dense uses semantic queries
    # bm25_query_text = BM25_QUERIES.get(query_key, query_text) if RETRIEVAL_MODE in ("bm25", "hybrid") else query_text
    bm25_query_text = query_text

    if RETRIEVAL_MODE in ("bm25", "hybrid"):
        bm25_query_text = BM25_QUERIES.get(query_key, query_text)

    # when reranker is on, fetch more chunks so it has a bigger pool to choose from
    # e.g. if top_k=5 and oversample=2, we fetch 10 and then rerank down to 5
    how_many_to_fetch = top_k * RERANKER_OVERSAMPLE if USE_RERANKER else top_k

    if RETRIEVAL_MODE == "dense":
        super_chunks, cosine_sims, ret_ids, fetched_ids = retrieve_super_chunks(
            collection, query_vector, pdf_filename, how_many_to_fetch, window_size)
    elif RETRIEVAL_MODE == "bm25":
        super_chunks, cosine_sims, ret_ids, fetched_ids = retrieve_bm25(
            collection, bm25_query_text, pdf_filename, how_many_to_fetch, window_size)
    elif RETRIEVAL_MODE == "hybrid":
        super_chunks, cosine_sims, ret_ids, fetched_ids = retrieve_hybrid_rrf(
            collection, query_vector, bm25_query_text, pdf_filename, how_many_to_fetch, window_size, rrf_k=RRF_K)
    else:
        raise ValueError(f"Unknown RETRIEVAL_MODE: {RETRIEVAL_MODE}")

    # RERANKING STEP
    # the reranker takes the bigger pool and picks the best top_k using keyword patterns.
    # it combines the original retrieval score with a keyword score to decide the final order.
    if USE_RERANKER and super_chunks:
        super_chunks, cosine_sims, ret_ids = rerank_chunks(
            query_key=query_key,
            chunk_texts=super_chunks,
            chunk_scores=cosine_sims,
            chunk_ids=ret_ids,
            top_k=top_k,
            alpha=RERANKER_ALPHA,
            debug=RERANKER_DEBUG,
        )
        # coverage should reflect what LLM actually sees (top_k chunks after reranking),
        # not the full oversampled pool.
        # otherwise coverage would be unfairly inflated compared to baseline experiments without reranker.
        fetched_ids = set(ret_ids)

    return super_chunks, cosine_sims, ret_ids, fetched_ids







#######################################################
# #######################################
# HIERARCHICAL CHILD-TO-PARENT RETRIEVAL
# #######################################
# After flat chunk experiments showed good recall (93-95%) but high coverage ( LLM reads too much of the document),
# i tried a different approach:
#       retrieve CHILDREN (small ~220 token pieces) but give the LLM complete PARENTS
#       (full legal arguments, ~500-1500 tokens each).
#
# Idea: search is precise because children are small and focused on one topic.
# But the LLM gets complete arguments because parents contain the full numbered point.
# This should give similar recall but with better-structured context.
#
# The flow:
# 1. For each query, retrieve top-K children from ChromaDB
# 2. Group retrieved children by parent_id
# 3. Score each parent by its best child's retrieval score + multi-query bonus
# 4. Select top N parents
# 5. Load parent texts from disk and build the context for the LLM

# caches for hier mode
_parent_cache = {}    # {pdf_filename: {parent_id: parent_record}}


def _load_parents(pdf_filename):
    """Load parent chunks from disk JSON.
    Parents are NOT in ChromaDB - they live as JSON because i only need them after i already know which parents to select."""
    if pdf_filename in _parent_cache:
        return _parent_cache[pdf_filename]

    clean_name = pdf_filename.replace(".pdf", "").replace(".json", "")
    path = os.path.join("data/03_chunked_docs_hier", f"{clean_name}_parents.json")
    with open(path, "r", encoding="utf-8") as f:
        parents = json.load(f)

    _parent_cache[pdf_filename] = {p["metadata"]["parent_id"]: p for p in parents}
    return _parent_cache[pdf_filename]


def _get_all_parent_texts(pdf_filename):
    """Load all parent texts for a document. Used for token-based coverage."""
    parent_lookup = _load_parents(pdf_filename)
    return {pid: p["page_content"] for pid, p in parent_lookup.items()}


######### Most important hier function ###########
def retrieve_hier_call(collection, call_queries, pdf_filename, fetch_k, max_parents, precalculated_vectors):
    """Run hierarchical retrieval for one LLM call (e.g. call1 = breach+rate+principal).

    For each query:
    1. Retrieve top-K children (dense/bm25/hybrid - same modes as flat)
    2. Collect all unique children and their scores

    Then:
    3. Group children by parent_id
    4. Score parents: best child score + bonus for multi-query hits
    5. Select top N parents
    6. Build context text from selected parents

    Returns dict with selected parents, context text, and metadata.
    """
    child_pool = {}  # saving all unique children across all queries

    for query_key in call_queries:  # call1 - q_breach, q_contract, q_rate, q_principal, call2 - q_outcome, q_reasoning, q_factors
        # retrieve children using the same do_retrieve() as flat mode
        # window_size=0 because children dont need window expansion
        child_texts, child_scores, child_ids, _ = do_retrieve(
            collection, query_key, QUERIES[query_key],
            precalculated_vectors.get(query_key),
            pdf_filename, fetch_k, window_size=0,
        )

        # deduplicating
        for i, child_id in enumerate(child_ids):
            if child_id not in child_pool:
                child_pool[child_id] = {
                    "child_id": child_id,
                    "text": child_texts[i],
                    "query_hits": [],
                    "query_scores": {},
                }

            # get parent_id from ChromaDB metadata
            if "parent_id" not in child_pool[child_id]:
                meta_result = collection.get(ids=[child_id], include=["metadatas"])
                if meta_result["metadatas"]:
                    child_pool[child_id]["parent_id"] = meta_result["metadatas"][0].get("parent_id", "unknown")
                else:
                    child_pool[child_id]["parent_id"] = "unknown"

            # saving which query found child - this child lwas found with query q_rate for example
            child_pool[child_id]["query_hits"].append(query_key)
            child_pool[child_id]["query_scores"][query_key] = child_scores[i] if i < len(child_scores) else 0.0

    # ### Score each child: best query score + multi-query bonus ###
    for child in child_pool.values():
        if not child["query_scores"]:   # if child has not score it gets 0.0
            child["aggregate_score"] = 0.0
            continue
        best_score = max(child["query_scores"].values())    # q_rate: 0.81 and q_principal: 0.76    - take best
        # bonus for being found by multiple queries - a child relevant to multiple
        # questions is probably important (e.g. a chunk about both penalty rate and breach)
        multi_bonus = 0.15 * (len(child["query_hits"]) - 1) # if found by one query - bonus = 0, len=2 - bonus = 0.15.    len=3 - bonus=0.3
        child["aggregate_score"] = best_score + multi_bonus #  best score + bonus for more queires found

    # ### Group children by parent and score parents ###
    parent_pool = {}
    for child in child_pool.values():       # going through all children and watching which parent they belong to
        pid = child.get("parent_id", "unknown")
        if pid not in parent_pool:  # if parent not in parent_pool - create it
            parent_pool[pid] = {
                "parent_id": pid,
                "children": [],
                "query_hits": set(),
                "parent_score": 0.0,
            }
        parent_pool[pid]["children"].append(child)  # adding child into parent
        parent_pool[pid]["query_hits"].update(child["query_hits"])  # adding query which hit the child

    for parent in parent_pool.values():
        child_scores = [c["aggregate_score"] for c in parent["children"]]   # for each parent get aggregate score of his childs
        best_child = max(child_scores) if child_scores else 0.0         # parent get score as his best child
        # bonus if this parent was hit by multiple different queries
        multi_query_bonus = 0.12 * (len(parent["query_hits"]) - 1)
        parent["parent_score"] = best_child + multi_query_bonus #

    # ### Select top parents ###
    ranked_parents = sorted(parent_pool.values(), key=lambda x: x["parent_score"], reverse=True)    # sort parents by score
    top_parents = ranked_parents[:max_parents]  # getting max k most relevant parents

    # ### Load parent texts and build context ###
    parent_lookup = _load_parents(pdf_filename) # in chromaDB are only children - whole parents are in json files in data/03_chunked_docs_hier
    selected_parent_ids = []
    selected_child_ids = set()
    context_parts = []

    # sort by parent index so the LLM reads them in document order
    top_parents_with_meta = []
    for item in top_parents:
        pid = item["parent_id"]
        if pid in parent_lookup:
            p_meta = parent_lookup[pid]["metadata"]
            top_parents_with_meta.append((p_meta.get("parent_index", 0), item, parent_lookup[pid]))
    top_parents_with_meta.sort(key=lambda x: x[0])

    for _, item, parent_record in top_parents_with_meta:
        pid = item["parent_id"]
        selected_parent_ids.append(pid)
        selected_child_ids.update(c["child_id"] for c in item["children"])

        meta = parent_record["metadata"]
        # just diagnostic label for example - PARENT 4 | point=5 | score=0.923 | queries=q_rate,q_principal
        label = (
            f"PARENT {meta['parent_index']}"
            f" | point={meta['point_number'] if meta['point_number'] != -1 else 'na'}"
            f" | score={item['parent_score']:.3f}"
            f" | queries={','.join(sorted(item['query_hits']))}"
        )
        context_parts.append(label)
        context_parts.append(parent_record["page_content"])

    context_text = "\n\n".join(context_parts)

    return {
        "selected_parent_ids": selected_parent_ids,
        "selected_child_ids": sorted(selected_child_ids),
        "selected_parents_count": len(selected_parent_ids),
        "candidate_child_count": len(child_pool),
        "context_text": context_text,
    }


# #################################################################
# ====== EVALUATION ORCHESTRATORS START HERE ======================
# #################################################################
# from here on this file is pure evaluation:
#   - run_hier_evaluation: orchestrator for hierarchical mode
#   - evaluate_query_group: per-document call1 / call2 evaluation
#     (defined further below, used by main)
#   - main: orchestrator for flat mode + golden CSV writer
#
# all of these compare retrieved chunks against the golden dataset
# (data/05_retrieval_evaluation/golden_dataset_template.csv) and
# write the per-experiment row into experiment_results_summary.csv.
# #################################################################

# ##################################################
# HIERARCHICAL EVALUATION (runs when CHUNK_MODE=hier)
# ################################################

def run_hier_evaluation():
    """Run hierarchical child-to-parent evaluation.
    Separate function because  flow is different from flat: children -> parents instead of chunks -> super_chunks."""
    from src.reranking.legal_patterns import CALL_TO_QUERIES   # which queries are wich call ?
    from src.retrieval.shared_utils import openai_token_len

    init(autoreset=True)        # just initailization of colors
    print(f"=== STARTING HIER EVALUATION: {EXPERIMENT_NAME} ===")
    print(f"  mode={RETRIEVAL_MODE} | fetch_k={FETCH_K_PER_QUERY} | call1_parents={CALL1_PARENTS} | call2_parents={CALL2_PARENTS}")
    # mode=hybrid | fetch_k=8 | call1_parents=5 | call2_parents=6

    # ### Connect to ChromaDB and load model ###
    precalculated_vectors = {}  # here will be all querie embeddings saved

    if RETRIEVAL_MODE in ("dense", "hybrid"):
        if MODEL_TYPE == "me5":
            print("Loading mE5-base model explicitly (CPU mode)...")
            # loading me5 model
            model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
            def embed_query(text):
                return model.encode([f"query: {text}"], normalize_embeddings=True).tolist() # important prefix - query
        elif MODEL_TYPE in ("openai", "openai_large"):
            # loading openaimodel
            OPENAI_MODEL = "text-embedding-3-large" if MODEL_TYPE == "openai_large" else "text-embedding-3-small"
            print(f"Using OpenAI {OPENAI_MODEL} via API...")
            openai_client = OpenAI()
            def embed_query(text):
                response = openai_client.embeddings.create(model=OPENAI_MODEL, input=[text])    # embeding queires
                return [response.data[0].embedding]
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

        print("Pre-calculating query vectors...")
        for q_key, q_text in QUERIES.items():   # calculating vectors by calling one of the fucntions above
            precalculated_vectors[q_key] = embed_query(q_text)  # later i will use just precalculated_vectors["q_reasoning"]
    else:
        print("BM25 mode - no embedding model needed")

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    total_questions = 0
    successful_hits = 0.0
    detail_rows = []


    print("Reading Golden Dataset...")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=';')

        for row in reader:
            pdf_filename = row["document_name"].replace(".json", ".pdf")
            print(f"\n--- Testing document: {pdf_filename} ---")

            # load all parent texts for coverage calculation
            all_parent_texts = _get_all_parent_texts(pdf_filename)
            total_parents = len(all_parent_texts)

            # retrieve for both LLM calls
            call1 = retrieve_hier_call(
                collection, CALL_TO_QUERIES["call1"], pdf_filename,
                FETCH_K_PER_QUERY, CALL1_PARENTS, precalculated_vectors,
            )
            call2 = retrieve_hier_call(
                collection, CALL_TO_QUERIES["call2"], pdf_filename,
                FETCH_K_PER_QUERY, CALL2_PARENTS, precalculated_vectors,
            )

            # token coverage: what % of document text will LLM see
            call1_cov = get_token_coverage(all_parent_texts, call1["selected_parent_ids"])
            call2_cov = get_token_coverage(all_parent_texts, call2["selected_parent_ids"])
            all_selected = set(call1["selected_parent_ids"]) | set(call2["selected_parent_ids"])
            total_cov = get_token_coverage(all_parent_texts, all_selected)

            # evaluate golden quotes against context text
            checks = [
                ("q1_context_quotes", "q1_context", call1, call1_cov),
                ("q2_penalty_quotes", "q2_combined", call1, call1_cov),
                ("q3_moderation_quotes", "q3_combined", call2, call2_cov),
            ]

            for csv_col, query_key, call_result, coverage_pct in checks:
                correct_answer_string = row[csv_col].strip()    # read golden quote string
                if not correct_answer_string:
                    continue

                total_questions += 1
                quotes_to_find = [clean_text(q) for q in correct_answer_string.split("|") if clean_text(q)] # getting all citations as list
                context_clean = clean_text(call_result["context_text"])
                found = sum(1 for q in quotes_to_find if q in context_clean)    # exact substring matching
                total_expected = len(quotes_to_find)

                if found == total_expected:
                    hit_type = "PERFECT"; hit_score = 1.0
                elif found > 0:
                    hit_type = "PARTIAL"; hit_score = found / total_expected
                else:
                    hit_type = "MISS"; hit_score = 0.0

                # adding to our score
                successful_hits += hit_score

                if hit_type == "PERFECT":
                    print(f"{Fore.GREEN}{query_key} -> PERFECT HIT! ({found}/{total_expected}){Style.RESET_ALL}")
                elif hit_type == "PARTIAL":
                    print(f"{Fore.YELLOW}{query_key} -> PARTIAL HIT! ({found}/{total_expected}){Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}{query_key} -> MISS! (0/{total_expected}){Style.RESET_ALL}")
                print(f"  coverage: {coverage_pct}% of document text ({call_result['selected_parents_count']} parents)")

                detail_rows.append({
                    "experiment_name": EXPERIMENT_NAME,
                    "document": pdf_filename,
                    "query_key": query_key,
                    "csv_column": csv_col,
                    "top_k": FETCH_K_PER_QUERY,
                    "window_size": 0,
                    "total_doc_chunks": total_parents,
                    "fetched_chunks": call_result["selected_parents_count"],
                    "coverage_pct": coverage_pct,
                    "quotes_expected": total_expected,
                    "quotes_found": found,
                    "hit_type": hit_type,
                    "hit_score": round(hit_score, 4),
                })

            # per-call coverage summary
            print(f"  --- LLM call coverage for {pdf_filename} ---")
            print(f"  Call 1 (breach+contract+rate+principal): {call1_cov}% ({call1['selected_parents_count']} parents)")
            print(f"  Call 2 (outcome+reasoning+factors): {call2_cov}% ({call2['selected_parents_count']} parents)")
            print(f"  Total (union): {total_cov}% ({len(all_selected)} unique parents)")

            detail_rows.append({
                "experiment_name": EXPERIMENT_NAME,
                "document": pdf_filename,
                "query_key": "per_call_coverage",
                "csv_column": "",
                "top_k": FETCH_K_PER_QUERY,
                "window_size": 0,
                "total_doc_chunks": total_parents,
                "call1_chunks": call1["selected_parents_count"],
                "call1_coverage": call1_cov,
                "call2_chunks": call2["selected_parents_count"],
                "call2_coverage": call2_cov,
                "total_chunks": len(all_selected),
                "total_coverage": total_cov,
            })

    # ### Final score ###
    if total_questions > 0:
        hit_rate = (successful_hits / total_questions) * 100    # overall quote recall

        coverage_rows = [r for r in detail_rows if 'coverage_pct' in r]
        avg_coverage = round(sum(r['coverage_pct'] for r in coverage_rows) / len(coverage_rows), 1) if coverage_rows else 0.0
        call_rows = [r for r in detail_rows if r.get('query_key') == 'per_call_coverage']
        avg_call1_cov = round(sum(r['call1_coverage'] for r in call_rows) / len(call_rows), 1) if call_rows else 0.0
        avg_call2_cov = round(sum(r['call2_coverage'] for r in call_rows) / len(call_rows), 1) if call_rows else 0.0
        avg_total_cov = round(sum(r['total_coverage'] for r in call_rows) / len(call_rows), 1) if call_rows else 0.0

        # weighted coverage: weight each document by its total_doc_chunks
        if call_rows:
            weighted_num = sum(r['total_coverage'] * r['total_doc_chunks'] for r in call_rows)
            weighted_den = sum(r['total_doc_chunks'] for r in call_rows)
            weighted_total_cov = round(weighted_num / weighted_den, 1) if weighted_den > 0 else 0.0
        else:
            weighted_total_cov = 0.0

        print("\n" + "=" * 50)
        print(f"=== FINAL SCORE: {EXPERIMENT_NAME} ===")
        print("=" * 50)
        print(f"Total Questions asked:            {total_questions}")
        print(f"Perfect/Partial Hits (Weighted):  {successful_hits:.2f}")
        print(f"Overall Quote Recall Rate:        {hit_rate:.2f}%")
        print(f"Avg Token Coverage (per query):   {avg_coverage}%")
        print(f"Avg Call 1 Token Coverage:        {avg_call1_cov}%")
        print(f"Avg Call 2 Token Coverage:        {avg_call2_cov}%")
        print(f"Avg Total Token Coverage (both):  {avg_total_cov}%")
        print(f"Weighted Total Coverage (by doc size): {weighted_total_cov}%")
        print("=" * 50)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # hier mode doesn't compute MRR/Precision@k (evaluates full context, not per-chunk)
        avg_mrr_hier = ""
        avg_pak_hier = ""

        summary_exists = os.path.isfile(RESULTS_SUMMARY_CSV)
        with open(RESULTS_SUMMARY_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            if not summary_exists:
                writer.writerow([
                    "timestamp", "experiment_name", "top_k", "window_size",
                    "total_questions", "weighted_hits", "recall_rate_percent",
                    "avg_coverage_pct", "avg_call1_cov", "avg_call2_cov", "avg_total_cov",
                    "weighted_total_cov", "avg_mrr", "avg_precision_at_k"
                ])
            writer.writerow([
                current_time, EXPERIMENT_NAME, FETCH_K_PER_QUERY, 0,
                total_questions, f"{successful_hits:.2f}", f"{hit_rate:.2f}",
                f"{avg_coverage}", f"{avg_call1_cov}", f"{avg_call2_cov}", f"{avg_total_cov}",
                f"{weighted_total_cov}", avg_mrr_hier, avg_pak_hier
            ])
        print(f"Summary saved to:  {RESULTS_SUMMARY_CSV}")

        if detail_rows:
            details_exists = os.path.isfile(RESULTS_DETAILS_CSV)
            all_fields = list(dict.fromkeys(k for row in detail_rows for k in row.keys()))
            with open(RESULTS_DETAILS_CSV, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields, delimiter=";", extrasaction="ignore")
                if not details_exists:
                    writer.writeheader()
                writer.writerows(detail_rows)
        print(f"Details saved to:  {RESULTS_DETAILS_CSV}")
    else:
        print("\nNo questions to evaluate.")


# ==========================================
# Shared helper: evaluate one query group (q1, q2, or q3)
# ==========================================
# This function runs retrieval for all sub-queries of one question group,
# merges their results, deduplicates, scores, prints, and builds the detail row.
# Extracted from main() because q1, q2, q3 used the same ~80 lines of code with
# only small differences (different sub-queries, different csv column,
# different "call" set, different per-sub-query detail fields for q2/q3).
#
# Behavior is preserved exactly — same retrieval calls, same dedup logic,
# same find_hits, same hit classification, same print output, same CSV row.
def evaluate_query_group(
    *,
    group_name,             # "q1_combined", "q2_combined", "q3_combined"
    csv_column,             # "q1_context_quotes", "q2_penalty_quotes", "q3_moderation_quotes"
    sub_queries,            # list of (query_key, padded_print_label, detail_field_or_None)
    row,                    # current row from golden CSV
    collection,
    precalculated_vectors,
    pdf_filename,
    sub_query_top_k,        # may be adaptive (e.g. q1_top_k for q1)
    all_chunk_texts,
    total_doc_chunks,
    call_fetched_set,       # call1_fetched or call2_fetched (modified in-place)
):
    """Run retrieval for one question group. Returns (detail_row, hit_score, count_added)
    where count_added is 1 if the question was answered (non-empty CSV cell), else 0.
    detail_row is None if no question to evaluate."""
    correct_answer_string = row[csv_column].strip()
    if correct_answer_string == "":
        return None, 0.0, 0

    quotes_to_find = [clean_text(q) for q in correct_answer_string.split('|') if clean_text(q)]
    total_quotes_expected = len(quotes_to_find)

    # 1. Run retrieval for each sub-query
    sub_results = []  # list of (query_key, super_chunks, sims, ids, fetched_set)
    for query_key, _padded_label, _detail_field in sub_queries:
        sc, sims, ids, fetched = do_retrieve(
            collection, query_key, QUERIES[query_key],
            precalculated_vectors.get(query_key),
            pdf_filename, sub_query_top_k, WINDOW_SIZE
        )
        sub_results.append((query_key, sc, sims, ids, fetched))

    # 2. Aggregate fetched IDs across all sub-queries
    all_fetched_ids = set()
    for _, _, _, _, fetched in sub_results:
        all_fetched_ids |= fetched
    call_fetched_set.update(all_fetched_ids)
    coverage_pct = get_token_coverage(all_chunk_texts, all_fetched_ids)

    # 3. Merge super_chunks / sims / ids in sub-query order (for find_hits)
    merged_super_chunks = []
    merged_cosine_sims = []
    merged_ids = []
    for _, sc, sims, ids, _ in sub_results:
        merged_super_chunks += sc
        merged_cosine_sims += sims
        merged_ids += ids

    # 4. Deduplicate by chunk_id before find_hits — without this, duplicate chunks
    # inflate the P@k denominator (making it artificially low) and add noise to
    # MRR ranking. Recall and coverage are unaffected (already use set-based dedup).
    seen_ids = set()
    dedup_chunks, dedup_sims, dedup_ids = [], [], []
    for chunk, sim, cid in zip(merged_super_chunks, merged_cosine_sims, merged_ids):
        if cid not in seen_ids:
            seen_ids.add(cid)
            dedup_chunks.append(chunk)
            dedup_sims.append(sim)
            dedup_ids.append(cid)

    quotes_found, hit_scores, miss_scores, chunks_with_hits, rr, pak = find_hits(
        dedup_chunks, dedup_sims, quotes_to_find
    )

    # 5. Classify hit type
    if quotes_found == total_quotes_expected:
        hit_type = "PERFECT"; hit_score = 1.0
    elif quotes_found > 0:
        hit_type = "PARTIAL"; hit_score = quotes_found / total_quotes_expected
    else:
        hit_type = "MISS"; hit_score = 0.0

    # 6. Print result line (color-coded)
    if hit_type == "PERFECT":
        print(f"{Fore.GREEN}{group_name} -> PERFECT HIT! ({quotes_found}/{total_quotes_expected}){Style.RESET_ALL}")
    elif hit_type == "PARTIAL":
        print(f"{Fore.YELLOW}{group_name} -> PARTIAL HIT! ({quotes_found}/{total_quotes_expected}){Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}{group_name} -> MISS! (0/{total_quotes_expected}){Style.RESET_ALL}")

    # 7. Print per-sub-query cosine scores with correct cumulative offsets.
    # Note: chunks_with_hits indices refer to dedup_chunks, not the merged list.
    # We use cumulative offset on the merged-list positions to keep visual
    # alignment with the original (non-dedup) order — same as the original code.
    cumulative_offset = 0
    for i, (_query_key, sc, sims, ids, _fetched) in enumerate(sub_results):
        _qk, padded_label, _detail_field = sub_queries[i]
        if sims:
            display = [f"{s:.3f}{'*' if (j + cumulative_offset) in chunks_with_hits else ' '}"
                       for j, s in enumerate(sims)]
            # only the LAST sub-query line gets the (* = ...) annotation
            suffix = "  (* = chunk contained answer)" if i == len(sub_results) - 1 else ""
            print(f"  {padded_label}[{', '.join(display)}]{suffix}")
        cumulative_offset += len(sims)

    print(f"  coverage: {coverage_pct}% of document text ({len(all_fetched_ids)} chunks)")

    # 8. Build detail row
    detail_row = {
        "experiment_name":      EXPERIMENT_NAME,
        "document":             pdf_filename,
        "query_key":            group_name,
        "csv_column":           csv_column,
        "top_k":                TOP_K,
        "window_size":          WINDOW_SIZE,
        "total_doc_chunks":     total_doc_chunks,
        "fetched_chunks":       len(all_fetched_ids),
        "coverage_pct":         coverage_pct,
        "retrieved_chunk_ids":  json.dumps(merged_ids),
        "all_cosine_scores":    json.dumps(merged_cosine_sims),
        "avg_all_cosine":       safe_avg(merged_cosine_sims),
        "max_all_cosine":       round(max(merged_cosine_sims), 4) if merged_cosine_sims else "",
        "hit_cosine_scores":    json.dumps(hit_scores),
        "avg_hit_cosine":       safe_avg(hit_scores),
        "miss_cosine_scores":   json.dumps(miss_scores),
        "avg_miss_cosine":      safe_avg(miss_scores),
        "quotes_expected":      total_quotes_expected,
        "quotes_found":         quotes_found,
        "hit_type":             hit_type,
        "hit_score":            round(hit_score, 4),
        "reciprocal_rank":      round(rr, 4),
        "precision_at_k":       round(pak, 4),
    }

    # 9. Add per-sub-query cosine score fields (only for groups that need them)
    # q1 has no per-sub-query fields, q2 has rate + principal, q3 has outcome + reasoning + factors
    for i, (_query_key, _padded_label, detail_field) in enumerate(sub_queries):
        if detail_field is not None:
            _qk, _sc, sims, _ids, _fetched = sub_results[i]
            detail_row[detail_field] = json.dumps(sims)

    return detail_row, hit_score, 1


def main():
    # ==========================================
    # DISPATCH: flat vs hierarchical evaluation
    # ==========================================
    if CHUNK_MODE == "hier":
        return run_hier_evaluation()

    # ==========================================
    # FIX: BIG BRAIN MOMENT!
    # I MUST initialize colorama here ONLY ONCE at the start!
    # Omg I had it inside the loop before and it was restarting my terminal 30 times.
    # It was completely freezing my Mac and I couldn't find the error for 5 hours! :D
    # ==========================================
    init(autoreset=True)
    print(f"=== STARTING EVALUATION: {EXPERIMENT_NAME} | TOP_K={TOP_K} | WINDOW=±{WINDOW_SIZE} ===")

    # ==========================================
    # STEP 3: Connect to the database and Model
    # ==========================================
    print(f"  Retrieval mode: {RETRIEVAL_MODE}")
    # show reranker status so i know if its on when running experiments
    if USE_RERANKER:
        print(f"  Reranker: ON (alpha={RERANKER_ALPHA}, oversample={RERANKER_OVERSAMPLE}x)")
    else:
        print(f"  Reranker: OFF")

    # ==========================================
    # MODEL LOADING - only needed for dense and hybrid modes
    # BM25 mode doesn't need any embedding model!
    # ==========================================
    precalculated_vectors = {}

    if RETRIEVAL_MODE in ("dense", "hybrid"):
        if MODEL_TYPE == "me5":
            print("Loading mE5-base model explicitly (CPU mode)...")
            model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

            def embed_query(text):
                """Embed a single query using mE5. Adds 'query: ' prefix as required by mE5."""
                return model.encode([f"query: {text}"], normalize_embeddings=True).tolist()

        elif MODEL_TYPE in ("openai", "openai_large"):
            OPENAI_MODEL = "text-embedding-3-large" if MODEL_TYPE == "openai_large" else "text-embedding-3-small"
            print(f"Using OpenAI {OPENAI_MODEL} via API...")
            openai_client = OpenAI()

            def embed_query(text):
                """Embed a single query using OpenAI. No prefix needed."""
                response = openai_client.embeddings.create(
                    model=OPENAI_MODEL,
                    input=[text],
                )
                return [response.data[0].embedding]

        else:
            raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Use 'me5', 'openai', or 'openai_large'.")

        # FIX 2: BIG OPTIMIZATION!
        # I am asking the EXACT same questions for every document.
        # So I will translate them into vectors only ONCE here, before the loop starts!
        print("Pre-calculating query vectors to save time...")
        for q_key, q_text in QUERIES.items():
            precalculated_vectors[q_key] = embed_query(q_text)
    else:
        print("BM25 mode - no embedding model needed, using keyword matching")

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)

    # Get the database table WITHOUT embedding function because I made vectors manually.
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Variables to track my score
    total_questions = 0
    successful_hits = 0

    # Collect all detailed rows during the loop, write to CSV at the end
    detail_rows = []

    # ==========================================
    # STEP 4: Read the Golden Dataset (CSV)
    # ==========================================
    print("Reading Golden Dataset...")

    # Pythons modul DictReader expects "," between sections but my lawyer excel saved it with ";"
    with open(CSV_PATH, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=';')

        for row in reader:
            doc_name = row["document_name"]

            # Fix filename: In DB I saved them as .pdf, but in CSV they have .json
            pdf_filename = doc_name.replace(".json", ".pdf")
            print(f"\n--- Testing document: {pdf_filename} ---")

            # get total chunk count once per document - reused for all queries below.
            # important: coverage of 80% means i read 80% of the doc - thats basically
            # not RAG anymore. i want HIGH recall at LOW coverage to prove RAG is worth it.
            total_doc_chunks = get_doc_total_chunks(collection, pdf_filename)

            # get all chunk texts for token-based coverage (fairer than chunk-count
            # because paragraphs vary hugely: some 50 tokens, others 1500+)
            all_chunk_texts = get_doc_all_texts(collection, pdf_filename)

            # track which chunks each LLM call will actually read
            # Call 1 = q_breach + q_contract + q_rate + q_principal (contract facts)
            # Call 2 = q_outcome + q_reasoning + q_factors (moderation analysis)
            call1_fetched = set()
            call2_fetched = set()

            # ==========================================
            # STEP 5.1: Combined query (q_breach + q_contract -> q1_context_quotes)
            # q_breach retrieves chunks about the breach/violation itself
            # q_contract retrieves chunks about the contract type, parties, and terms
            #
            # NOTE: i tried adding a 3rd sub-query "q_facts" here to capture the
            # factual background (skutkový stav). but experiments showed it only added
            # +0.42% recall for +1.6% coverage - the hybrid retriever with reranker
            # already finds those chunks through q_breach and q_contract. so i removed
            # it to keep coverage efficient. see experiment_results_summary_with_q3sub.csv
            # vs experiment_results_summary_without_q3sub.csv for the comparison.
            # ==========================================
            # Adaptive top_k for q1: larger documents need more chunks to find
            # contract context, because the factual background is spread across
            # more numbered points. For small docs (<30 chunks) we use default top_k.
            # FIX: raised cap from +3 to +5 and changed divisor from 20 to 15.
            # This matches the updated formula in precompute_retrieval.py.
            if total_doc_chunks > 30:
                q1_top_k = min(TOP_K + 5, TOP_K + (total_doc_chunks - 30) // 15)
            else:
                q1_top_k = TOP_K

            q1_detail, q1_hit_score, q1_added = evaluate_query_group(
                group_name="q1_combined",
                csv_column="q1_context_quotes",
                sub_queries=[
                    ("q_breach",   "q_breach scores:   ", None),
                    ("q_contract", "q_contract scores: ", None),
                ],
                row=row,
                collection=collection,
                precalculated_vectors=precalculated_vectors,
                pdf_filename=pdf_filename,
                sub_query_top_k=q1_top_k,
                all_chunk_texts=all_chunk_texts,
                total_doc_chunks=total_doc_chunks,
                call_fetched_set=call1_fetched,  # q_breach + q_contract go to Call 1
            )
            if q1_detail is not None:
                total_questions += q1_added
                successful_hits += q1_hit_score
                detail_rows.append(q1_detail)

            # ==========================================
            # STEP 5.2: Combined query (q_rate + q_principal -> q2_penalty_quotes)
            # q_rate retrieves chunks about penalty rate/calculation method
            # q_principal retrieves chunks about the secured amount (istina)
            # These are often in the same paragraph but sometimes in different sections.
            # Merging gives better coverage than one long multi-topic query.
            # ==========================================
            q2_detail, q2_hit_score, q2_added = evaluate_query_group(
                group_name="q2_combined",
                csv_column="q2_penalty_quotes",
                sub_queries=[
                    ("q_rate",      "q_rate scores:      ", "rate_cosine_scores"),
                    ("q_principal", "q_principal scores: ", "principal_cosine_scores"),
                ],
                row=row,
                collection=collection,
                precalculated_vectors=precalculated_vectors,
                pdf_filename=pdf_filename,
                sub_query_top_k=TOP_K,
                all_chunk_texts=all_chunk_texts,
                total_doc_chunks=total_doc_chunks,
                call_fetched_set=call1_fetched,  # q_rate + q_principal go to Call 1
            )
            if q2_detail is not None:
                total_questions += q2_added
                successful_hits += q2_hit_score
                detail_rows.append(q2_detail)

            # ==========================================
            # STEP 5.3: Combined query (q_outcome + q_reasoning + q_factors -> q3_moderation_quotes)
            # Three focused queries each retrieve chunks about a different aspect:
            # q_outcome  -> what happened (awarded/dismissed/returned)
            # q_reasoning -> why (court's argumentation)
            # q_factors  -> specific legal factors (dobré mravy, pomer k istine, škoda...)
            # Merging all three gives the full picture of the court's moderation analysis.
            # ==========================================
            q3_detail, q3_hit_score, q3_added = evaluate_query_group(
                group_name="q3_combined",
                csv_column="q3_moderation_quotes",
                sub_queries=[
                    ("q_outcome",   "q_outcome scores:   ", "outcome_cosine_scores"),
                    ("q_reasoning", "q_reasoning scores: ", "reasoning_cosine_scores"),
                    ("q_factors",   "q_factors scores:   ", "factors_cosine_scores"),
                ],
                row=row,
                collection=collection,
                precalculated_vectors=precalculated_vectors,
                pdf_filename=pdf_filename,
                sub_query_top_k=TOP_K,
                all_chunk_texts=all_chunk_texts,
                total_doc_chunks=total_doc_chunks,
                call_fetched_set=call2_fetched,  # q_outcome + q_reasoning + q_factors go to Call 2
            )
            if q3_detail is not None:
                total_questions += q3_added
                successful_hits += q3_hit_score
                detail_rows.append(q3_detail)

            # ==========================================
            # STEP 6: Per-call coverage
            # This is what actually matters - how much of the document each LLM call reads.
            # The per-query coverage above is useful for debugging but this is the real metric.
            # ==========================================
            # token-based coverage: what % of document TEXT each call reads
            # fairer than chunk-count since paragraphs vary hugely in size
            call1_cov = get_token_coverage(all_chunk_texts, call1_fetched)
            call2_cov = get_token_coverage(all_chunk_texts, call2_fetched)
            total_fetched = call1_fetched | call2_fetched
            total_cov = get_token_coverage(all_chunk_texts, total_fetched)

            print(f"  --- LLM call coverage for {pdf_filename} (token-based) ---")
            print(f"  Call 1 (breach+contract+rate+principal): {call1_cov}% ({len(call1_fetched)} chunks)")
            print(f"  Call 2 (outcome+reasoning+factors): {call2_cov}% ({len(call2_fetched)} chunks)")
            print(f"  Total (union of both calls): {total_cov}% ({len(total_fetched)} unique chunks)")

            detail_rows.append({
                "experiment_name":  EXPERIMENT_NAME,
                "document":         pdf_filename,
                "query_key":        "per_call_coverage",
                "csv_column":       "",
                "top_k":            TOP_K,
                "window_size":      WINDOW_SIZE,
                "total_doc_chunks": total_doc_chunks,
                "call1_chunks":     len(call1_fetched),
                "call1_coverage":   call1_cov,
                "call2_chunks":     len(call2_fetched),
                "call2_coverage":   call2_cov,
                "total_chunks":     len(total_fetched),
                "total_coverage":   total_cov,
            })

    # ==========================================
    # STEP 7: Calculate, print AND SAVE final score
    # ==========================================
    if total_questions > 0:
        hit_rate = (successful_hits / total_questions) * 100

        # avg coverage across all rows - key thesis metric next to recall
        coverage_rows = [r for r in detail_rows if 'coverage_pct' in r]
        avg_coverage = round(sum(r['coverage_pct'] for r in coverage_rows) / len(coverage_rows), 1) if coverage_rows else 0.0

        # per-call coverage averages - this is what the LLM actually sees
        call_rows = [r for r in detail_rows if r.get('query_key') == 'per_call_coverage']
        avg_call1_cov = round(sum(r['call1_coverage'] for r in call_rows) / len(call_rows), 1) if call_rows else 0.0
        avg_call2_cov = round(sum(r['call2_coverage'] for r in call_rows) / len(call_rows), 1) if call_rows else 0.0
        avg_total_cov = round(sum(r['total_coverage'] for r in call_rows) / len(call_rows), 1) if call_rows else 0.0

        # weighted coverage: weight each document by its total_doc_chunks
        # Simple average is biased by tiny docs (4 chunks → 100% trivially).
        # Weighted average gives larger (harder) documents more influence,
        # which is fairer for thesis evaluation.
        if call_rows:
            weighted_num = sum(r['total_coverage'] * r['total_doc_chunks'] for r in call_rows)
            weighted_den = sum(r['total_doc_chunks'] for r in call_rows)
            weighted_total_cov = round(weighted_num / weighted_den, 1) if weighted_den > 0 else 0.0
        else:
            weighted_total_cov = 0.0

        # --- Aggregate MRR and Precision@k across all query groups ---
        # MRR = Mean Reciprocal Rank: how high the first relevant chunk ranks (1.0 = first position)
        # Precision@k = what fraction of retrieved chunks actually contain golden quotes
        mrr_vals = [r['reciprocal_rank'] for r in detail_rows if 'reciprocal_rank' in r]
        avg_mrr = round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else 0.0
        pak_vals = [r['precision_at_k'] for r in detail_rows if 'precision_at_k' in r]
        avg_precision_at_k = round(sum(pak_vals) / len(pak_vals), 4) if pak_vals else 0.0

        print("\n" + "=" * 50)
        print(f"=== FINAL SCORE: {EXPERIMENT_NAME} ===")
        print("=" * 50)
        print(f"Total Questions asked:            {total_questions}")
        print(f"Perfect/Partial Hits (Weighted):  {successful_hits:.2f}")
        print(f"Overall Quote Recall Rate:        {hit_rate:.2f}%")
        print(f"Avg Coverage (per query group):   {avg_coverage}%  (token-based)")
        print(f"Avg Call 1 Token Coverage:        {avg_call1_cov}%")
        print(f"Avg Call 2 Token Coverage:        {avg_call2_cov}%")
        print(f"Avg Total Token Coverage (both):  {avg_total_cov}%")
        print(f"Weighted Total Coverage (by doc size): {weighted_total_cov}%")
        print(f"Mean Reciprocal Rank (MRR):       {avg_mrr}")
        print(f"Avg Precision@k:                  {avg_precision_at_k}")
        print("=" * 50)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---------------------------------------------------------
        # SAVE 1: SUMMARY CSV
        # One row per experiment run. Good for high-level comparison between experiments.
        # avg_coverage_pct is the key tradeoff metric: recall vs how much of the doc we read
        # MRR shows ranking quality, precision@k shows retrieval precision
        # ---------------------------------------------------------
        summary_exists = os.path.isfile(RESULTS_SUMMARY_CSV)
        with open(RESULTS_SUMMARY_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            if not summary_exists:
                writer.writerow([
                    "timestamp", "experiment_name", "top_k", "window_size",
                    "total_questions", "weighted_hits", "recall_rate_percent",
                    "avg_coverage_pct", "avg_call1_cov", "avg_call2_cov", "avg_total_cov",
                    "weighted_total_cov", "avg_mrr", "avg_precision_at_k"
                ])
            writer.writerow([
                current_time, EXPERIMENT_NAME, TOP_K, WINDOW_SIZE,
                total_questions, f"{successful_hits:.2f}", f"{hit_rate:.2f}",
                f"{avg_coverage}", f"{avg_call1_cov}", f"{avg_call2_cov}", f"{avg_total_cov}",
                f"{weighted_total_cov}", f"{avg_mrr}", f"{avg_precision_at_k}"
            ])
        print(f"Summary saved to:  {RESULTS_SUMMARY_CSV}")

        # ---------------------------------------------------------
        # SAVE 2: DETAILS CSV
        # One row per question per document.
        # Key columns for scientific analysis:
        #   coverage_pct             -> % of doc chunks read by this query
        #   all_cosine_scores        -> full retrieval picture
        #   outcome_cosine_scores    -> what q_outcome alone retrieved (only in q3_combined rows)
        #   reasoning_cosine_scores  -> what q_reasoning alone retrieved (only in q3_combined rows)
        #   hit_cosine_scores        -> true retrieval confidence (only chunks with answers)
        #   miss_cosine_scores       -> chunks model ranked high but were useless
        # ---------------------------------------------------------
        if detail_rows:
            details_exists = os.path.isfile(RESULTS_DETAILS_CSV)
            # collect all possible fieldnames across all rows (q3_combined has extra fields)
            all_fields = list(dict.fromkeys(k for row in detail_rows for k in row.keys()))
            with open(RESULTS_DETAILS_CSV, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields, delimiter=";", extrasaction="ignore")
                if not details_exists:
                    writer.writeheader()
                writer.writerows(detail_rows)
        print(f"Details saved to:  {RESULTS_DETAILS_CSV}")

    else:
        print("\nNo questions to evaluate. Is the CSV empty?")


if __name__ == "__main__":
    main()
