import os
import csv
import json  # FIX: needed for json.dumps() when saving lists to CSV - see detail_rows below
import chromadb
import datetime
import re
from colorama import Fore, Style, init

# these are set by experiment_runner.py via env variables so i dont have to edit this file
# for each experiment. if running manually, just change the defaults here.
EXPERIMENT_NAME = os.environ.get("EXP_NAME",        "mE5_chunk380_top5_window1")
TOP_K           = int(os.environ.get("EXP_TOP_K",   "5"))
WINDOW_SIZE     = int(os.environ.get("EXP_WINDOW",  "1"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "legal_decisions_me5_380")
MODEL_TYPE      = os.environ.get("MODEL_TYPE",      "me5")  # "me5" or "openai"

# ==========================================
# RETRIEVAL MODE: "dense" (default), "bm25", or "hybrid" (dense + BM25 with RRF)
# - dense:  cosine similarity via embedding model (original behavior)
# - bm25:   keyword matching via BM25 (no embeddings needed)
# - hybrid: both dense + BM25 combined using Reciprocal Rank Fusion (RRF)
# ==========================================
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "dense")
RRF_K = int(os.environ.get("RRF_K", "60"))  # RRF constant, standard value = 60

# ==========================================
# WEIGHTED RRF: how much dense vs BM25 matters in hybrid
# alpha=0.5 = equal weight (original RRF)
# alpha=0.7 = dense dominates, BM25 is a light boost
# alpha=1.0 = pure dense (BM25 ignored)
#
# Why not 0.5? BM25 with paragraph chunks pulls in noisy chunks (procedural text
# full of legal keywords) that push out good dense results. Dense is better at
# understanding semantic relevance, but misses introductory chunks (chunk_0)
# where facts are stated as context, not as main topic.
# alpha=0.7 lets BM25 boost chunks with keyword matches without overriding dense.
# ==========================================
RRF_ALPHA = float(os.environ.get("RRF_ALPHA", "0.5"))  # default 0.5 = standard equal-weight RRF

# ==========================================
# RERANKER: keyword-based reranking after retrieval
# ==========================================
# When turned on, retrieval fetches MORE chunks than usual (e.g. 10 instead of 5)
# and then the reranker re-scores them using legal keyword patterns and picks the
# best top_k. The idea is that retrieval sometimes puts procedural chunks above
# informative ones because they are semantically similar, and keyword matching
# can fix that.
#
# USE_RERANKER=1 to turn it on, =0 to keep original behavior (default off)
# RERANKER_ALPHA controls balance: 0.6 means retrieval is more important than keywords
# RERANKER_OVERSAMPLE: how many times more chunks to fetch (2 = fetch 2x, rerank to top_k)
# RERANKER_DEBUG=1 to print detailed scoring info for each chunk
# ==========================================
USE_RERANKER = os.environ.get("USE_RERANKER", "0") == "1"
RERANKER_ALPHA = float(os.environ.get("RERANKER_ALPHA", "0.6"))
RERANKER_OVERSAMPLE = int(os.environ.get("RERANKER_OVERSAMPLE", "2"))
RERANKER_DEBUG = os.environ.get("RERANKER_DEBUG", "0") == "1"

if USE_RERANKER:
    # need to add project root to path so python can find src.candidates.reranker
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.candidates.reranker import rerank_chunks

# ==========================================
# FIX: Mac M2 freezing problem!
# These lines stop my Mac M2 from freezing. It turns off parallel processing and threads.
# I also disable Chroma telemetry so it doesn't hang my script randomly.
# ==========================================
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

# ==========================================
# STEP 1: Setup paths and basic variables
# ==========================================
DB_DIR = "data/04_vectorstore"
EVAL_DIR = "data/05_retrieval_evaluation"
CSV_PATH = os.path.join(EVAL_DIR, "golden_dataset_template.csv")

# same as DB_DIR above - make sure folder exists before we try to write CSVs into it
# without this the script would crash at the very end with FileNotFoundError after
# doing all the work (already computed everything, just cant save it - very annoying)
os.makedirs(EVAL_DIR, exist_ok=True)

# Two output CSVs:
# - SUMMARY: one row per experiment run (for comparing experiments at high level)
# - DETAILS: one row per question per document (for deep analysis, cosine scores, thesis charts)
RESULTS_SUMMARY_CSV = os.path.join(EVAL_DIR, "experiment_results_summary.csv")
RESULTS_DETAILS_CSV = os.path.join(EVAL_DIR, "experiment_results_details.csv")

# ==========================================
# STEP 2: Define  queries
# ==========================================
# I now use 4 queries instead of 3. The old q3 was too broad - it asked about
# both the OUTCOME (was penalty reduced?) and the REASONING (why?) in one query.
# This was bad for cases where §301 was NOT applied - the dense model would retrieve
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

# old ones
# QUERIES = {
#     # Channel A
#     "q_context": "Čo bolo predmetom zmluvy medzi stranami? Akú povinnosť si dlžník prevzal a ako ju porušil?",
#
#     # Channel B
#     # "q_penalty": "Aká je výška, sadzba a mena zmluvnej pokuty? Z akej sumy sa počíta a za aké porušenie povinnosti bola dohodnutá?", - (in slovak so it would be understandable) istina zavazku ktory pokuta zabezpecuje — je obcas uvedena len v kontexte zmluvy, nie v odseku o pokute. Query "Z akej sumy sa počíta" to vacsinou zachytila, ale nie vzdy
#     "q_penalty": "Aká je výška, sadzba a mena zmluvnej pokuty? Z akej sumy sa počíta a za aké porušenie povinnosti bola dohodnutá? Aká je výška hlavného záväzku (istiny) ktorý pokuta zabezpečuje?",
#
#     # Channel C1 - vysledok konania (what happened to the penalty)
#     # This explicitly covers ALL possible outcomes: §301 reduction, dismissal for invalidity,
#     # procedural dismissal, case returned for retrial - not just the §301 moderation case!
#     "q_outcome": "Aký bol výsledok konania o zmluvnej pokute? Bola priznaná, znížená alebo zamietnutá? Uplatnil súd moderačné oprávnenie podľa § 301 Obchodného zákonníka? Alebo bola pokuta zamietnutá pre neplatnosť zmluvy, procesný dôvod, alebo vrátená na ďalšie konanie?",
#
#     # Channel C2 - argumenty súdu (why)
#     # This retrieves the reasoning chunks regardless of what the outcome was.
#     "q_reasoning": "Aké dôvody uviedol súd pri posudzovaní zmluvnej pokuty? Napríklad výška škody, rozpor s dobrými mravmi, pomer k istine, správanie dlžníka, zabezpečovacia funkcia, kumulácia s úrokom z omeškania?",
# }

QUERIES = {
    # Call 1
    "q_breach":    "Čo bolo predmetom zmluvy a akú povinnosť dlžník porušil?",
    "q_rate":      "Aká je sadzba a spôsob výpočtu zmluvnej pokuty?",
    "q_principal": "Aká je výška istiny, dlžnej sumy alebo hlavného záväzku?",
    # Call 2
    "q_outcome":   "Bola zmluvná pokuta priznaná, znížená podľa § 301, zamietnutá alebo vrátená?",
    "q_reasoning": "Aké argumenty súd použil pri posudzovaní primeranosti zmluvnej pokuty?",
    "q_factors":   "Posúdil súd rozpor s dobrými mravmi, pomer pokuty k istine, výšku škody alebo kumuláciu s úrokom?",
}

# ==========================================
# BM25-SPECIFIC QUERIES
# ==========================================
# BM25 matches exact keywords, not semantic meaning. So queries must contain
# the actual words that appear in the text, not questions about them.
#
# Dense: "Aká je výška istiny?" → embedding captures intent → finds chunks about amounts
# BM25:  "istina suma eur pohľadávka dlh záväzok" → matches exact tokens
#
# The BM25 queries include multiple synonyms and inflected forms that simplemma
# might not perfectly lemmatize. This is intentional redundancy.
# ==========================================
BM25_QUERIES = {
    # Call 1
    # Added "uzatvorená skutkový stav sporové strany napadnutým rozsudkom" — these words appear
    # in introductory paragraphs (chunk_0) where the contract and breach are DESCRIBED as context.
    # Without them, BM25 only finds chunks where breach is DISCUSSED, missing basic case facts.
    "q_breach":    "zmluva uzatvorená predmet porušenie povinnosť záväzok dlžník zhotoviteľ objednávateľ žalovaný neuhradil nesplnil skutkový stav sporové strany napadnutým rozsudkom",
    "q_rate":      "zmluvná pokuta výška sadzba percentá ročne denne omeškanie eur suma článok bod zmluvy dohodnutá",
    "q_principal": "istina suma pohľadávka dlh záväzok eur faktúra cena dielo splatnosť uhradiť zaplatiť",
    # Call 2
    "q_outcome":   "zmluvná pokuta priznaná znížená zamietnutá moderácia moderačné oprávnenie §301 neplatnosť vrátená rozhodol uložil zaviazal",
    "q_reasoning": "primeranosť neprimeraná dôvod posúdenie zníženie argumenty súd záver odôvodnenie odvolací potvrdil zmenil",
    "q_factors":   "dobré mravy škoda pomer istina úrok omeškanie kumulácia zabezpečovacia funkcia hodnota význam zabezpečovanej povinnosti",
}

# This tells the evaluator which CSV column each query maps to.
# q_outcome and q_reasoning SHARE the same CSV column - their retrieved chunks are
# merged into one pool before checking for golden quotes. See Step 5.2 for details.
# QUERY_TO_CSV_COLUMN = {
#     "q_context":   "q1_context_quotes",
#     "q_penalty":   "q2_penalty_quotes",
#     "q_outcome":   "q3_moderation_quotes",   # ─┐ evaluated
#     "q_reasoning": "q3_moderation_quotes",   # ─┘ together
# }

# mapping
QUERY_TO_CSV_COLUMN = {
    "q_breach":    "q1_context_quotes",
    "q_rate":      "q2_penalty_quotes",    # ─┐ evaluated
    "q_principal": "q2_penalty_quotes",    # ─┘ together
    "q_outcome":   "q3_moderation_quotes", # ─┐
    "q_reasoning": "q3_moderation_quotes", # ─┤ evaluated
    "q_factors":   "q3_moderation_quotes", # ─┘ together
}



# ==========================================
# FIX: TEXT CLEANING SO PYTHON DOESNT FAIL STUPIDLY
# If my PDF has "500 \n eur" but Excel has "500 eur", normal Python says "MISS!".
# This function removes all new lines and extra spaces so matching is 100% bulletproof.
# ==========================================
def clean_text(text):
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r'\s+', ' ', text).strip()


def safe_avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else ""


# ==========================================
# BM25 SUPPORT
# ==========================================
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
# ==========================================

# Slovak legal stopwords - common function words that appear in nearly every chunk
# and carry no retrieval signal. Kept short to avoid removing anything useful.
SLOVAK_STOPWORDS = {
    "a", "aj", "ak", "ako", "ale", "alebo", "ani", "áno", "asi",
    "by", "bol", "bola", "bolo", "boli", "buď", "byť", "bez",
    "do", "dňa",
    "ho", "jeho", "jej", "ich",
    "je", "ju",
    "keď", "keďže", "kde", "ku", "ktorý", "ktorá", "ktoré", "ktorej", "ktorým", "ktorom",
    "na", "nad", "nie", "no", "než",
    "od", "ods",
    "po", "pod", "podľa", "pre", "pri", "pred", "preto",
    "sa", "si", "so", "sú",
    "ten", "to", "tá", "tú", "tej", "tom", "tomu", "tým", "tento", "tiež", "tak", "takže",
    "vo", "vz",
    "za", "zo", "že",
}


def tokenize_slovak(text):
    """Tokenizer for Slovak legal text with number preservation and lemmatization.

    Uses a single regex findall to extract three types of tokens:
    1. Legal references: §301, §544 (§ + digits, joined even if space in original)
    2. Numbers/percentages: 0,05% | 15.234,60 | 2.000 (Slovak decimal format)
    3. Words: sequences of letters (including Slovak diacritics)

    Then lemmatizes words and filters stopwords.
    """
    text = text.lower()

    # Join § with following number before tokenization: "§ 301" → "§301"
    text = re.sub(r'§\s*(\d+)', r'§\1', text)

    # Single findall extracts all token types in priority order:
    # 1. §-references (§301) — must be first to prevent partial number match
    # 2. Numbers: Slovak format with . thousands, , decimals, optional %
    # 3. Words: letter sequences (Slovak alphabet including diacritics)
    raw_tokens = re.findall(
        r'§\d+'                                    # legal refs: §301, §544
        r'|\d{1,3}(?:\.\d{3})*(?:,\d+)?%?'        # numbers: 0,05% | 15.234,60 | 2.000
        r'|[a-záäčďéíľĺňóôŕšťúýžA-ZÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ]+'  # words with Slovak chars
        , text
    )

    # Lemmatize words, keep numbers and § refs as-is, filter stopwords
    tokens = []
    for t in raw_tokens:
        if t.startswith("§") or t[0].isdigit():
            tokens.append(t)  # keep numbers and legal refs as-is
        elif len(t) > 1:
            lemma = simplemma.lemmatize(t, lang='sk')
            if lemma not in SLOVAK_STOPWORDS and len(lemma) > 1:
                tokens.append(lemma)

    return tokens


# Cache BM25 index per document to avoid rebuilding for each query
_bm25_cache = {}  # {pdf_filename: (bm25_index, chunk_ids, chunk_texts, chunk_metadatas)}

def get_bm25_index(collection, pdf_filename):
    """Build (or retrieve cached) BM25 index for all chunks of a document."""
    if pdf_filename in _bm25_cache:
        return _bm25_cache[pdf_filename]

    # Get ALL chunks for this document from ChromaDB
    result = collection.get(
        where={"source_file": pdf_filename},
        include=["documents", "metadatas"]
    )

    # Sort by chunk_index to ensure consistent ordering
    paired = list(zip(result['ids'], result['documents'], result['metadatas']))
    paired.sort(key=lambda x: x[2]['chunk_index'])

    chunk_ids = [p[0] for p in paired]
    chunk_texts = [p[1] for p in paired]
    chunk_metadatas = [p[2] for p in paired]

    # Build BM25 index from tokenized chunks
    # k1=1.2: faster saturation (legal chunks are short, term appearing 2-3x is enough signal)
    # b=0.4: reduced length normalization (paragraph chunks are similar in length)
    tokenized_corpus = [tokenize_slovak(text) for text in chunk_texts]
    bm25_index = BM25Okapi(tokenized_corpus, k1=1.2, b=0.4)

    _bm25_cache[pdf_filename] = (bm25_index, chunk_ids, chunk_texts, chunk_metadatas)
    return _bm25_cache[pdf_filename]


def retrieve_bm25(collection, query_text, pdf_filename, top_k, window_size):
    """BM25 retrieval - returns same format as retrieve_super_chunks for compatibility.
    Returns (super_chunks, scores, retrieved_ids, fetched_chunk_ids).
    """
    bm25_index, all_ids, all_texts, all_metas = get_bm25_index(collection, pdf_filename)

    tokenized_query = tokenize_slovak(query_text)
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # Get top_k indices by BM25 score (descending)
    ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    super_chunks = []
    scores = []
    ret_ids = []
    fetched_chunk_ids = set()

    for idx in ranked_indices:
        c_idx = all_metas[idx]['chunk_index']
        score = round(float(bm25_scores[idx]), 4)

        # Window expansion - same logic as dense retrieval
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

        scores.append(score)
        ret_ids.append(all_ids[idx])

    return super_chunks, scores, ret_ids, fetched_chunk_ids


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

    Returns same format as retrieve_super_chunks.
    """
    alpha = RRF_ALPHA  # read from env/global

    # --- Dense ranking ---
    results = collection.query(
        query_embeddings=query_vector,
        n_results=min(top_k * 3, 20),  # get more candidates for fusion
        where={"source_file": pdf_filename},
        include=["documents", "metadatas", "distances"]
    )
    dense_ids = results['ids'][0] if results['ids'] else []
    dense_ranks = {doc_id: rank for rank, doc_id in enumerate(dense_ids)}

    # --- BM25 ranking ---
    bm25_index, all_ids, all_texts, all_metas = get_bm25_index(collection, pdf_filename)
    tokenized_query = tokenize_slovak(query_text)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_id_ranks = {all_ids[idx]: rank for rank, idx in enumerate(bm25_ranked_indices[:top_k * 3])}

    # --- Weighted RRF fusion ---
    all_candidate_ids = set(dense_ranks.keys()) | set(bm25_id_ranks.keys())
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

    # --- Build super_chunks with window expansion ---
    super_chunks = []
    scores = []
    ret_ids = []
    fetched_chunk_ids = set()

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


def retrieve_super_chunks(collection, query_vector, pdf_filename, top_k, window_size):
    """Run one query, return (super_chunks, cosine_sims, retrieved_ids, fetched_chunk_ids).
    super_chunks[i] maps 1:1 to cosine_sims[i] and retrieved_ids[i].
    fetched_chunk_ids = set of ALL chunk IDs actually read from DB (TOP_K + window neighbors).
    I need this set to compute coverage - how much of the doc did this query actually look at?
    """
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
        where={"source_file": pdf_filename},
        include=["documents", "metadatas", "distances"]
    )

    raw_distances = results['distances'][0] if results['distances'] else []
    cosine_sims = [round(1 - d, 4) for d in raw_distances]
    ret_ids = results['ids'][0] if results['ids'] else []

    super_chunks = []
    fetched_chunk_ids = set()  # track every chunk we actually read (TOP_K + their neighbors)

    if len(results['ids']) > 0 and len(results['ids'][0]) > 0:
        retrieved_metadatas = results['metadatas'][0]

        for i in range(len(ret_ids)):
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
    """Check which quotes are found in super_chunks.
    Returns (quotes_found_count, hit_cosine_scores, miss_cosine_scores, chunks_that_had_hits).
    """
    chunks_that_had_hits = set()

    for quote in quotes_to_find:
        for j, super_chunk in enumerate(super_chunks):
            if quote in super_chunk:
                chunks_that_had_hits.add(j)
                break

    quotes_found_count = 0
    for quote in quotes_to_find:
        for j, super_chunk in enumerate(super_chunks):
            if quote in super_chunk:
                quotes_found_count += 1
                break

    hit_scores = [sim for j, sim in enumerate(cosine_sims) if j in chunks_that_had_hits]
    miss_scores = [sim for j, sim in enumerate(cosine_sims) if j not in chunks_that_had_hits]

    return quotes_found_count, hit_scores, miss_scores, chunks_that_had_hits


def get_doc_total_chunks(collection, pdf_filename):
    # how many chunks does this document have in total?
    # used as denominator for coverage: fetched_chunks / total_chunks
    result = collection.get(where={"source_file": pdf_filename}, include=["metadatas"])
    return len(result['ids'])


def do_retrieve(collection, query_key, query_text, query_vector, pdf_filename, top_k, window_size):
    """Pick the right retrieval method and optionally rerank the results.

    For BM25 and hybrid modes, uses BM25_QUERIES (keyword-optimized) instead of
    the semantic QUERIES used for dense retrieval. This is intentional:
    - Dense needs natural language questions for good embeddings
    - BM25 needs domain keywords for exact token matching

    When the reranker is turned on, this fetches more chunks than needed (2x by default),
    then the reranker re-scores them using legal keywords and picks the best top_k.
    """
    # BM25 uses keyword queries; dense uses semantic queries
    bm25_query_text = BM25_QUERIES.get(query_key, query_text) if RETRIEVAL_MODE in ("bm25", "hybrid") else query_text

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
        # not the full oversampled pool. otherwise coverage would be unfairly inflated
        # compared to baseline experiments without reranker.
        fetched_ids = set(ret_ids)

    return super_chunks, cosine_sims, ret_ids, fetched_ids


def main():
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

            # track which chunks each LLM call will actually read
            # Call 1 = q_breach + q_rate + q_principal (contract facts)
            # Call 2 = q_outcome + q_reasoning + q_factors (moderation analysis)
            call1_fetched = set()
            call2_fetched = set()

            # ==========================================
            # STEP 5.1: Single queries (q_context and q_penalty)
            # These map 1:1 to a CSV column so evaluation is straightforward.
            # ==========================================
            # for q_key in ["q_context", "q_penalty"]:
            for q_key in ["q_breach"]:
                csv_col = QUERY_TO_CSV_COLUMN[q_key]
                correct_answer_string = row[csv_col].strip()

                if correct_answer_string == "":
                    continue

                total_questions += 1
                quotes_to_find = [clean_text(q) for q in correct_answer_string.split('|') if clean_text(q)]

                super_chunks, cosine_sims, ret_ids, fetched_ids = do_retrieve(
                    collection, q_key, QUERIES[q_key], precalculated_vectors.get(q_key),
                    pdf_filename, TOP_K, WINDOW_SIZE
                )
                call1_fetched.update(fetched_ids)  # q_breach goes to Call 1

                coverage_pct = round(len(fetched_ids) / total_doc_chunks * 100, 1) if total_doc_chunks > 0 else 0.0

                quotes_found, hit_scores, miss_scores, chunks_with_hits = find_hits(
                    super_chunks, cosine_sims, quotes_to_find
                )

                total_quotes_expected = len(quotes_to_find)

                if quotes_found == total_quotes_expected:
                    hit_type = "PERFECT"; hit_score = 1.0
                elif quotes_found > 0:
                    hit_type = "PARTIAL"; hit_score = quotes_found / total_quotes_expected
                else:
                    hit_type = "MISS"; hit_score = 0.0

                successful_hits += hit_score

                if hit_type == "PERFECT":
                    print(f"{Fore.GREEN}{q_key} -> PERFECT HIT! ({quotes_found}/{total_quotes_expected}){Style.RESET_ALL}")
                elif hit_type == "PARTIAL":
                    print(f"{Fore.YELLOW}{q_key} -> PARTIAL HIT! ({quotes_found}/{total_quotes_expected}){Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}{q_key} -> MISS! (0/{total_quotes_expected}){Style.RESET_ALL}")

                if cosine_sims:
                    score_display = [f"{s:.3f}{'*' if j in chunks_with_hits else ' '}" for j, s in enumerate(cosine_sims)]
                    print(f"  scores: [{', '.join(score_display)}]  (* = chunk contained answer)")
                    if hit_scores:
                        print(f"  hit avg={sum(hit_scores)/len(hit_scores):.3f}  miss avg={sum(miss_scores)/len(miss_scores):.3f}" if miss_scores else f"  hit avg={sum(hit_scores)/len(hit_scores):.3f}")
                # coverage: what % of the doc did this query actually read?
                print(f"  coverage: {coverage_pct}% ({len(fetched_ids)}/{total_doc_chunks} chunks)")

                detail_rows.append({
                    "experiment_name":      EXPERIMENT_NAME,
                    "document":             pdf_filename,
                    "query_key":            q_key,
                    "csv_column":           csv_col,
                    "top_k":                TOP_K,
                    "window_size":          WINDOW_SIZE,
                    "total_doc_chunks":     total_doc_chunks,
                    "fetched_chunks":       len(fetched_ids),
                    "coverage_pct":         coverage_pct,
                    "retrieved_chunk_ids":  json.dumps(ret_ids),
                    "all_cosine_scores":    json.dumps(cosine_sims),
                    "avg_all_cosine":       safe_avg(cosine_sims),
                    "max_all_cosine":       round(max(cosine_sims), 4) if cosine_sims else "",
                    "hit_cosine_scores":    json.dumps(hit_scores),
                    "avg_hit_cosine":       safe_avg(hit_scores),
                    "miss_cosine_scores":   json.dumps(miss_scores),
                    "avg_miss_cosine":      safe_avg(miss_scores),
                    "quotes_expected":      total_quotes_expected,
                    "quotes_found":         quotes_found,
                    "hit_type":             hit_type,
                    "hit_score":            round(hit_score, 4),
                })

            # ==========================================
            # STEP 5.2: Combined query (q_rate + q_principal -> q2_penalty_quotes)
            # q_rate retrieves chunks about penalty rate/calculation method
            # q_principal retrieves chunks about the secured amount (istina)
            # These are often in the same paragraph but sometimes in different sections.
            # Merging gives better coverage than one long multi-topic query.
            # ==========================================
            correct_answer_string = row["q2_penalty_quotes"].strip()

            if correct_answer_string != "":
                total_questions += 1
                quotes_to_find = [clean_text(q) for q in correct_answer_string.split('|') if clean_text(q)]
                total_quotes_expected = len(quotes_to_find)

                sc_rate, sims_rate, ids_rate, fetched_rate = do_retrieve(
                    collection, "q_rate", QUERIES["q_rate"], precalculated_vectors.get("q_rate"),
                    pdf_filename, TOP_K, WINDOW_SIZE
                )
                sc_principal, sims_principal, ids_principal, fetched_principal = do_retrieve(
                    collection, "q_principal", QUERIES["q_principal"], precalculated_vectors.get("q_principal"),
                    pdf_filename, TOP_K, WINDOW_SIZE
                )

                all_fetched_ids = fetched_rate | fetched_principal
                call1_fetched.update(all_fetched_ids)  # q_rate + q_principal go to Call 1
                coverage_pct = round(len(all_fetched_ids) / total_doc_chunks * 100, 1) if total_doc_chunks > 0 else 0.0

                merged_super_chunks = sc_rate + sc_principal
                merged_cosine_sims = sims_rate + sims_principal
                merged_ids = ids_rate + ids_principal

                quotes_found, hit_scores, miss_scores, chunks_with_hits = find_hits(
                    merged_super_chunks, merged_cosine_sims, quotes_to_find
                )

                if quotes_found == total_quotes_expected:
                    hit_type = "PERFECT"; hit_score = 1.0
                elif quotes_found > 0:
                    hit_type = "PARTIAL"; hit_score = quotes_found / total_quotes_expected
                else:
                    hit_type = "MISS"; hit_score = 0.0

                successful_hits += hit_score

                if hit_type == "PERFECT":
                    print(f"{Fore.GREEN}q2_combined -> PERFECT HIT! ({quotes_found}/{total_quotes_expected}){Style.RESET_ALL}")
                elif hit_type == "PARTIAL":
                    print(f"{Fore.YELLOW}q2_combined -> PARTIAL HIT! ({quotes_found}/{total_quotes_expected}){Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}q2_combined -> MISS! (0/{total_quotes_expected}){Style.RESET_ALL}")

                if sims_rate:
                    rate_display = [f"{s:.3f}{'*' if j in chunks_with_hits else ' '}" for j, s in enumerate(sims_rate)]
                    print(f"  q_rate scores:      [{', '.join(rate_display)}]")
                if sims_principal:
                    pri_display = [f"{s:.3f}{'*' if (j + len(sims_rate)) in chunks_with_hits else ' '}" for j, s in enumerate(sims_principal)]
                    print(f"  q_principal scores: [{', '.join(pri_display)}]  (* = chunk contained answer)")
                print(f"  coverage: {coverage_pct}% ({len(all_fetched_ids)}/{total_doc_chunks} chunks)")

                detail_rows.append({
                    "experiment_name":          EXPERIMENT_NAME,
                    "document":                 pdf_filename,
                    "query_key":                "q2_combined",
                    "csv_column":               "q2_penalty_quotes",
                    "top_k":                    TOP_K,
                    "window_size":              WINDOW_SIZE,
                    "total_doc_chunks":         total_doc_chunks,
                    "fetched_chunks":           len(all_fetched_ids),
                    "coverage_pct":             coverage_pct,
                    "retrieved_chunk_ids":      json.dumps(merged_ids),
                    "all_cosine_scores":        json.dumps(merged_cosine_sims),
                    "rate_cosine_scores":       json.dumps(sims_rate),
                    "principal_cosine_scores":  json.dumps(sims_principal),
                    "avg_all_cosine":           safe_avg(merged_cosine_sims),
                    "max_all_cosine":           round(max(merged_cosine_sims), 4) if merged_cosine_sims else "",
                    "hit_cosine_scores":        json.dumps(hit_scores),
                    "avg_hit_cosine":           safe_avg(hit_scores),
                    "miss_cosine_scores":       json.dumps(miss_scores),
                    "avg_miss_cosine":          safe_avg(miss_scores),
                    "quotes_expected":          total_quotes_expected,
                    "quotes_found":             quotes_found,
                    "hit_type":                 hit_type,
                    "hit_score":                round(hit_score, 4),
                })

            # ==========================================
            # STEP 5.3: Combined query (q_outcome + q_reasoning + q_factors -> q3_moderation_quotes)
            # Three focused queries each retrieve chunks about a different aspect:
            # q_outcome  -> what happened (awarded/dismissed/returned)
            # q_reasoning -> why (court's argumentation)
            # q_factors  -> specific legal factors (dobré mravy, pomer k istine, škoda...)
            # Merging all three gives the full picture of the court's moderation analysis.
            # ==========================================
            correct_answer_string = row["q3_moderation_quotes"].strip()

            if correct_answer_string != "":
                total_questions += 1
                quotes_to_find = [clean_text(q) for q in correct_answer_string.split('|') if clean_text(q)]
                total_quotes_expected = len(quotes_to_find)

                sc_outcome, sims_outcome, ids_outcome, fetched_outcome = do_retrieve(
                    collection, "q_outcome", QUERIES["q_outcome"], precalculated_vectors.get("q_outcome"),
                    pdf_filename, TOP_K, WINDOW_SIZE
                )
                sc_reasoning, sims_reasoning, ids_reasoning, fetched_reasoning = do_retrieve(
                    collection, "q_reasoning", QUERIES["q_reasoning"], precalculated_vectors.get("q_reasoning"),
                    pdf_filename, TOP_K, WINDOW_SIZE
                )
                sc_factors, sims_factors, ids_factors, fetched_factors = do_retrieve(
                    collection, "q_factors", QUERIES["q_factors"], precalculated_vectors.get("q_factors"),
                    pdf_filename, TOP_K, WINDOW_SIZE
                )

                all_fetched_ids = fetched_outcome | fetched_reasoning | fetched_factors
                call2_fetched.update(all_fetched_ids)  # q_outcome + q_reasoning + q_factors go to Call 2
                coverage_pct = round(len(all_fetched_ids) / total_doc_chunks * 100, 1) if total_doc_chunks > 0 else 0.0

                merged_super_chunks = sc_outcome + sc_reasoning + sc_factors
                merged_cosine_sims = sims_outcome + sims_reasoning + sims_factors
                merged_ids = ids_outcome + ids_reasoning + ids_factors

                quotes_found, hit_scores, miss_scores, chunks_with_hits = find_hits(
                    merged_super_chunks, merged_cosine_sims, quotes_to_find
                )

                if quotes_found == total_quotes_expected:
                    hit_type = "PERFECT"; hit_score = 1.0
                elif quotes_found > 0:
                    hit_type = "PARTIAL"; hit_score = quotes_found / total_quotes_expected
                else:
                    hit_type = "MISS"; hit_score = 0.0

                successful_hits += hit_score

                if hit_type == "PERFECT":
                    print(f"{Fore.GREEN}q3_combined -> PERFECT HIT! ({quotes_found}/{total_quotes_expected}){Style.RESET_ALL}")
                elif hit_type == "PARTIAL":
                    print(f"{Fore.YELLOW}q3_combined -> PARTIAL HIT! ({quotes_found}/{total_quotes_expected}){Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}q3_combined -> MISS! (0/{total_quotes_expected}){Style.RESET_ALL}")

                if sims_outcome:
                    out_display = [f"{s:.3f}{'*' if j in chunks_with_hits else ' '}" for j, s in enumerate(sims_outcome)]
                    print(f"  q_outcome scores:   [{', '.join(out_display)}]")
                if sims_reasoning:
                    offset_r = len(sims_outcome)
                    rea_display = [f"{s:.3f}{'*' if (j + offset_r) in chunks_with_hits else ' '}" for j, s in enumerate(sims_reasoning)]
                    print(f"  q_reasoning scores: [{', '.join(rea_display)}]")
                if sims_factors:
                    offset_f = len(sims_outcome) + len(sims_reasoning)
                    fac_display = [f"{s:.3f}{'*' if (j + offset_f) in chunks_with_hits else ' '}" for j, s in enumerate(sims_factors)]
                    print(f"  q_factors scores:   [{', '.join(fac_display)}]  (* = chunk contained answer)")
                print(f"  coverage: {coverage_pct}% ({len(all_fetched_ids)}/{total_doc_chunks} chunks)")

                detail_rows.append({
                    "experiment_name":          EXPERIMENT_NAME,
                    "document":                 pdf_filename,
                    "query_key":                "q3_combined",
                    "csv_column":               "q3_moderation_quotes",
                    "top_k":                    TOP_K,
                    "window_size":              WINDOW_SIZE,
                    "total_doc_chunks":         total_doc_chunks,
                    "fetched_chunks":           len(all_fetched_ids),
                    "coverage_pct":             coverage_pct,
                    "retrieved_chunk_ids":      json.dumps(merged_ids),
                    "all_cosine_scores":        json.dumps(merged_cosine_sims),
                    "outcome_cosine_scores":    json.dumps(sims_outcome),
                    "reasoning_cosine_scores":  json.dumps(sims_reasoning),
                    "factors_cosine_scores":    json.dumps(sims_factors),
                    "avg_all_cosine":           safe_avg(merged_cosine_sims),
                    "max_all_cosine":           round(max(merged_cosine_sims), 4) if merged_cosine_sims else "",
                    "hit_cosine_scores":        json.dumps(hit_scores),
                    "avg_hit_cosine":           safe_avg(hit_scores),
                    "miss_cosine_scores":       json.dumps(miss_scores),
                    "avg_miss_cosine":          safe_avg(miss_scores),
                    "quotes_expected":          total_quotes_expected,
                    "quotes_found":             quotes_found,
                    "hit_type":                 hit_type,
                    "hit_score":                round(hit_score, 4),
                })

            # ==========================================
            # STEP 6: Per-call coverage
            # This is what actually matters - how much of the document each LLM call reads.
            # The per-query coverage above is useful for debugging but this is the real metric.
            # ==========================================
            call1_cov = round(len(call1_fetched) / total_doc_chunks * 100, 1) if total_doc_chunks > 0 else 0.0
            call2_cov = round(len(call2_fetched) / total_doc_chunks * 100, 1) if total_doc_chunks > 0 else 0.0
            total_fetched = call1_fetched | call2_fetched
            total_cov = round(len(total_fetched) / total_doc_chunks * 100, 1) if total_doc_chunks > 0 else 0.0

            print(f"  --- LLM call coverage for {pdf_filename} ---")
            print(f"  Call 1 (breach+rate+principal): {call1_cov}% ({len(call1_fetched)}/{total_doc_chunks})")
            print(f"  Call 2 (outcome+reasoning+factors): {call2_cov}% ({len(call2_fetched)}/{total_doc_chunks})")
            print(f"  Total unique chunks read: {total_cov}% ({len(total_fetched)}/{total_doc_chunks})")

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

        print("\n" + "=" * 50)
        print(f"=== FINAL SCORE: {EXPERIMENT_NAME} ===")
        print("=" * 50)
        print(f"Total Questions asked:            {total_questions}")
        print(f"Perfect/Partial Hits (Weighted):  {successful_hits:.2f}")
        print(f"Overall Quote Recall Rate:        {hit_rate:.2f}%")
        print(f"Avg Coverage (per query group):   {avg_coverage}%")
        print(f"Avg Call 1 Coverage:              {avg_call1_cov}%")
        print(f"Avg Call 2 Coverage:              {avg_call2_cov}%")
        print(f"Avg Total Coverage (all 6 queries): {avg_total_cov}%")
        print("=" * 50)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---------------------------------------------------------
        # SAVE 1: SUMMARY CSV
        # One row per experiment run. Good for high-level comparison between experiments.
        # avg_coverage_pct is the key tradeoff metric: recall vs how much of the doc we read
        # ---------------------------------------------------------
        summary_exists = os.path.isfile(RESULTS_SUMMARY_CSV)
        with open(RESULTS_SUMMARY_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            if not summary_exists:
                writer.writerow([
                    "timestamp", "experiment_name", "top_k", "window_size",
                    "total_questions", "weighted_hits", "recall_rate_percent",
                    "avg_coverage_pct", "avg_call1_cov", "avg_call2_cov", "avg_total_cov"
                ])
            writer.writerow([
                current_time, EXPERIMENT_NAME, TOP_K, WINDOW_SIZE,
                total_questions, f"{successful_hits:.2f}", f"{hit_rate:.2f}",
                f"{avg_coverage}", f"{avg_call1_cov}", f"{avg_call2_cov}", f"{avg_total_cov}"
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