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

# ==========================================
# FIX: Mac M2 freezing problem!
# These lines stop my Mac M2 from freezing. It turns off parallel processing and threads.
# I also disable Chroma telemetry so it doesn't hang my script randomly.
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# NEW IMPORT: I will call the model directly myself, not through ChromaDB black box.
from sentence_transformers import SentenceTransformer

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

QUERIES = {
    # Channel A
    "q_context": "Čo bolo predmetom zmluvy medzi stranami? Akú povinnosť si dlžník prevzal a ako ju porušil?",

    # Channel B
    # "q_penalty": "Aká je výška, sadzba a mena zmluvnej pokuty? Z akej sumy sa počíta a za aké porušenie povinnosti bola dohodnutá?", - (in slovak so it would be understandable) istina zavazku ktory pokuta zabezpecuje — je obcas uvedena len v kontexte zmluvy, nie v odseku o pokute. Query "Z akej sumy sa počíta" to vacsinou zachytila, ale nie vzdy
    "q_penalty": "Aká je výška, sadzba a mena zmluvnej pokuty? Z akej sumy sa počíta a za aké porušenie povinnosti bola dohodnutá? Aká je výška hlavného záväzku (istiny) ktorý pokuta zabezpečuje?",

    # Channel C1 - vysledok konania (what happened to the penalty)
    # This explicitly covers ALL possible outcomes: §301 reduction, dismissal for invalidity,
    # procedural dismissal, case returned for retrial - not just the §301 moderation case!
    "q_outcome": "Aký bol výsledok konania o zmluvnej pokute? Bola priznaná, znížená alebo zamietnutá? Uplatnil súd moderačné oprávnenie podľa § 301 Obchodného zákonníka? Alebo bola pokuta zamietnutá pre neplatnosť zmluvy, procesný dôvod, alebo vrátená na ďalšie konanie?",

    # Channel C2 - argumenty súdu (why)
    # This retrieves the reasoning chunks regardless of what the outcome was.
    "q_reasoning": "Aké dôvody uviedol súd pri posudzovaní zmluvnej pokuty? Napríklad výška škody, rozpor s dobrými mravmi, pomer k istine, správanie dlžníka, zabezpečovacia funkcia, kumulácia s úrokom z omeškania?",
}

# This tells the evaluator which CSV column each query maps to.
# q_outcome and q_reasoning SHARE the same CSV column - their retrieved chunks are
# merged into one pool before checking for golden quotes. See Step 5.2 for details.
QUERY_TO_CSV_COLUMN = {
    "q_context":   "q1_context_quotes",
    "q_penalty":   "q2_penalty_quotes",
    "q_outcome":   "q3_moderation_quotes",   # ─┐ evaluated
    "q_reasoning": "q3_moderation_quotes",   # ─┘ together
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
    print("Loading mE5 model explicitly (CPU mode)...")

    # Load the model myself and force it to run on CPU so it doesn't crash the Apple GPU.
    # model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

    # FIX 2: BIG OPTIMIZATION!
    # I am asking the EXACT same questions for every document.
    # So I will translate them into vectors only ONCE here, before the loop starts!
    print("Pre-calculating query vectors to save time...")
    precalculated_vectors = {}
    for q_key, q_text in QUERIES.items():
        # BACHELOR THESIS TRICK: The mE5 model needs the question to start with word "query: " to work good.
        me5_formatted_query = f"query: {q_text}"
        # FIX: normalize_embeddings=True - must match how documents were indexed in vector_store.py!
        # If documents are normalized and queries are not, cosine similarity scores are wrong.
        precalculated_vectors[q_key] = model.encode(
            [me5_formatted_query],
            normalize_embeddings=True
        ).tolist()

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

            # ==========================================
            # STEP 5.1: Single queries (q_context and q_penalty)
            # These map 1:1 to a CSV column so evaluation is straightforward.
            # ==========================================
            for q_key in ["q_context", "q_penalty"]:
                csv_col = QUERY_TO_CSV_COLUMN[q_key]
                correct_answer_string = row[csv_col].strip()

                if correct_answer_string == "":
                    continue

                total_questions += 1
                quotes_to_find = [clean_text(q) for q in correct_answer_string.split('|') if clean_text(q)]

                super_chunks, cosine_sims, ret_ids, fetched_ids = retrieve_super_chunks(
                    collection, precalculated_vectors[q_key], pdf_filename, TOP_K, WINDOW_SIZE
                )

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
            # STEP 5.2: Combined query (q_outcome + q_reasoning -> q3_moderation_quotes)
            # Both queries retrieve chunks, their super_chunks are MERGED into one pool,
            # then we check if q3 golden quotes are found anywhere in the combined pool.
            #
            # Why merge instead of evaluate separately?
            # q_outcome finds chunks about what happened (awarded/dismissed/returned)
            # q_reasoning finds chunks about why (dobré mravy, škoda, pomer k istine)
            # For cases like 8Cob/64/2011 where §301 was NOT applied, the relevant
            # chunks are split across both retrieval types. Merging gives the full picture.
            # ==========================================
            correct_answer_string = row["q3_moderation_quotes"].strip()

            if correct_answer_string != "":
                total_questions += 1
                quotes_to_find = [clean_text(q) for q in correct_answer_string.split('|') if clean_text(q)]
                total_quotes_expected = len(quotes_to_find)

                # Run both queries separately
                sc_outcome, sims_outcome, ids_outcome, fetched_outcome = retrieve_super_chunks(
                    collection, precalculated_vectors["q_outcome"], pdf_filename, TOP_K, WINDOW_SIZE
                )
                sc_reasoning, sims_reasoning, ids_reasoning, fetched_reasoning = retrieve_super_chunks(
                    collection, precalculated_vectors["q_reasoning"], pdf_filename, TOP_K, WINDOW_SIZE
                )

                # union of both fetched sets = all unique chunks read by both C queries together
                all_fetched_ids = fetched_outcome | fetched_reasoning
                coverage_pct = round(len(all_fetched_ids) / total_doc_chunks * 100, 1) if total_doc_chunks > 0 else 0.0

                # Merge super_chunk pools - now we have up to 2*TOP_K super_chunks to search
                # We keep the cosine scores separate per query so we can analyze them in notebook
                merged_super_chunks = sc_outcome + sc_reasoning
                merged_cosine_sims = sims_outcome + sims_reasoning
                merged_ids = ids_outcome + ids_reasoning

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

                # Print scores for each sub-query separately so i can see which one helped
                if sims_outcome:
                    out_display = [f"{s:.3f}{'*' if j in chunks_with_hits else ' '}" for j, s in enumerate(sims_outcome)]
                    print(f"  q_outcome scores:   [{', '.join(out_display)}]")
                if sims_reasoning:
                    # offset j because merged list starts with outcome chunks
                    rea_display = [f"{s:.3f}{'*' if (j + len(sims_outcome)) in chunks_with_hits else ' '}" for j, s in enumerate(sims_reasoning)]
                    print(f"  q_reasoning scores: [{', '.join(rea_display)}]  (* = chunk contained answer)")
                print(f"  coverage: {coverage_pct}% ({len(all_fetched_ids)}/{total_doc_chunks} chunks)")

                # Save one combined row - store both query scores separately for notebook analysis
                detail_rows.append({
                    "experiment_name":          EXPERIMENT_NAME,
                    "document":                 pdf_filename,
                    "query_key":                "q3_combined",
                    "csv_column":               "q3_moderation_quotes",
                    "top_k":                    TOP_K,
                    "window_size":              WINDOW_SIZE,
                    "total_doc_chunks":          total_doc_chunks,
                    "fetched_chunks":             len(all_fetched_ids),
                    "coverage_pct":               coverage_pct,
                    "retrieved_chunk_ids":        json.dumps(merged_ids),
                    # outcome and reasoning scores stored separately for analysis
                    "all_cosine_scores":          json.dumps(merged_cosine_sims),
                    "outcome_cosine_scores":       json.dumps(sims_outcome),
                    "reasoning_cosine_scores":     json.dumps(sims_reasoning),
                    "avg_all_cosine":              safe_avg(merged_cosine_sims),
                    "max_all_cosine":              round(max(merged_cosine_sims), 4) if merged_cosine_sims else "",
                    "hit_cosine_scores":           json.dumps(hit_scores),
                    "avg_hit_cosine":              safe_avg(hit_scores),
                    "miss_cosine_scores":          json.dumps(miss_scores),
                    "avg_miss_cosine":             safe_avg(miss_scores),
                    "quotes_expected":             total_quotes_expected,
                    "quotes_found":                quotes_found,
                    "hit_type":                    hit_type,
                    "hit_score":                   round(hit_score, 4),
                })

    # ==========================================
    # STEP 7: Calculate, print AND SAVE final score
    # ==========================================
    if total_questions > 0:
        hit_rate = (successful_hits / total_questions) * 100

        # avg coverage across all rows - key thesis metric next to recall
        coverage_rows = [r for r in detail_rows if 'coverage_pct' in r]
        avg_coverage = round(sum(r['coverage_pct'] for r in coverage_rows) / len(coverage_rows), 1) if coverage_rows else 0.0

        print("\n" + "=" * 50)
        print(f"=== FINAL SCORE: {EXPERIMENT_NAME} ===")
        print("=" * 50)
        print(f"Total Questions asked:            {total_questions}")
        print(f"Perfect/Partial Hits (Weighted):  {successful_hits:.2f}")
        print(f"Overall Quote Recall Rate:        {hit_rate:.2f}%")
        print(f"Avg Coverage (chunks read):       {avg_coverage}%")
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
                    "total_questions", "weighted_hits", "recall_rate_percent", "avg_coverage_pct"
                ])
            writer.writerow([
                current_time, EXPERIMENT_NAME, TOP_K, WINDOW_SIZE,
                total_questions, f"{successful_hits:.2f}", f"{hit_rate:.2f}", f"{avg_coverage}"
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