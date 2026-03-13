import os
import csv
import chromadb
from chromadb.utils import embedding_functions
from colorama import Fore, Style, init
import datetime

# ==========================================
# FIX: Mac M2 freezing problem!
# These lines will stop my Mac M2 from freezing. It turns off parallel processing and threads.
# I also disable Chroma telemetry so it doesn't hang my script randomly.
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# ==========================================
# STEP 1: Setup paths and basic variables
# ==========================================
DB_DIR = "data/04_vectorstore"
EVAL_DIR = "data/05_retrieval_evaluation"
CSV_PATH = os.path.join(EVAL_DIR, "golden_dataset_template.csv")

# How many chunks do we want the database to return for each question?
TOP_K = 5

# ==========================================
# STEP 2: Define our questions
# ==========================================
# These are the exact questions we will ask the vector database
QUERIES = {
    "q1_context_quotes": "Kto je žalobca a žalovaný? Na základe akej zmluvy vznikol spor a čo je predmetom sporu?",
    "q2_penalty_quotes": "Aká je výška, sadzba a mena zmluvnej pokuty? Z akej sumy sa počíta a za aké porušenie povinnosti bola dohodnutá?",
    "q3_moderation_quotes": "Ako súd posúdil primeranosť zmluvnej pokuty podľa § 301 Obchodného zákonníka? Znížil súd zmluvnú pokutu a z akých dôvodov (napríklad pre rozpor s dobrými mravmi alebo výšku škody)?"
}


def main():
    # ==========================================
    # FIX: BIG BRAIN MOMENT!
    # I MUST initialize colorama here ONLY ONCE at the start!
    # Omg I had it inside the loop before and it was restarting my terminal 30 times.
    # It was completely freezing my Mac and I couldn't find the error for 5 hours! :D
    # ==========================================
    init(autoreset=True)

    print("=== STARTING EVALUATION (mE5 Baseline) ===")

    # ==========================================
    # STEP 3: Connect to the database
    # ==========================================
    print("Connecting to ChromaDB...")

    # We must use the exact same embedding model that we used to build the DB!
    me5_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small"
    )

    # Open the database from our disk
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(
        name="legal_decisions_me5",
        embedding_function=me5_model
    )

    # Variables to track our score
    total_questions = 0
    successful_hits = 0

    # ==========================================
    # STEP 4: Read the Golden Dataset (CSV)
    # ==========================================
    print("Reading Golden Dataset...")

    # Pythons modul DictReader expects "," between sections but excel saved it with ";"
    with open(CSV_PATH, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=';')

        # Loop through each document in our CSV file
        for row in reader:
            doc_name = row["document_name"]

            # Fix filename: In DB we saved them as .pdf, but in CSV they have .json
            pdf_filename = doc_name.replace(".json", ".pdf")
            print(f"\n--- Testing document: {pdf_filename} ---")

            # ==========================================
            # STEP 5: Ask questions for this document
            # ==========================================
            # Loop through our 3 questions
            for q_key, q_text in QUERIES.items():

                # Get the correct answer (quotes) from Excel
                correct_answer_string = row[q_key].strip()

                # If Excel cell is empty, skip this question
                if correct_answer_string == "":
                    continue

                total_questions += 1

                # Split quotes if we used the pipe '|' symbol
                # Example: "quote 1 | quote 2" -> ["quote 1", "quote 2"]
                quotes_to_find = correct_answer_string.split('|')

                # Clean up empty spaces around quotes
                quotes_to_find = [quote.strip() for quote in quotes_to_find]

                # BACHELOR THESIS TRICK: Adding "query: " makes mE5 model much smarter!
                me5_query = f"query: {q_text}"

                # Search in the database!
                # IMPORTANT: We use 'where' to search ONLY inside this one specific PDF
                results = collection.query(
                    query_texts=[me5_query],
                    n_results=TOP_K,
                    where={"source_file": pdf_filename}
                )

                # Get the text of the chunks the database found safely
                retrieved_chunks = results['documents'][0] if len(results['documents']) > 0 else []

                # ==========================================
                # STEP 6: Check if the database was correct (Quote Recall)
                # ==========================================
                # NEW LOGIC: Instead of just finding ONE quote, I want to check
                # how many of the REQUIRED quotes from Excel were actually found in the 5 chunks!
                quotes_found_count = 0
                total_quotes_expected = len(quotes_to_find)

                # Check each quote separately
                for quote in quotes_to_find:
                    quote_is_found = False

                    # Look for this specific quote in ALL 5 retrieved chunks
                    for chunk in retrieved_chunks:
                        if quote in chunk:
                            quote_is_found = True
                            break  # Found this quote! Move to the next quote.

                    if quote_is_found:
                        quotes_found_count += 1

                # Score update with colors
                if quotes_found_count > 0:
                    # If I found ALL expected quotes, it's a PERFECT HIT!
                    if quotes_found_count == total_quotes_expected:
                        successful_hits += 1
                        print(
                            f"{Fore.GREEN}{q_key} -> PERFECT HIT! (Found {quotes_found_count}/{total_quotes_expected} required quotes){Style.RESET_ALL}")
                    # If I found only some, it's a PARTIAL HIT
                    else:
                        # We can count this as a half-hit, or just track it for info
                        successful_hits += (quotes_found_count / total_quotes_expected)
                        print(
                            f"{Fore.YELLOW}{q_key} -> PARTIAL HIT! (Found {quotes_found_count}/{total_quotes_expected} required quotes){Style.RESET_ALL}")
                else:
                    print(
                        f"{Fore.RED}{q_key} -> MISS! (Found 0/{total_quotes_expected} required quotes){Style.RESET_ALL}")

            # ==========================================
            # STEP 7: Calculate, print AND SAVE final score
            # ==========================================
            if total_questions > 0:
                # Hit rate is now mathematically more precise (Recall)
                hit_rate = (successful_hits / total_questions) * 100

                print("\n" + "=" * 50)
                print("=== FINAL SCORE (QUOTE RECALL) ===")
                print("=" * 50)
                print(f"Total Questions asked: {total_questions}")
                print(f"Perfect/Partial Hits (Weighted): {successful_hits:.2f}")
                print(f"Overall Quote Recall Rate: {hit_rate:.2f}%")
                print("=" * 50)

                # ---------------------------------------------------------
                # NEW TRACKING LOGIC: Save results to CSV for Bachelor Thesis
                # ---------------------------------------------------------
                EXPERIMENT_NAME = "mE5_baseline_dense"  # Change this name when testing other models!
                RESULTS_CSV_PATH = os.path.join(EVAL_DIR, "experiment_results.csv")

                # Check if file exists so we know if we need to write the header row
                file_exists = os.path.isfile(RESULTS_CSV_PATH)

                with open(RESULTS_CSV_PATH, mode="a", newline="", encoding="utf-8") as res_file:
                    writer = csv.writer(res_file, delimiter=";")

                    # Write header only if it's a completely new file
                    if not file_exists:
                        writer.writerow(
                            ["timestamp", "experiment_name", "top_k", "total_questions", "weighted_hits",
                             "recall_rate_percent"])

                    # Append the result of THIS run
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([
                        current_time,
                        EXPERIMENT_NAME,
                        TOP_K,
                        total_questions,
                        f"{successful_hits:.2f}",
                        f"{hit_rate:.2f}"
                    ])

                print(f"Experiment saved to: {RESULTS_CSV_PATH}")

            else:
                print("\nNo questions to evaluate. Is the CSV empty?")


if __name__ == "__main__":
    main()