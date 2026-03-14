import os
import csv
import chromadb
import datetime
import re
from colorama import Fore, Style, init

# ==========================================
# FIX: Mac M2 freezing problem!
# These lines  stop my Mac M2 from freezing. It turns off parallel processing and threads.
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


# ==========================================
# FIX: TEXT CLEANING SO PYTHON DOESNT FAIL STUPIDLY
# If my PDF has "500 \n eur" but Excel has "500 eur", normal Python says "MISS!".
# This function removes all new lines and extra spaces so matching is 100% bulletproof.
# ==========================================
def clean_text(text):
    if not text:
        return ""
    # Replace newlines with spaces and squish multiple spaces into one
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r'\s+', ' ', text).strip()


def main():
    # ==========================================
    # FIX: BIG BRAIN MOMENT!
    # I MUST initialize colorama here ONLY ONCE at the start!
    # Omg I had it inside the loop before and it was restarting my terminal 30 times.
    # It was completely freezing my Mac and I couldn't find the error for 5 hours! :D
    # ==========================================
    init(autoreset=True)
    print("=== STARTING EVALUATION (mE5 with Context Window Expansion) ===")

    # ==========================================
    # STEP 3: Connect to the database and Model
    # ==========================================
    print("Loading mE5 model explicitly (CPU mode)...")

    # Load the model myself and force it to run on CPU so it doesn't crash the Apple GPU.
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

    # FIX 2: BIG OPTIMIZATION!
    # I am asking the EXACT same 3 questions for every document.
    # So I will translate them into vectors only ONCE here, before the loop starts!
    print("Pre-calculating query vectors to save time...")
    precalculated_vectors = {}
    for q_key, q_text in QUERIES.items():
        # BACHELOR THESIS TRICK: The mE5 model needs the question to start with word "query: " to work good.
        me5_formatted_query = f"query: {q_text}"
        precalculated_vectors[q_key] = model.encode([me5_formatted_query]).tolist()

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)

    # Get the database table WITHOUT embedding function because I made vectors manually.
    collection = chroma_client.get_collection(name="legal_decisions_me5")

    # Variables to track my score
    total_questions = 0
    successful_hits = 0

    # ==========================================
    # STEP 4: Read the Golden Dataset (CSV)
    # ==========================================
    print("Reading Golden Dataset...")

    # Pythons modul DictReader expects "," between sections but my lawyer excel saved it with ";"
    with open(CSV_PATH, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=';')

        # Loop through each document in my CSV file
        for row in reader:
            doc_name = row["document_name"]

            # Fix filename: In DB I saved them as .pdf, but in CSV they have .json
            pdf_filename = doc_name.replace(".json", ".pdf")
            print(f"\n--- Testing document: {pdf_filename} ---")

            # ==========================================
            # STEP 5: Ask questions for this document
            # ==========================================
            for q_key, q_text in QUERIES.items():

                # Get the correct answer (quotes) from Excel
                correct_answer_string = row[q_key].strip()

                # If Excel cell is empty, skip this question
                if correct_answer_string == "":
                    continue

                total_questions += 1

                # Split quotes if I used the pipe '|' symbol and clean them with my smart function
                quotes_to_find = [clean_text(quote) for quote in correct_answer_string.split('|') if clean_text(quote)]

                # Use the vector I already calculated in STEP 3! Extremely fast!
                query_vector = precalculated_vectors[q_key]

                # I ask the database to find the best SMALL chunks
                # IMPORTANT: I use 'where' to search ONLY inside this one specific PDF
                results = collection.query(
                    query_embeddings=query_vector,
                    n_results=TOP_K,
                    where={"source_file": pdf_filename}
                )

                # ==========================================
                # STEP 5.1: BIG BRAIN FIX! (Small-to-Big Retrieval / Window Expansion)
                # The exact quote might be cut in half between two chunks!
                # So I don't use just the one small chunk.
                # I ask database for Chunk-1, Chunk, and Chunk+1 and glue them together!
                # This fixes the boundary problem and gives LLM a huge context to read.
                # ==========================================
                super_chunks = []

                # Check if we found anything
                if len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                    retrieved_ids = results['ids'][0]
                    retrieved_metadatas = results['metadatas'][0]

                    for i in range(len(retrieved_ids)):
                        c_idx = retrieved_metadatas[i]['chunk_index']

                        # Build IDs for previous, current, and next chunk (3 chunks total)
                        # If I want more context for LLM later, I can just add c_idx-2 and c_idx+2 here!
                        window_ids = [
                            f"{pdf_filename}_chunk_{c_idx - 1}",
                            f"{pdf_filename}_chunk_{c_idx}",
                            f"{pdf_filename}_chunk_{c_idx + 1}"
                        ]

                        # Fetch them directly from DB! (Chroma smartly ignores IDs that don't exist, e.g., chunk_-1)
                        db_fetch = collection.get(ids=window_ids)

                        # Glue them together into one massive Super Chunk and clean the text
                        combined_text = " ".join(db_fetch['documents'])
                        super_chunks.append(clean_text(combined_text))

                # ==========================================
                # STEP 6: Check if the database was correct (Quote Recall)
                # Now we search inside our massive Super Chunks!
                # ==========================================
                # NEW LOGIC: Instead of just finding ONE quote, I want to check
                # how many of the REQUIRED quotes from Excel were actually found in my big Super Chunks!
                quotes_found_count = 0
                total_quotes_expected = len(quotes_to_find)

                for quote in quotes_to_find:
                    quote_is_found = False

                    for super_chunk in super_chunks:
                        if quote in super_chunk:
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
        hit_rate = (successful_hits / total_questions) * 100

        print("\n" + "=" * 50)
        print("=== FINAL SCORE (WINDOW EXPANSION RECALL) ===")
        print("=" * 50)
        print(f"Total Questions asked: {total_questions}")
        print(f"Perfect/Partial Hits (Weighted): {successful_hits:.2f}")
        print(f"Overall Quote Recall Rate: {hit_rate:.2f}%")
        print("=" * 50)

        # ---------------------------------------------------------
        # SAVE TO EXCEL (Experiment Tracking)
        # ---------------------------------------------------------
        EXPERIMENT_NAME = "mE5_expanded_window_dense"  # We changed the logic, so we change the name!
        RESULTS_CSV_PATH = os.path.join(EVAL_DIR, "experiment_results.csv")

        file_exists = os.path.isfile(RESULTS_CSV_PATH)

        with open(RESULTS_CSV_PATH, mode="a", newline="", encoding="utf-8") as res_file:
            writer = csv.writer(res_file, delimiter=";")

            # Write header only if it's a completely new file
            if not file_exists:
                writer.writerow(["timestamp", "experiment_name", "top_k", "total_questions", "weighted_hits",
                                 "recall_rate_percent"])

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(
                [current_time, EXPERIMENT_NAME, TOP_K, total_questions, f"{successful_hits:.2f}", f"{hit_rate:.2f}"])

        print(f"Experiment saved to: {RESULTS_CSV_PATH}")

    else:
        print("\nNo questions to evaluate. Is the CSV empty?")


if __name__ == "__main__":
    main()