import os
import csv
import chromadb
from chromadb.utils import embedding_functions
from colorama import Fore, Style, init

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

    with open(CSV_PATH, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=';')        # pythons modul DictReader xpects "," between sections but excel filled with lawyer saved it with ; what did problems

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

                # Search in the database!
                # IMPORTANT: We use 'where' to search ONLY inside this one specific PDF
                results = collection.query(
                    query_texts=[q_text],
                    n_results=TOP_K,
                    where={"source_file": pdf_filename}
                )

                # Get the text of the chunks the database found
                # retrieved_chunks = results['documents'][0]
                retrieved_chunks = results['documents'][0] if len(results['documents']) > 0 else []

                # ==========================================
                # STEP 6: Check if the database was correct
                # ==========================================
                hit = False  # Assume we failed until we find a match

                # Look at each chunk the database returned
                for chunk in retrieved_chunks:
                    # Check if any of our expected quotes are hidden inside this chunk
                    for quote in quotes_to_find:
                        if quote in chunk:
                            hit = True
                            break  # Stop searching this chunk, we already found it!

                    if hit == True:
                        break  # Stop checking other chunks, we already have a point!

                # Score update
                init(autoreset=True)
                if hit:
                    successful_hits += 1
                    print(f"{Fore.GREEN}{q_key} -> HIT!{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}{q_key} -> MISS! (Database returned wrong chunks){Style.RESET_ALL}")

    # ==========================================
    # STEP 7: Calculate and print final score
    # ==========================================
    if total_questions > 0:
        hit_rate = (successful_hits / total_questions) * 100

        print("\n" + "=" * 40)
        print("=== FINAL SCORE ===")
        print("=" * 40)
        print(f"Total Questions asked: {total_questions}")
        print(f"Correct Answers (Hits): {successful_hits}")
        print(f"Hit-Rate (TOP-{TOP_K}): {hit_rate:.2f}%")
        print("=" * 40)
    else:
        print("\nNo questions to evaluate. Is the CSV empty?")


if __name__ == "__main__":
    main()