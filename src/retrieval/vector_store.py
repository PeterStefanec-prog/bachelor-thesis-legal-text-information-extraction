import os
import json
import glob
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# --- 1. PATH SETUP ---
# Important: Run this script from the ROOT project folder
INPUT_DIR = "data/03_chunked_docs"
DB_DIR = "data/04_vectorstore"  # Here ChromaDB will save all its files

# Make DB dir if it doesn't exist
os.makedirs(DB_DIR, exist_ok=True)

# --- 2. DATABASE & MODEL SETUP ---
print("Initializing ChromaDB and loading the embedding model (this might take a minute the first time)...")

# PersistentClient means it will save the database to my disk (not just in RAM)
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# I am using mE5-small. It's great for Slovak language and completely free/local!
# Chroma will download it automatically from HuggingFace.
me5_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)

# Create a "table" (collection) in my database
# get_or_create is safe - if it already exists, it just opens it
collection = chroma_client.get_or_create_collection(
    name="legal_decisions_me5",
    embedding_function=me5_embedding_function,
    metadata={"description": "Chunks of Slovak legal decisions optimized for mE5"}
)


# --- 3. MAIN PIPELINE ---
def main():
    # I only want to load the ME5 chunks for this specific database
    json_files = glob.glob(os.path.join(INPUT_DIR, "*_chunks_ME5.json"))

    if not json_files:
        print(f"Error: No ME5 chunk files found in {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} document files. Starting vectorization and insertion...\n")

    for file_path in tqdm(json_files, desc="Adding docs to ChromaDB"):
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            continue

        # Chroma needs 3 lists: texts, metadatas, and unique IDs
        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            text = chunk["page_content"]
            meta = chunk["metadata"]

            # Chroma requires metadata values to be strings, ints, or floats (no complex dicts)
            # So I make sure my metadata is clean
            clean_meta = {
                "source_file": str(meta.get("source_file", "unknown")),
                "case_id": str(meta.get("case_id", "unknown")),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "strategy": str(meta.get("strategy", "unknown"))
            }

            # Create a unique ID for each chunk (e.g. "KS_Bratislava_1CoZm.pdf_chunk_5")
            chunk_id = f"{clean_meta['source_file']}_chunk_{clean_meta['chunk_index']}"

            documents.append(text)
            metadatas.append(clean_meta)
            ids.append(chunk_id)

        # Insert everything into the database (batch processing)
        # Chroma automatically calculates the vectors because of my embedding_function!
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            # If a document is already in the DB, it might throw an error or skip.
            # I can use upsert() instead of add() if I want to overwrite existing ones.
            print(f"Warning: Could not add chunks from {file_path}. Error: {e}")

    # --- 4. FINAL REPORT ---
    # Check how many chunks are actually inside the database now
    total_chunks_in_db = collection.count()
    print("\n=== VECTOR STORE CREATION COMPLETED ===")
    print(f"Total vectors (chunks) currently stored in database: {total_chunks_in_db}")
    print("Database is safely saved on your disk.")


if __name__ == "__main__":
    main()