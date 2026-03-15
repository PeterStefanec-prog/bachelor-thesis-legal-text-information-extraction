import os
import json
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- 1. PATH SETUP ---
# Important: Run this script from the ROOT project folder
INPUT_DIR = "data/03_chunked_docs"
DB_DIR = "data/04_vectorstore"  # Here ChromaDB will save all its files

os.makedirs(DB_DIR, exist_ok=True)

# --- 2. MODEL SETUP ---
# I load the model manually instead of letting ChromaDB do it internally.
# Why: ChromaDB's built-in SentenceTransformerEmbeddingFunction is a black box -
# i can't control normalize_embeddings and i can't add the 'passage: ' prefix
# only for embedding while keeping the raw text clean for LLM later.
print("Loading mE5 model manually...")
model = SentenceTransformer("intfloat/multilingual-e5-small")

# --- 3. DATABASE SETUP ---
print("Initializing ChromaDB...")

# PersistentClient means it will save the database to my disk (not just in RAM)
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# FIX: hnsw:space must be "cosine" so ChromaDB uses cosine similarity under the hood.
# Default is "l2" (Euclidean distance) which gives wrong results for text embeddings.
# FIX: i do NOT pass embedding_function here anymore. i will compute embeddings manually
# and pass them directly via embeddings= parameter in upsert().
# This gives me full control over how vectors are computed.
collection = chroma_client.get_or_create_collection(
    name="legal_decisions_me5",
    metadata={
        "description": "Chunks of Slovak legal decisions, mE5-small embeddings, cosine space",
        "hnsw:space": "cosine",
    }
)


# --- 4. MAIN PIPELINE ---
def main():
    # I only want to load the ME5 chunks for this specific database
    json_files = glob.glob(os.path.join(INPUT_DIR, "*_chunks_ME5.json"))

    if not json_files:
        print(f"Error: No ME5 chunk files found in {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} document files. Starting vectorization and insertion...\n")

    total_inserted = 0
    failed_files = 0

    for file_path in tqdm(json_files, desc="Embedding and inserting"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if not chunks:
                continue

            # --- Build the three lists ChromaDB needs ---
            raw_texts = []       # clean text WITHOUT prefix -> this is what gets stored in DB
            texts_to_embed = []  # text WITH 'passage: ' prefix -> only used for computing vectors
            metadatas = []
            ids = []

            for chunk in chunks:
                text = chunk["page_content"]
                meta = chunk["metadata"]

                raw_texts.append(text)

                # mE5 needs 'passage: ' prefix during indexing so it knows it's encoding
                # a document and not a query. The matching 'query: ' prefix is added in
                # evaluate_retrieval.py. Keeping these separate is critical:
                # raw_texts -> stored in DB, LLM reads this (clean, no prefix noise)
                # texts_to_embed -> only used for vector computation, never stored
                texts_to_embed.append(f"passage: {text}")

                # Chroma requires metadata values to be str, int, or float (no nested dicts)
                clean_meta = {
                    "source_file":  str(meta.get("source_file", "unknown")),
                    "case_id":      str(meta.get("case_id", "unknown")),
                    "chunk_index":  int(meta.get("chunk_index", 0)),
                    "strategy":     str(meta.get("strategy", "unknown")),
                    "char_len":     int(meta.get("char_len", 0)),
                    "token_len":    int(meta.get("token_len", 0)),
                }

                # unique ID for each chunk - consistent with evaluate_retrieval.py
                chunk_id = f"{clean_meta['source_file']}_chunk_{clean_meta['chunk_index']}"

                metadatas.append(clean_meta)
                ids.append(chunk_id)

            # --- Compute embeddings manually ---
            # normalize_embeddings=True forces all vectors to unit length (L2 norm = 1).
            # required so that cosine similarity scores are in range [-1, 1] and correctly
            # interpretable. also consistent with how query vectors are computed in evaluate_retrieval.py
            # - both sides must be normalized the same way.
            # show_progress_bar=False because tqdm above already shows file-level progress
            embeddings = model.encode(
                texts_to_embed,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # embeddings is a numpy array, ChromaDB needs a plain Python list
            embeddings_list = embeddings.tolist()

            # --- Insert into DB in batches ---
            # upsert() instead of add() so re-running the script doesn't crash on duplicate IDs.
            # batch size 100 is safe for ChromaDB memory limits.
            BATCH_SIZE = 100
            for i in range(0, len(raw_texts), BATCH_SIZE):
                collection.upsert(
                    ids=ids[i:i + BATCH_SIZE],
                    embeddings=embeddings_list[i:i + BATCH_SIZE],
                    documents=raw_texts[i:i + BATCH_SIZE],   # raw text, no prefix
                    metadatas=metadatas[i:i + BATCH_SIZE],
                )

            total_inserted += len(raw_texts)

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            failed_files += 1

    # --- 5. FINAL REPORT ---
    total_chunks_in_db = collection.count()
    print("\n=== VECTOR STORE CREATION COMPLETED ===")
    print(f"Files processed:   {len(json_files) - failed_files} / {len(json_files)}")
    print(f"Failed:            {failed_files}")
    print(f"Chunks inserted:   {total_inserted}")
    print(f"Total in DB now:   {total_chunks_in_db}")
    print(f"Database saved to: {DB_DIR}")


if __name__ == "__main__":
    main()