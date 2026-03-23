import os
import json
import glob
import chromadb
from tqdm import tqdm

# ==========================================
# ENV VARIABLES
# ==========================================
# MODEL_TYPE: "me5" (local mE5-base), "openai" (text-embedding-3-small), or "openai_large" (text-embedding-3-large)
# CHUNK_SUFFIX: matches the chunking strategy suffix from chunk_processor.py
# COLLECTION_NAME: ChromaDB collection name (auto-computed if not set)
MODEL_TYPE      = os.environ.get("MODEL_TYPE",      "me5")
CHUNK_SUFFIX    = os.environ.get("CHUNK_SUFFIX",    "ME5_380")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", f"legal_decisions_{CHUNK_SUFFIX.lower()}")

# --- 1. PATH SETUP ---
# Important: Run this script from the ROOT project folder
INPUT_DIR = "data/03_chunked_docs"
DB_DIR = "data/04_vectorstore"  # Here ChromaDB will save all its files

os.makedirs(DB_DIR, exist_ok=True)

# ==========================================
# --- 2. MODEL SETUP ---
# ==========================================
# I load the model manually instead of letting ChromaDB do it internally.
# Why: ChromaDB's built-in SentenceTransformerEmbeddingFunction is a black box -
# i can't control normalize_embeddings and i can't add the 'passage: ' prefix
# only for embedding while keeping the raw text clean for LLM later.

print(f"Model: {MODEL_TYPE} | Chunks: {CHUNK_SUFFIX} | Collection: {COLLECTION_NAME}")

if MODEL_TYPE == "me5":
    from sentence_transformers import SentenceTransformer
    print("Loading mE5-base model locally...")
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    def embed_texts(texts, is_query=False):
        """Embed texts using mE5-base. Adds 'passage: ' or 'query: ' prefix as required by mE5."""
        prefix = "query: " if is_query else "passage: "
        prefixed = [f"{prefix}{t}" for t in texts]
        embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

elif MODEL_TYPE in ("openai", "openai_large"):
    from openai import OpenAI
    OPENAI_MODEL = "text-embedding-3-large" if MODEL_TYPE == "openai_large" else "text-embedding-3-small"
    print(f"Using OpenAI {OPENAI_MODEL} via API...")
    openai_client = OpenAI()  # reads OPENAI_API_KEY from env

    def embed_texts(texts, is_query=False):
        """Embed texts using OpenAI embedding model. No prefix needed."""
        # OpenAI API has a batch limit, process in chunks of 2048
        all_embeddings = []
        BATCH = 2048
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            response = openai_client.embeddings.create(
                model=OPENAI_MODEL,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Use 'me5', 'openai', or 'openai_large'.")


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
    name=COLLECTION_NAME,
    metadata={
        "description": f"Chunks of Slovak legal decisions, {MODEL_TYPE}/{CHUNK_SUFFIX} strategy, cosine space",
        "hnsw:space": "cosine",
    }
)


# --- 4. MAIN PIPELINE ---
def main():
    # I only want to load the chunks for this specific strategy
    json_files = glob.glob(os.path.join(INPUT_DIR, f"*_chunks_{CHUNK_SUFFIX}.json"))

    if not json_files:
        print(f"Error: No chunk files found in {INPUT_DIR} matching *_chunks_{CHUNK_SUFFIX}.json")
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
            metadatas = []
            ids = []

            for chunk in chunks:
                text = chunk["page_content"]
                meta = chunk["metadata"]

                raw_texts.append(text)

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

            # --- Compute embeddings using the configured model ---
            embeddings_list = embed_texts(raw_texts, is_query=False)

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
    print(f"Model:             {MODEL_TYPE}")
    print(f"Chunk strategy:    {CHUNK_SUFFIX}")
    print(f"Files processed:   {len(json_files) - failed_files} / {len(json_files)}")
    print(f"Failed:            {failed_files}")
    print(f"Chunks inserted:   {total_inserted}")
    print(f"Total in DB now:   {total_chunks_in_db}")
    print(f"Database saved to: {DB_DIR}")


if __name__ == "__main__":
    main()
