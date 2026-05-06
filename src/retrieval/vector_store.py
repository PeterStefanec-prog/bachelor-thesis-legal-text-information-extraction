# INPUT - jsons from      data/03_chunked_docs
#               data/03_chunked_docs/*_chunks_{STRATEGY}.json or *_children.json
#  saves them as vectors in ChromaDB
# OUTPUT directory - data/04_vectorstore/ or "data/04_vectorstore_hier"

# WIN CONFIG
# Win config (paragraph chunks + OpenAI text-embedding-3-small):
'''
MODEL_TYPE=openai \
CHUNK_SUFFIX=OPENAI_PARA \
COLLECTION_NAME=legal_decisions_openai_para \
.venv/bin/python src/retrieval/vector_store.py
'''

import os
import json
import glob # searching for documents based on pattern
import chromadb # my vector database
from tqdm import tqdm

# #############################################
# CONFIG - ENV VARIABLES
# ##########################################
# MODEL_TYPE: "me5" (local mE5-base), "openai" (text-embedding-3-small), or "openai_large" (text-embedding-3-large)
#                   i also tried local mE5-small at the beggining but was quietly worse
# CHUNK_SUFFIX: matches the chunking strategy suffix from chunk_processor.py
# COLLECTION_NAME: ChromaDB collection name
# CHUNK_MODE: "flat" (default) or "hier" (hierarchical - parents and children)
MODEL_TYPE      = os.environ.get("MODEL_TYPE",      "me5")  # default is local me5
CHUNK_SUFFIX    = os.environ.get("CHUNK_SUFFIX",    "ME5_380")  # which files to laod
CHUNK_MODE      = os.environ.get("CHUNK_MODE",      "flat")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", f"legal_decisions_{CHUNK_SUFFIX.lower()}")

# ### 1. PATH SETUP ###
# again - always runnin this script from the ROOT project folder
# if hier - change input and output dir
if CHUNK_MODE == "hier":
    INPUT_DIR = "data/03_chunked_docs_hier"
    DB_DIR = "data/04_vectorstore_hier"
    if not os.environ.get("COLLECTION_NAME"):
        COLLECTION_NAME = f"legal_decisions_hier_{MODEL_TYPE}"
else:
    INPUT_DIR = "data/03_chunked_docs"
    DB_DIR = "data/04_vectorstore"  # Here ChromaDB will save all its files

os.makedirs(DB_DIR, exist_ok=True)  # creates DB folder if doesnot exist
# ##########################################
# ##########################################



# #####################################
# --- 2. MODEL SETUP ---
# ####################################
# I load  model manually instead of letting ChromaDB do it internally.
# Because ChromaDBs built-in SentenceTransformerEmbeddingFunction is a black box -
# i cant control normalize_embeddings and i cant add the 'passage: ' prefix only for embedding while keeping  raw text clean for LLM later.
# At first i did it all throguh chromadb internal function,
#           but found out that passage prefix: and query: is missing and bcs of that performs badly
#           also did not have control over normalization

# just for debugging
print(f"Model: {MODEL_TYPE} | Mode: {CHUNK_MODE} | Collection: {COLLECTION_NAME}")

if MODEL_TYPE == "me5":
    from sentence_transformers import SentenceTransformer
    print("Loading mE5-base model locally...")
    model = SentenceTransformer("intfloat/multilingual-e5-base")    # if doesnt have - it downloads from Hugging Face Hub

    def embed_texts(texts, is_query=False):
        """Embed texts using mE5-base. Adds 'passage: ' or 'query: ' prefix as required by mE5 -  found  in model card
        NOTE: in this script is_query is always False (we only embed passages here).
        Query embedding happens in evaluate_retrieval.py directly. Kept the param just for completeness."""
        prefix = "query: " if is_query else "passage: " # e5 models were trained that it differs query and passage
        # create list of texts where each gets prefix at the beggining
        # [
        #   "passage: Súd znížil pokutu.",
        #   "passage: Žalobca sa domáhal zaplatenia."
        # ]
        prefixed = []
        for t in texts:
            prefixed.append(prefix + t)

        # change texts into vectors with normalization - each embedding is now of length 1 (right for cosine similarity)
        # what means i compare direction, not the length of vectors
        embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        # model.encode(...) returns nupy array but ChhromaDB wants puthon list
        return embeddings.tolist()

elif MODEL_TYPE in ("openai", "openai_large"):
    from openai import OpenAI

    if MODEL_TYPE == "openai_large":
        OPENAI_MODEL = "text-embedding-3-large"
    else:
        OPENAI_MODEL = "text-embedding-3-small"

    print(f"Using OpenAI {OPENAI_MODEL} via API...")
    # openai_client = OpenAI(api_key="sk-...")
    openai_client = OpenAI()  # reads OPENAI_API_KEY from env - mapping do the OpenA(() library internally

    def embed_texts(texts, is_query=False):     # no need for is query param, but maybe.. just to be sure implemented here
        """Embed texts using OpenAI embedding model. No prefix needed."""
        # OpenAI API has  batch limit, process in chunks of 2048
        all_embeddings = []
        BATCH = 2048    # 2048 chunks in one batch
        # iterating through texts (chunks)
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]  # actual batch
            response = openai_client.embeddings.create(
                model=OPENAI_MODEL,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]   # get embedding from response data (in response is data, model, object a usage)
            all_embeddings.extend(batch_embeddings)  # extend adds multiple elements at once (append would insert whole list as one element - got catched .. :) )
        return all_embeddings

else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Use 'me5', 'openai', or 'openai_large'.")  # modeltype is not valid


# ### 3. DATABASE SETUP ###
print("Initializing ChromaDB...")

# PersistentClient means it will save the database to my disk (not just in RAM) - was considering FAISS for having embeddings in ram and search in them
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# FIX: hnsw:space must be "cosine" so ChromaDB uses cosine similarity (copmares just direction of vector) under the hood - not base L2 metric
# Default is "l2" (Euclidean distance) which gives wrong results for text embeddings - i want just direction of vector in comparison
# hnsw is the alghoritmus =- type of index for searching - hnsw is like way of how chroma organize embeddings, so it could find similar entites quickly

# FIX: i do NOT pass embedding_function here anymore. i will compute embeddings manually - its up in the code
# like this
# collection = chroma_client.get_or_create_collection(
#     name="my_collection",
#     embedding_function=some_embedding_function
# )
# and pass them directly via embeddings= parameter in upsert().
# This gives me full control over how vectors are computed, not blackbox where i cant even add prefixes like passage

# collection is something like table (or dataset)
collection = chroma_client.get_or_create_collection(        # if collections with this name exist - open it - otherwise create
    name=COLLECTION_NAME,
    metadata={
        "description": f"Chunks of Slovak legal decisions, {MODEL_TYPE}/{CHUNK_MODE} mode, cosine space",
        "hnsw:space": "cosine",
    }
)


def clean_hier_metadata(meta):
    """Clean metadata for hierarchical children - ChromaDB only accepts str, int, float. No dict, lists"""

    clean = {
        "source_file": str(meta.get("source_file", "unknown")),
        "case_id": str(meta.get("case_id", "unknown")),
        "parent_id": str(meta.get("parent_id", "unknown")),
        "child_id": str(meta.get("child_id", "unknown")),
        "child_index": int(meta.get("child_index", 0)),
        "child_count_in_parent": int(meta.get("child_count_in_parent", 1)),
        "parent_index": int(meta.get("parent_index", 0)),
        "section": str(meta.get("section", "reasoning")),
        "point_number": int(meta.get("point_number", -1)),
        "position_ratio": float(meta.get("position_ratio", 0.0)),
        "char_len": int(meta.get("char_len", 0)),
        "token_len": int(meta.get("token_len", 0)),
    }
    return clean

# #############################
# --- 4. MAIN PIPELINE ---
################################
def main():
    # ChOOSING files to process
    if CHUNK_MODE == "hier":
        # hierarchical mode: only process *_children.json files
        json_files = glob.glob(os.path.join(INPUT_DIR, "*_children.json"))
    else:
        # flat mode: process chunks for the specified strategy - if i want to choose which chunk strategy to embedd
        # for example *_chunks_ME5_380.json
        json_files = glob.glob(os.path.join(INPUT_DIR, f"*_chunks_{CHUNK_SUFFIX}.json"))    # it is called from run_all_experiments with env variable

    if not json_files:
        print(f"Error: No chunk files found in {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} files. Starting vectorization and insertion...\n")

    total_inserted = 0  # just counters
    failed_files = 0

    # loop through all jsons - docuemtns with tehir chunks
    for file_path in tqdm(json_files, desc="Embedding and inserting"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)   # loading chunks

            if not chunks:
                continue

            # --- Build  3 lists ChromaDB needs --- docs.trychroma.com
            raw_texts = []       # clean text WITHOUT prefix - this is what gets stored in DB
            metadatas = []       # metadata of  chunk
            ids = []            # unique id of  chunk
            # and later embeddings_list[i]   -- -- all of that belongs to 1 chunk

            # LOOP through all chunks
            for chunk in chunks:
                text = chunk["page_content"]    # saving to database withou passage and query (do not want that in LLM)
                meta = chunk["metadata"]
                raw_texts.append(text)          # clean texts without prefixes

                if CHUNK_MODE == "hier":
                    clean_meta = clean_hier_metadata(meta)
                    chunk_id = clean_meta["child_id"]
                else:
                    # Chroma requires metadata values to be str, int, or float (no nested dicts) - same as was doing up there with hier
                    clean_meta = {
                        "source_file":  str(meta.get("source_file", "unknown")),
                        "case_id":      str(meta.get("case_id", "unknown")),
                        "chunk_index":  int(meta.get("chunk_index", 0)),
                        "strategy":     str(meta.get("strategy", "unknown")),
                        "char_len":     int(meta.get("char_len", 0)),
                        "token_len":    int(meta.get("token_len", 0)),
                    }
                    # unique ID for each chunk - consistent with evaluate_retrieval.py  (NS_SR_1Cdo_85_2023_00_dokument.pdf_chunk_4)
                    chunk_id = f"{clean_meta['source_file']}_chunk_{clean_meta['chunk_index']}"

                metadatas.append(clean_meta)    # adding to list
                ids.append(chunk_id)

            # ### Compute embeddings using the configured model ###
            embeddings_list = embed_texts(raw_texts, is_query=False)    # is query - so i would know if i need "passage" in me5

            # --- Insert into DB in batches ---
            # upsert() (if exits - update/rewrite) instead of add() so re-running the script doesn't crash on duplicate IDs.
            # batch size 100 is safer for ChromaDB memory limits
            BATCH_SIZE = 100
            for i in range(0, len(raw_texts), BATCH_SIZE):
                collection.upsert(          # upsert - if exists, just rewrite
                    ids=ids[i:i + BATCH_SIZE],
                    embeddings=embeddings_list[i:i + BATCH_SIZE],   # vectors of chunks
                    documents=raw_texts[i:i + BATCH_SIZE],   # raw texts of chunks, no prefix
                    metadatas=metadatas[i:i + BATCH_SIZE],  # metadata of chunks
                )

            # ID: 2
            # Cob_69_2020.pdf_chunk_3
            # VECTOR: [0.012, -0.193, ...]
            # TEXT: "Odvolací súd posudzoval primeranosť..."
            # METADATA: {"case_id": "2Cob/69/2020", "chunk_index": 3, ...}

            total_inserted += len(raw_texts)

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            failed_files += 1

    # --- 5. FINAL REPORT ---
    total_chunks_in_db = collection.count()
    print("\n=== VECTOR STORE CREATION COMPLETED ===")
    print(f"Model:             {MODEL_TYPE}")
    print(f"Mode:              {CHUNK_MODE}")
    print(f"Files processed:   {len(json_files) - failed_files} / {len(json_files)}")
    print(f"Failed:            {failed_files}")
    print(f"Chunks inserted:   {total_inserted}")
    print(f"Total in DB now:   {total_chunks_in_db}")
    print(f"Database saved to: {DB_DIR}")


if __name__ == "__main__":
    main()
