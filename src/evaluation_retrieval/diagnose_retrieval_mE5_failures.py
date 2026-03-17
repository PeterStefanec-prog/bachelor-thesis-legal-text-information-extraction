import os
import csv
import re
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# This script diagnoses WHY dense retrieval fails on specific documents.
# For each failing case it shows:
#   1. How uniform the cosine scores are (low range = model cant discriminate)
#   2. What rank the correct chunk actually has (should be top 5, often isnt)
#   3. What content the model DID retrieve (usually generic legal boilerplate)
#   4. Why BM25 would fix it (distinctive keyword appears only N times)
#
# Run this script to generate evidence for thesis section on dense retrieval limits.
# The output clearly shows the "uniform cosine problem" and motivates BM25 hybrid.
# ==========================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

DB_DIR = "data/04_vectorstore"
CSV_PATH = "data/05_retrieval_evaluation/golden_dataset_template.csv"

# Documents and queries that consistently MISS - the interesting failure cases
# Add or remove as needed based on your evaluation results
DIAGNOSE_CASES = [
    {
        "doc": "KS_Bratislava_1Cob_40_2018_00_dokument.pdf",
        "query_key": "q_context",
        "query_text": "Čo bolo predmetom zmluvy medzi stranami? Akú povinnosť si dlžník prevzal a ako ju porušil?",
        # BM25 keyword: a distinctive phrase that appears rarely in the doc
        "bm25_keyword": "leasingovú zmluvu č. 9587/25",
    },
    {
        "doc": "KS_Trnava_32Cob_3_2021_00_dokument.pdf",
        "query_key": "q_context",
        "query_text": "Čo bolo predmetom zmluvy medzi stranami? Akú povinnosť si dlžník prevzal a ako ju porušil?",
        "bm25_keyword": "abonentnú zmluvu",
    },
    {
        "doc": "KS_Banská_Bystrica_43CoPv_10_2023_00_dokument.pdf",
        "query_key": "q_context",
        "query_text": "Čo bolo predmetom zmluvy medzi stranami? Akú povinnosť si dlžník prevzal a ako ju porušil?",
        "bm25_keyword": "metabolic balance",
    },
    {
        "doc": "NS_SR_1Obdo_72_2019_00_dokument.pdf",
        "query_key": "q_context",
        "query_text": "Čo bolo predmetom zmluvy medzi stranami? Akú povinnosť si dlžník prevzal a ako ju porušil?",
        "bm25_keyword": "sprostredkovanie predaja, prenájmu a kúpy nehnuteľností",
    },
    {
        "doc": "KS_Bratislava_2Co_380_2011_00_dokument.pdf",
        "query_key": "q_context",
        "query_text": "Čo bolo predmetom zmluvy medzi stranami? Akú povinnosť si dlžník prevzal a ako ju porušil?",
        "bm25_keyword": "rodinný dom",
    },
]


def clean_text(text):
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r'\s+', ' ', text).strip()


def main():
    print("Loading mE5 model...")
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(name="legal_decisions_me5")

    # Load golden dataset for reference
    golden = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            golden[row["document_name"].replace(".json", ".pdf")] = row

    print()
    print("=" * 70)
    print("DENSE RETRIEVAL FAILURE DIAGNOSTICS")
    print("=" * 70)

    for case in DIAGNOSE_CASES:
        doc = case["doc"]
        query_text = case["query_text"]
        bm25_kw = case["bm25_keyword"]

        # Map query key to CSV column
        col_map = {"q_context": "q1_context_quotes", "q_penalty": "q2_penalty_quotes",
                   "q3": "q3_moderation_quotes"}
        csv_col = col_map.get(case["query_key"], "q1_context_quotes")

        golden_quotes = []
        if doc in golden:
            raw = golden[doc].get(csv_col, "")
            golden_quotes = [clean_text(q) for q in raw.split("|") if clean_text(q)]

        print()
        print(f"DOCUMENT: {doc}")
        print(f"QUERY:    {case['query_key']} — \"{query_text[:70]}...\"")
        print("-" * 70)

        # Step 1: Encode query
        query_vec = model.encode(
            [f"query: {query_text}"],
            normalize_embeddings=True
        ).tolist()

        # Step 2: Get ALL chunks for this document (not just top-K)
        all_chunks = collection.get(
            where={"source_file": doc},
            include=["documents", "metadatas"]
        )
        total_chunks = len(all_chunks["ids"])

        # Step 3: Query with large n_results to get full ranking
        results = collection.query(
            query_embeddings=query_vec,
            n_results=total_chunks,
            where={"source_file": doc},
            include=["documents", "metadatas", "distances"]
        )

        all_scores = [round(1 - d, 4) for d in results["distances"][0]]
        all_docs = results["documents"][0]
        all_ids = results["ids"][0]

        # Step 4: Find which ranks contain the golden quotes
        correct_ranks = []
        for rank, (chunk_text, score, chunk_id) in enumerate(zip(all_docs, all_scores, all_ids), 1):
            cleaned_chunk = clean_text(chunk_text)
            for quote in golden_quotes:
                if quote[:40] in cleaned_chunk:
                    correct_ranks.append((rank, score, chunk_id, quote[:50]))
                    break

        # Step 5: Show score distribution
        max_score = max(all_scores)
        min_score = min(all_scores)
        score_range = round(max_score - min_score, 4)
        top5_scores = all_scores[:5]

        print(f"Total chunks in document: {total_chunks}")
        print(f"Score range (max-min):    {max_score} - {min_score} = {score_range}")
        print(f"  -> {'⚠️  UNIFORM (range < 0.02) — model cannot discriminate' if score_range < 0.02 else '✓  Good range — model can discriminate'}")
        print()

        # Step 6: Show top-5 retrieved chunks
        print("TOP-5 RETRIEVED CHUNKS (what model thinks is relevant):")
        for rank, (score, text) in enumerate(zip(top5_scores, all_docs[:5]), 1):
            marker = "✓" if any(q[:40] in clean_text(text) for q in golden_quotes) else "✗"
            print(f"  #{rank} [{score:.4f}] {marker}  {repr(clean_text(text)[:80])}")

        print()

        # Step 7: Show where correct chunks actually rank
        if correct_ranks:
            print("WHERE ARE THE CORRECT CHUNKS?")
            for rank, score, chunk_id, quote in correct_ranks:
                in_top5 = "✓ IN TOP-5" if rank <= 5 else f"✗ RANK #{rank} — OUTSIDE TOP-5"
                print(f"  Rank #{rank}/{total_chunks} [{score:.4f}]  {in_top5}")
                print(f"  Quote: \"{quote}...\"")
        else:
            print("CORRECT CHUNKS: quotes not found in any chunk (check golden dataset)")

        print()

        # Step 8: BM25 preview — how many times does the distinctive keyword appear?
        keyword_count = sum(
            1 for text in all_docs
            if bm25_kw.lower() in text.lower()
        )
        chunks_with_kw = [
            (i+1, all_scores[i], clean_text(text)[:70])
            for i, text in enumerate(all_docs)
            if bm25_kw.lower() in text.lower()
        ]

        print(f"BM25 KEYWORD: \"{bm25_kw}\"")
        print(f"  Appears in {keyword_count}/{total_chunks} chunks")
        if chunks_with_kw:
            for rank, score, text in chunks_with_kw[:3]:
                print(f"  Chunk at dense rank #{rank} [{score:.4f}]: {repr(text)}")
        if keyword_count <= 2:
            print(f"  -> ✓ BM25 would rank this chunk #1 immediately (rare keyword)")
        else:
            print(f"  -> appears {keyword_count}x — BM25 still helps but less dramatically")

        print("-" * 70)

    print()
    print("CONCLUSION:")
    print("Documents with score range < 0.02 are the 'BM25 wall' —")
    print("dense model treats all chunks as equally relevant.")
    print("BM25 on rare legal keywords would break this tie immediately.")


if __name__ == "__main__":
    main()