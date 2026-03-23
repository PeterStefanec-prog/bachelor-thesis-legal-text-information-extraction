import subprocess
import sys
import os

# ==========================================
# MASTER EXPERIMENT RUNNER
# ==========================================
# Runs ALL experiment combinations systematically:
#   1. For each (model_type, chunk_suffix) pair:
#      a) Creates ChromaDB vector store (if needed)
#      b) Runs evaluation grid: TOP_K × WINDOW combinations
#
# This is the ONE script you run to get all results for the thesis.
#
# Usage:
#   python src/evaluation_retrieval/run_all_experiments.py
#
# Prerequisites:
#   - Chunks must already exist in data/03_chunked_docs/
#     (run: python src/chunking/chunk_processor.py)
#   - For OpenAI experiments: OPENAI_API_KEY must be set in environment
#
# Results are APPENDED to:
#   - data/05_retrieval_evaluation/experiment_results_summary.csv
#   - data/05_retrieval_evaluation/experiment_results_details.csv
# ==========================================

VECTOR_STORE_SCRIPT = "src/retrieval/vector_store.py"
EVAL_SCRIPT         = "src/evaluation_retrieval/evaluate_retrieval.py"

# --- EXPERIMENT GRID ---
# Each entry: (model_type, chunk_suffix, short_name_for_experiment, retrieval_mode)
# short_name is used in experiment_name column in CSV results
# retrieval_mode: "dense" (default), "bm25", or "hybrid"
EXPERIMENTS = [
    # Phase 1: mE5-base with different chunk sizes (dense)
    ("me5",    "ME5_200",     "mE5base_200",    "dense"),
    ("me5",    "ME5_380",     "mE5base_380",    "dense"),

    # Phase 2: OpenAI text-embedding-3-small with different chunk sizes (dense)
    ("openai", "OPENAI_200",  "openai3s_200",   "dense"),
    ("openai", "OPENAI_500",  "openai3s_500",   "dense"),
    ("openai", "OPENAI_PARA", "openai3s_para",  "dense"),

    # Phase 3: BM25-only on paragraph chunks (no embedding model needed, but still
    # needs ChromaDB collection to exist for chunk storage and window expansion)
    ("openai", "OPENAI_PARA", "bm25_para",      "bm25"),

    # Phase 4: Hybrid (dense + BM25 with RRF) on paragraph chunks
    # RRF_K=60 is standard value from literature (designed for web search with thousands of docs)
    ("openai", "OPENAI_PARA", "hybrid_para",    "hybrid"),

    # Phase 5: Hybrid with lower RRF_K
    # Our documents have only 10-21 chunks, so with k=60 the rank differences are almost zero
    # (rank 1 → 1/61=0.0164, rank 5 → 1/65=0.0154 — only 6% difference!)
    # Lower k=20 makes top ranks matter more (rank 1 → 1/21=0.048, rank 5 → 1/25=0.040 — 19% diff)
    # This should help hybrid combine dense + BM25 better on small document collections
    ("openai", "OPENAI_PARA", "hybrid_para_k20", "hybrid"),

    # Phase 6: WEIGHTED hybrid - dense-dominant RRF
    # Standard RRF gives equal weight to dense and BM25 (alpha=0.5).
    # Problem: BM25 introduces noisy chunks that displace good dense results.
    # Solution: weighted RRF where dense has higher weight (alpha > 0.5).
    #
    # Analysis showed:
    # - Dense alone: 95.56% recall (best) but misses chunk_0 (introductory facts)
    # - BM25 alone: 89.44% recall (worst) but consistently FINDS chunk_0
    # - Equal hybrid: 92.50% recall (WORSE than dense! BM25 noise hurts)
    #
    # Weighted RRF with alpha=0.7: dense gets 70% weight, BM25 gets 30%.
    # Dense keeps its good chunks, BM25 gently boosts keyword-matching chunks.
    # Also improved BM25 queries to better target introductory paragraphs.
    ("openai", "OPENAI_PARA", "hybrid_weighted_a7", "hybrid"),  # alpha=0.7
    ("openai", "OPENAI_PARA", "hybrid_weighted_a8", "hybrid"),  # alpha=0.8

    # Phase 7: OpenAI text-embedding-3-large (SOTA) - paragraph chunks only
    # Hypothesis: larger model (3072-dim) finds better semantic matches,
    # potentially achieving same recall at lower top_k (= less noise for LLM).
    # Uses same OPENAI_PARA chunks (same tokenizer) but different collection for embeddings.
    ("openai_large", "OPENAI_PARA", "openai3l_para", "dense"),

]

# RRF_K overrides per experiment - when experiment needs different RRF_K than default (60)
# evaluate_retrieval.py reads RRF_K from env variable, default is 60
RRF_K_OVERRIDES = {
    "hybrid_para_k20": "20",
}

# RRF_ALPHA overrides per experiment - controls dense vs BM25 weight in hybrid
# default is 0.5 (equal weight = standard RRF)
# higher alpha = more dense influence, lower = more BM25
RRF_ALPHA_OVERRIDES = {
    "hybrid_weighted_a7": "0.7",
    "hybrid_weighted_a8": "0.8",
}

# Retrieval config grid - same for all experiments
TOP_K_VALUES  = [2, 3, 4, 5]
WINDOW_VALUES = [0, 1]

# --- SKIP LIST ---
# Add experiment short_names here to skip them (e.g. if already computed)
# Example: SKIP = {"mE5base_380"}  # skip because we already have these results
# Fresh run: all experiments from scratch (CSV files were deleted)
SKIP = set()



def run_command(description, script, env_vars):
    """Run a Python script with extra env variables. Returns True on success."""
    print(f"\n{'─'*60}")
    print(f"  {description}")
    print(f"{'─'*60}")

    env = os.environ.copy()
    env.update(env_vars)

    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        print(f"FAILED: {description}")
        return False
    return True


def main():
    print("=" * 60)
    print("  MASTER EXPERIMENT RUNNER")
    print("  Running all (model × chunk_size × top_k × window) combos")
    print("=" * 60)

    total_configs = 0
    failed_configs = []

    for model_type, chunk_suffix, short_name, retrieval_mode in EXPERIMENTS:
        if short_name in SKIP:
            print(f"\n{'='*60}")
            print(f"  SKIPPING: {short_name} (in SKIP list)")
            print(f"{'='*60}")
            continue

        # Collection name includes model for openai_large to avoid collision with openai
        # (different embedding dimensions: 1536 vs 3072)
        if model_type == "openai_large":
            collection_name = f"legal_decisions_openai_large_{chunk_suffix.lower()}"
        else:
            collection_name = f"legal_decisions_{chunk_suffix.lower()}"

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT GROUP: {short_name}")
        print(f"  Model: {model_type} | Chunks: {chunk_suffix} | Collection: {collection_name}")
        print(f"  Retrieval mode: {retrieval_mode}")
        print(f"{'='*60}")

        # Step 1: Create vector store for this (model, chunk_size) combination
        # (needed even for BM25 - chunks are stored in ChromaDB)
        vs_ok = run_command(
            f"Vectorizing: {model_type} / {chunk_suffix}",
            VECTOR_STORE_SCRIPT,
            {
                "MODEL_TYPE":      model_type,
                "CHUNK_SUFFIX":    chunk_suffix,
                "COLLECTION_NAME": collection_name,
            }
        )

        if not vs_ok:
            print(f"  Skipping evaluation for {short_name} due to vectorization failure")
            failed_configs.append(f"{short_name} (vectorization)")
            continue

        # Step 2: Run evaluation grid (TOP_K × WINDOW)
        # For paragraph-level chunks (BM25/hybrid), window=1 reads ~70-100% of the doc
        # which defeats the purpose of RAG. Only use window=0 for paragraph experiments.
        window_values = [0] if chunk_suffix == "OPENAI_PARA" else WINDOW_VALUES

        for top_k in TOP_K_VALUES:
            for window in window_values:
                exp_name = f"{short_name}_top{top_k}_w{window}"
                total_configs += 1

                # build env vars for this experiment
                eval_env = {
                    "MODEL_TYPE":      model_type,
                    "EXP_NAME":        exp_name,
                    "EXP_TOP_K":       str(top_k),
                    "EXP_WINDOW":      str(window),
                    "COLLECTION_NAME": collection_name,
                    "RETRIEVAL_MODE":  retrieval_mode,
                }

                # some experiments need different RRF_K (e.g. hybrid_para_k20 uses k=20)
                if short_name in RRF_K_OVERRIDES:
                    eval_env["RRF_K"] = RRF_K_OVERRIDES[short_name]

                # some experiments need different RRF_ALPHA (weighted hybrid)
                if short_name in RRF_ALPHA_OVERRIDES:
                    eval_env["RRF_ALPHA"] = RRF_ALPHA_OVERRIDES[short_name]

                eval_ok = run_command(
                    f"Evaluating: {exp_name}",
                    EVAL_SCRIPT,
                    eval_env
                )

                if not eval_ok:
                    failed_configs.append(exp_name)

    # --- FINAL REPORT ---
    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
    print(f"  Total configs evaluated: {total_configs}")
    print(f"  Failed: {len(failed_configs)}")
    if failed_configs:
        for f in failed_configs:
            print(f"    - {f}")
    print(f"\n  Results in: data/05_retrieval_evaluation/experiment_results_summary.csv")
    print(f"  Details in: data/05_retrieval_evaluation/experiment_results_details.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
