# cd /Users/stefanec/STU_FIIT/bachelor-thesis-legal-text-information-extraction
# export OPENAI_API_KEY="sk-..."
#   .venv/bin/python src/evaluation_retrieval/run_all_experiments.py
#      duration - 4 to 8 hours (macbook M2 RAM 24GB)

import subprocess
import sys
import os
import chromadb

# ########################################
# MASTER EXPERIMENT RUNNER
# ########################################
# Runs ALL experiment combinations systematically:
#   1. For each (model_type, chunk_suffix) pair:
#      a) Creates ChromaDB vector store (if needed)
#      b) Runs evaluation grid: TOP_K × WINDOW combinations
#   2. Hierarchical child-to-parent experiments (Phase 10+):
#      Uses CHUNK_MODE=hier with different grid (fetch_k × parents)
#
# This is the ONE script you run to get all results for the thesis.
#
# Prerequisites INPUT:
#   - Chunks must already exist in data/03_chunked_docs/
#     (run: python src/chunking/chunk_processor.py)
#   - For hierarchical: data/03_chunked_docs_hier/ must exist too
#   - For OpenAI experiments: OPENAI_API_KEY must be set in environment
#
# Results are appended to OUTPUT:
#   - data/05_retrieval_evaluation/experiment_results_summary.csv
#   - data/05_retrieval_evaluation/experiment_results_details.csv
# ###############################################

VECTOR_STORE_SCRIPT = "src/retrieval/vector_store.py"
EVAL_SCRIPT         = "src/evaluation_retrieval/evaluate_retrieval.py"

# ### EXPERIMENT GRID ###
# Each entry: (model_type, chunk_suffix, short_name_for_experiment, retrieval_mode)
# short_name is used in experiment_name column in CSV results
# retrieval_mode: "dense" (default), "bm25", or "hybrid"
EXPERIMENTS = [
    # Phase 1: mE5-base with different chunk sizes (dense)
    ("me5",    "ME5_200",     "mE5base_200",    "dense"),
    ("me5",    "ME5_380",     "mE5base_380",    "dense"),

    # Phase 2: OpenAI text-embedding-3-small with different chunk sizes (dense)
    ("openai", "OPENAI_200",  "openai3s_200",   "dense"),
    ("openai", "OPENAI_380",  "openai3s_380",   "dense"),  # added for fair embedding-model comparison
    ("openai", "OPENAI_500",  "openai3s_500",   "dense"),
    ("openai", "OPENAI_PARA", "openai3s_para",  "dense"),

    # Phase 3: BM25-only on paragraph chunks (no embedding model needed, but still
    # needs ChromaDB collection to exist for chunk storage and window expansion)
    ("openai", "OPENAI_PARA", "bm25_para",      "bm25"),

    # Phase 4: Hybrid (dense + BM25 with RRF) on paragraph chunks
    # RRF_K=60 is standard value from literature (designed for web search with thousands of docs)
    ("openai", "OPENAI_PARA", "hybrid_para",    "hybrid"),

    # Phase 5: Hybrid with lower RRF_K
    # Our documents have only 10-21 chunks, so with k=60 the rank differences are almost zero (rank 1 -> 1/61=0.0164, rank 5 -> 1/65=0.0154 - only 6% difference!)
    # Lower k=20 makes top ranks matter more (rank 1 -> 1/21=0.048, rank 5 -> 1/25=0.040 - 19% diff)
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
    # Hypothesis: larger model (3072-dim) finds better semantic matches potentially achieving same recall at lower top_k (= less noise for LLM).
    # Uses same OPENAI_PARA chunks (same tokenizer) but different collection for embeddings.
    ("openai_large", "OPENAI_PARA", "openai3l_para", "dense"),
    ("openai_large", "OPENAI_380",  "openai3l_380",  "dense"),  # added for fair 380t embedding-model ablation

    # Phase 8: Reranking experiments - domain-specific keyword reranker on top of retrieval
    # The reranker fetches 2x chunks (oversample=2), re-scores them with legal keyword
    # patterns, and picks the best top_k. This should fix the last few % of recall
    # where retrieval puts procedural chunks above informative ones.
    #
    # Two configs to compare:
    # a) hybrid a=0.7 + reranker - our best hybrid config with reranker on top
    # b) dense + reranker - can the reranker fix dense retrieval without BM25?
    ("openai", "OPENAI_PARA", "hybrid_a7_reranked", "hybrid"),
    ("openai", "OPENAI_PARA", "dense_reranked", "dense"),

    # Phase 9: Hybrid + reranker with HIGH reranker alpha (0.85)
    # Phase 8 showed that hybrid + reranker(alpha=0.6) hurts recall at top_k=4,5
    # because keywords get double-boosted (BM25 + reranker = 58% keyword influence).
    # With alpha=0.85, retrieval score keeps 85% weight and keywords only 15%,
    # so the total keyword influence drops to ~39% (BM25 24% + reranker 15%).
    ("openai", "OPENAI_PARA", "hybrid_a7_reranked_a85", "hybrid"),

]

# ########################################
# HIERARCHICAL EXPERIMENTS (Phase 10+)
# ########################################
# After flat experiments plateaued (~93% recall, ~76% coverage), we tried  different architecture: retrieve small children (~220 tokens),
# group by parent, send complete parents to LLM.
# Goal is better COVERAGE -  LLM sees full legal arguments instead of chopped-up paragraph pieces.
#
# Grid: model × retrieval_mode × fetch_k × call1_parents × call2_parents
# These use CHUNK_MODE=hier and the same evaluate_retrieval.py script.
# No TOP_K/WINDOW grid - parent selection replaces that.
#
# Each entry: (model_type, retrieval_mode, short_name_prefix)
# Hierarchical experiments are kept for comparison purposes - they don't
# benefit from v2 reranker fixes (hier doesn't use the reranker), but
# the existing archived vectorstore at data/04_vectorstore_hier
# (restored from data/04_vectorstore_hier_04_03) makes them runnable
# without re-embedding. Each hier group runs the
# fetch_k × call1_parents × call2_parents grid (18 configs each → 90 total).
HIER_EXPERIMENTS = [
    # Phase 10: mE5 local scan (free, fast) - find best config
    ("me5",    "dense",  "hier_me5_dense"),
    ("me5",    "bm25",   "hier_me5_bm25"),
    ("me5",    "hybrid", "hier_me5_hybrid"),

    # Phase 11: OpenAI confirmation with best config from Phase 10
    ("openai", "dense",  "hier_openai_dense"),
    ("openai", "hybrid", "hier_openai_hybrid"),
]

# Hierarchical grid parameters
HIER_FETCH_K_VALUES      = [6, 8]
HIER_CALL1_PARENT_VALUES = [5, 7, 9]
HIER_CALL2_PARENT_VALUES = [7, 9, 11]


# RRF params for hierarchical hybrid experiments
HIER_RRF_K     = "20"
HIER_RRF_ALPHA = "0.7"




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
    "hybrid_a7_reranked": "0.7",
    "hybrid_a7_reranked_a85": "0.7",
}

# Reranker overrides per experiment - which experiments use the domain-specific reranker
# USE_RERANKER=1 turns it on, RERANKER_ALPHA controls retrieval vs keyword balance,
# RERANKER_OVERSAMPLE controls how many extra chunks to fetch (2 = fetch 2x top_k)
#
# FIX: changed RERANKER_OVERSAMPLE from 3 to 5 to match the production pipeline
# (precompute_retrieval.py uses oversample=5).
# Before this, experiments tested with oversample=3 but the actual extraction used oversample=5 - so experiments
# underestimated the reranker's real recall. With 5, the reranker picks top_k
# from 5*k candidates which should improve results especially for hybrid_a7_reranked.
RERANKER_OVERRIDES = {
    "hybrid_a7_reranked": {
        "USE_RERANKER": "1",
        "RERANKER_ALPHA": "0.6",
        "RERANKER_OVERSAMPLE": "5",
    },
    "dense_reranked": {
        "USE_RERANKER": "1",
        "RERANKER_ALPHA": "0.6",
        "RERANKER_OVERSAMPLE": "5",
    },
    "hybrid_a7_reranked_a85": {
        "USE_RERANKER": "1",
        "RERANKER_ALPHA": "0.85",
        "RERANKER_OVERSAMPLE": "5",
    },
}

# Retrieval config grid
# After switching to TRUE paragraph chunking (one numbered point = one chunk),
# PARA chunks are smaller than before (~100-800 tokens vs old merged ~1000-1500).
# the old top_k=5 effectively retrieved ~15 paragraphs (because merged chunks had
# 2-3 points each). now top_k=5 gets exactly 5 paragraphs, so we need higher top_k
# to compensate. range [2..10] lets us find the sweet spot.
# Small fixed-size chunks (200/380/500 tokens) also use extended range.
TOP_K_VALUES            = [2, 3, 4, 5, 6, 7, 10]   # PARA experiments - added k=6 for finer granularity
TOP_K_VALUES_SMALL      = [2, 3, 4, 5, 6, 7, 8] # fixed-size chunks
WINDOW_VALUES = [0, 1]

# --- SKIP LIST ---
# Add experiment short_names here to skip them (e.g. if already computed)
# Example: SKIP = {"mE5base_380"}  # skip because we already have these results
# Fresh run: empty set = run everything
# skip fixed-size experiments that didn't change (only PARA chunks changed)
# remove these after running PARA experiments to get complete results
SKIP = {
    # PARA + fixed-size (126 rows already in CSV)
    "bm25_para",
    "dense_reranked",
    "hybrid_a7_reranked",
    "hybrid_a7_reranked_a85",
    "hybrid_para",
    "hybrid_para_k20",
    "hybrid_weighted_a7",
    "hybrid_weighted_a8",
    "mE5base_200",
    "mE5base_380",
    "openai3l_para",
    "openai3s_200",
    "openai3s_500",
    "openai3s_para",
    # mE5 hier (54 rows already in CSV - successful previous run)
    "hier_me5_dense",
    "hier_me5_bm25",
    "hier_me5_hybrid",
    # OpenAI hier (46 rows already in CSV from earlier run - DON'T re-run, costly)
    "hier_openai_dense",
    "hier_openai_hybrid",
}



def collection_exists(db_dir, collection_name):
    """Check if  ChromaDB collection already exists and has data.
    Returns chunk count if exists, 0 otherwise."""
    try:
        client = chromadb.PersistentClient(path=db_dir)
        col = client.get_collection(collection_name)
        count = col.count()
        return count
    except Exception:
        return 0


def run_command(description, script, env_vars):
    """Run  Python script with extra env variables. Returns True on success."""
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
    # track which collections have already been vectorized this run
    # so we don't re-embed the same chunks multiple times (saves time + API costs)
    vectorized_collections = set()

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
        # Skip if we already vectorized this collection in this run
        if collection_name in vectorized_collections:
            print(f"\n  Collection '{collection_name}' already vectorized, skipping re-embedding")
        else:
            existing = collection_exists("data/04_vectorstore", collection_name)
            if existing > 0:
                print(f"\n  Collection '{collection_name}' already in ChromaDB ({existing} chunks), skipping re-embedding")
                vectorized_collections.add(collection_name)
            else:
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
                vectorized_collections.add(collection_name)

        # Step 2: Run evaluation grid (TOP_K × WINDOW)
        # For paragraph-level chunks (BM25/hybrid), window=1 reads ~70-100% of the doc
        # which defeats the purpose of RAG. Only use window=0 for paragraph experiments.
        # For small fixed-size chunks, use extended TOP_K range (up to 8) to give them
        # a fair shot at matching PARA recall - even at higher coverage.
        window_values = [0] if chunk_suffix == "OPENAI_PARA" else WINDOW_VALUES
        top_k_values  = TOP_K_VALUES if chunk_suffix == "OPENAI_PARA" else TOP_K_VALUES_SMALL

        for top_k in top_k_values:
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

                # some experiments use the domain-specific reranker
                if short_name in RERANKER_OVERRIDES:
                    eval_env.update(RERANKER_OVERRIDES[short_name])

                eval_ok = run_command(
                    f"Evaluating: {exp_name}",
                    EVAL_SCRIPT,
                    eval_env
                )

                if not eval_ok:
                    failed_configs.append(exp_name)

    # ##################################################
    # HIERARCHICAL EXPERIMENTS
    # ##################################################
    # Same evaluate_retrieval.py but with CHUNK_MODE=hier.
    # Instead of TOP_K × WINDOW, we iterate over fetch_k × parent counts.
    vectorized_hier_collections = set()

    for model_type, retrieval_mode, short_prefix in HIER_EXPERIMENTS:
        if short_prefix in SKIP:
            print(f"\n{'='*60}")
            print(f"  SKIPPING: {short_prefix} (in SKIP list)")
            print(f"{'='*60}")
            continue

        # hierarchical collections use a different naming convention
        collection_name = f"legal_decisions_hier_{model_type}"

        print(f"\n{'='*60}")
        print(f"  HIER EXPERIMENT GROUP: {short_prefix}")
        print(f"  Model: {model_type} | Mode: hier | Retrieval: {retrieval_mode}")
        print(f"{'='*60}")

        # vectorize hierarchical children (once per model)
        if collection_name not in vectorized_hier_collections:
            existing = collection_exists("data/04_vectorstore_hier", collection_name)
            if existing > 0:
                print(f"\n  Collection '{collection_name}' already in ChromaDB ({existing} chunks), skipping re-embedding")
                vectorized_hier_collections.add(collection_name)
            else:
                vs_ok = run_command(
                    f"Vectorizing hier: {model_type}",
                    VECTOR_STORE_SCRIPT,
                    {
                        "MODEL_TYPE":      model_type,
                        "CHUNK_MODE":      "hier",
                        "COLLECTION_NAME": collection_name,
                    }
                )
                if not vs_ok:
                    print(f"  Skipping {short_prefix} due to vectorization failure")
                    failed_configs.append(f"{short_prefix} (vectorization)")
                    continue
                vectorized_hier_collections.add(collection_name)

        # run grid: fetch_k × call1_parents × call2_parents
        for fetch_k in HIER_FETCH_K_VALUES:
            for c1p in HIER_CALL1_PARENT_VALUES:
                for c2p in HIER_CALL2_PARENT_VALUES:
                    exp_name = f"{short_prefix}_fk{fetch_k}_c1p{c1p}_c2p{c2p}"
                    total_configs += 1

                    eval_env = {
                        "MODEL_TYPE":       model_type,
                        "CHUNK_MODE":       "hier",
                        "EXP_NAME":         exp_name,
                        "COLLECTION_NAME":  collection_name,
                        "RETRIEVAL_MODE":   retrieval_mode,
                        "FETCH_K_PER_QUERY": str(fetch_k),
                        "CALL1_PARENTS":    str(c1p),
                        "CALL2_PARENTS":    str(c2p),
                    }

                    if retrieval_mode == "hybrid":
                        eval_env["RRF_K"] = HIER_RRF_K
                        eval_env["RRF_ALPHA"] = HIER_RRF_ALPHA

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
