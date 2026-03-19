import subprocess
import sys
import os

# ==========================================
# Simple experiment runner.
# Just loops over TOP-K x WINDOW combinations and calls evaluate_retrieval.py
# for each one. No code duplication - everything stays in evaluate_retrieval.py.
#
# Run from project root: python src/evaluation_retrieval/experiment_runner.py
# ==========================================

EVAL_SCRIPT = "src/evaluation_retrieval/evaluate_retrieval.py"

# chunk strategy for this run - change this when testing different chunk sizes
# matching collection must already exist in ChromaDB (run vector_store.py first)
CHUNK_SUFFIX    = "ME5_380"
COLLECTION_NAME = f"legal_decisions_{CHUNK_SUFFIX.lower()}"

TOP_K_VALUES  = [3, 4, 5, 6, 7]
WINDOW_VALUES = [0, 1]

for top_k in TOP_K_VALUES:
    for window in WINDOW_VALUES:
        experiment_name = f"mE5_{CHUNK_SUFFIX.lower()}_top{top_k}_window{window}"
        print(f"\n{'='*60}")
        print(f"RUNNING: {experiment_name}")
        print(f"{'='*60}")

        # Pass config as env variables - evaluate_retrieval.py reads these at the top
        env = os.environ.copy()
        env["EXP_NAME"]        = experiment_name
        env["EXP_TOP_K"]       = str(top_k)
        env["EXP_WINDOW"]      = str(window)
        env["COLLECTION_NAME"] = COLLECTION_NAME

        result = subprocess.run([sys.executable, EVAL_SCRIPT], env=env)

        if result.returncode != 0:
            print(f"ERROR in {experiment_name}, continuing...")