"""
Main extraction pipeline - runs LLM extraction on precomputed retrieval results (it just orchestrate extraction)

For each document this script:
1. Loads precomputed retrieval chunks from data/06_retrieval_results/
2. Builds prompts (system + user) from templates
3. Calls the LLM (Call 1: facts, Call 2: moderation, or Full-doc: everything at once) through llm_client.py
4. Parses JSON response
5. Merges Call 1 + Call 2 (for RAG mode)
6. Saves result to data/07_extractions/{model}_{mode}/

Two modes per model:
- RAG mode: 2 LLM calls with focused chunks (hybrid_a7_reranked_top7 retrieval)
- Full-doc mode: 1 LLM call with entire reasoning tex

run from  root:
    python src/extraction/run_extraction.py

I tested  prompts first in:
- OpenAI Playground (https://platform.openai.com/playground) - paste system + user prompt,
  set model to gpt-4o, set response format to JSON. Someree credits
- Google AI Studio (https://aistudio.google.com) - paste full prompt, select Gemini 2.5 Pro.
  Completely free
- Ollama locally, but only slovak version of qwen3-14B-sk (but just pasting prompts to reminal)
"""

import os
import sys
import json
import glob
import tiktoken
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))     # just set root address
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)    # so imports like this (from src.extraction.prompt_builder import ...) wil work
os.chdir(PROJECT_ROOT)      # set working idrectory as project root

# importing my own modules
from src.extraction.prompt_builder import (
    build_system_prompt, build_call1_prompt, build_call2_prompt, build_fulldoc_prompt)  # bu
from src.extraction.llm_client import call_llm      # call model
from src.extraction.response_parser import parse_llm_response, merge_call1_call2    # parsing output and merge call 1 + call 2

# tiktoken = OpenAIs tokenizer library. I use it to count exact tokens before sending to the API
# I originally used a rough heuristic (chars / 4) but that underestimated Slovak legal text by appr. 18% and caused me 429 errors
# Tiktoken gives  real number so i know exactly when i need to trim the document.
_tokenizer = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    """Count exact number of tokens using tiktoken - as the winning strategy uses openai embedding model"""
    return len(_tokenizer.encode(text))


# ==========================================
# CONFIG - what model and mode is running
# ==========================================
# change these to run different experiments.
# each combination creates a separate output folder

MODEL_KEY = "qwen3.5-397b"  # "gpt-4o" | "gemini-2.5-flash" | "qwen3.5-397b"
MODE = "rag"                      # "rag" (2 calls with chunks) | "fulldoc" (1 call with full text)

# test on single document first (has to be None for  full run)
# SINGLE_DOC_TEST = "KS_Trenčín_8Cob_64_2011_00_dokument"
SINGLE_DOC_TEST = None

# set to True to only process extraction (not golden from retrieval - randomly selected but some of them from golden) dataset documents (20 docs)
# i use this to test and debug before running on all 176 documents
GOLDEN_ONLY = True

RETRIEVAL_DIR = "data/06_retrieval_results"
# NOTE: OUTPUT_DIR is recomputed inside main() from the current MODEL_KEY/MODE
# so that     import run_extraction as r; r.MODE = "fulldoc"; r.main()     works correctly.
# If i compute it at module load time, changing r.MODE later has no effect and the output saves in wrong folder.
OUTPUT_DIR = f"data/07_extractions/{MODEL_KEY}_{MODE}"


################ MAIN FUNCTION FOR RAG EXTRACTION  ################
def run_rag_extraction(retrieval_data, model_key):      # json with retrieval results for 1 doc, model which will be used
    """Run RAG extraction: Call 1 (facts) + Call 2 (moderation), then merge.

     Main pipeline:
    1. Build Call 1 prompt with contract/penalty chunks
    2. Send to LLM, parse response
    3. Build Call 2 prompt with moderation chunks + prefilled JSON from Call 1
    4. Send to LLM, parse response
    5. Merge both into final JSON

    Returns: (final_result_dict, metadata_dict)
    """
    system_prompt = build_system_prompt()   # common for all calls
    all_meta = {"call1": {}, "call2": {}}  # track API usage for both callas

    # #3# CALL 1: contract facts + penalty definition ###
    call1_prompt = build_call1_prompt(retrieval_data)
    print(f" Call 1: sending {len(call1_prompt):,} chars to {model_key}..")

    # here i am using sth like wrapper - llm_client.py
    call1_response = call_llm(model_key, system_prompt, call1_prompt, call_type="call1")  # response is total tokens, cost, etc..
    # delete real answer and save other meta
    all_meta["call1"] = call1_response.copy()
    del all_meta["call1"]["response_text"]

    # change raw json string from model to python dictionary
    call1_result, error = parse_llm_response(call1_response["response_text"])
    if error:
        print(f"  ERROR parsing Call 1: {error}")
        # DEBUG: dumping first 500 chars of raw response so i can see whats wrong
        raw = call1_response.get("response_text", "")
        print(f"  RAW RESPONSE (first 500 chars): {raw[:500]}")
        return None, all_meta

    print(f"  Call 1 done: {call1_response['prompt_tokens']} prompt + {call1_response['completion_tokens']} completion tokens")

    # ### CALL 2: moderation analysis ###
    call2_prompt = build_call2_prompt(retrieval_data, call1_result) # call2 chunks, and filled json from call 1
    print(f"  Call 2: sending {len(call2_prompt):,} chars to {model_key}...")

    # sending it to llm
    call2_response = call_llm(model_key, system_prompt, call2_prompt, call_type="call2")
    # again save metadata ofc except LLM response
    all_meta["call2"] = call2_response.copy()
    all_meta["call2"].pop("response_text", None)

    call2_result, error = parse_llm_response(call2_response["response_text"])   # to python dict by function from response_parser
    if error:
        print(f"  ERROR parsing Call 2: {error}")
        # still return call1 result with empty moderation
        return merge_call1_call2(call1_result, None), all_meta  # return at least answer from call 1..

    print(f" Call 2 done: {call2_response['prompt_tokens']} prompt + {call2_response['completion_tokens']} completion tokens")

    # --- MERGE + EVIDENCE GROUNDING CHECK ---
    # FIX 2: i build  lookup dict (chunk_id -> text) so  can checked if evidence quotes are real substrings of the chunks
    # I added this after testing on the first document - 5 out of 13 quotes were wrong!
    #  LLM was changing uppercase/lowercase and cutting quotes short with periods (but changed preprocess of response later)
    # Without this check i would never know the evidence is fake.
    #
    # FIX 3 follow-up: i also add "header" and "verdict" to the lookup because after i gave the verdict to Call 2, the LLM started citing from it (whichis correct!). But without these keys the grounding check flagged those as
    # EVIDENCE_NOT_FOUND which was a false alarm.
    chunk_lookup = {
        "header": retrieval_data.get("header", ""),
        "verdict": retrieval_data.get("verdict", ""),
    }
    for c in retrieval_data["call1_chunks"] + retrieval_data["call2_chunks"]:
        chunk_lookup[c["chunk_id"]] = c["text"]

    # merging both extractions and optionally giving chunk lookup (for checking if llm halicantes - validation)
    final = merge_call1_call2(call1_result, call2_result, chunk_lookup=chunk_lookup)
    return final, all_meta


####### basline mode - only one call #######
def run_fulldoc_extraction(retrieval_data, model_key):
    """Run full-document extraction: 1 call with entire text, complete schema

    This is the baseline experiment -  LLM reads the whole document instead of focused chunks
    This tests whether RAG (retrieval) is better than just giving the model everything.

    Returns: (result_dict, metadata_dict)
    """
    system_prompt = build_system_prompt() # just getting system prompt

    # FIX: GPT-4o on my Tier 1 has 30k TPM (tokens per minute) limit
    # The model supports 128k context but the API wont let me send more than 30k tokens in one request
    # Luckily only 3 of my 176 documents are little bit over this.
    # Instead of skipping them i trim the reasoning from the end - thats where "poučenie o odvolaní, procesne opucenia"  usually is (boilerplate instructions about appeal
    # rights that have nothing to do with contractual penalties)
    GPT4O_TOKEN_LIMIT = 29500  # small buffer under the 30k limit
    reasoning = retrieval_data["reasoning_full"]

    if model_key == "gpt-4o":
        full_text = system_prompt + retrieval_data["header"] + retrieval_data["verdict"] + reasoning
        token_count = count_tokens(full_text)

        if token_count > GPT4O_TOKEN_LIMIT:
            # encode the reasoning into tokens, cut from end, decode back.
            # tiktoken handle Slovak chars correctly so no broken words.
            reasoning_tokens = _tokenizer.encode(reasoning)     # want to trim reasoning - no prompt  _ "hellow world" to tokens like [1234, 5678, 999]
            other_tokens = token_count - len(reasoning_tokens)
            max_reasoning_tokens = GPT4O_TOKEN_LIMIT - other_tokens
            reasoning_tokens = reasoning_tokens[:max_reasoning_tokens]
            reasoning = _tokenizer.decode(reasoning_tokens) # decode because i need to send text to LLM not tokens
            print(f"  TRIMMED reasoning to {max_reasoning_tokens:,} tokens (was {token_count:,}, limit {GPT4O_TOKEN_LIMIT:,})")
            retrieval_data = {**retrieval_data, "reasoning_full": reasoning}

    fulldoc_prompt = build_fulldoc_prompt(retrieval_data)   # just get fulldoc prompt
    token_count = count_tokens(system_prompt + fulldoc_prompt)
    print(f"  Full-doc: sending {len(fulldoc_prompt):,} chars ({token_count:,} tokens) to {model_key}...")

    response = call_llm(model_key, system_prompt, fulldoc_prompt, call_type="fulldoc")  # call llm through my wrapper
    # again saving meta - excluding real response text
    meta = dict(response)
    meta.pop("response_text", None)

    result, error = parse_llm_response(response["response_text"])
    if error:
        print(f"  ERROR parsing response: {error}")
        return None, meta

    print(f"   Done: {response['prompt_tokens']} prompt + {response['completion_tokens']} completion tokens")

    # FIX 2+5: i run the same evidence grounding check on fulldoc results too.
    # in fulldoc mode chunk_id is always "full_doc", so  lookup just maps "full_doc" to  entire reasoning text.
    # I also add header and verdict because  LLM might cite from those sections too
    chunk_lookup = {
        "full_doc": retrieval_data.get("reasoning_full", ""),
        "header": retrieval_data.get("header", ""),
        "verdict": retrieval_data.get("verdict", ""),
    }

    # FIX 5: in fulldoc mode the LLM returns  complete schema in one call.
    # but sometimes it forgets the quality_control section, so i add it here and run  same validation as RAG mode.
    # I didnt have this before and fulldoc results had no flags at all which made them look perfect when they actually had the same problems as RAG...
    if "quality_control" not in result:
        result["quality_control"] = {"flags": [], "missing_fields": []}

    from src.extraction.response_parser import validate_extraction
    flags, missing = validate_extraction(result, chunk_lookup=chunk_lookup)
    result["quality_control"]["flags"].extend(flags)        # append adds all object as one element - extend adds elements from list one by one
    result["quality_control"]["missing_fields"].extend(missing)

    return result, meta


# ##################################
# MAIN LOOP through all documents
# # ##################################

def main():
    import time
    pipeline_start = time.time()

    # FIX: recompute OUTPUT_DIR from current MODEL_KEY/MODE.
    # If  caller modified r.MODEL_KEY or r.MODE after import, the module-level OUTPUT_DIR is stale and points to the wrong folder
    # Doing it here makes overrides work correctly.
    global OUTPUT_DIR
    OUTPUT_DIR = f"data/07_extractions/{MODEL_KEY}_{MODE}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # creates folder if dont exist

    # USER FRIENDLY printing
    print(f"=== EXTRACTION PIPELINE ===")
    print(f"Model: {MODEL_KEY}  |  Mode: {MODE}")
    print(f"Input: {RETRIEVAL_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # get list of retrieval result files
    if SINGLE_DOC_TEST:
        files = [os.path.join(RETRIEVAL_DIR, f"{SINGLE_DOC_TEST}.json")]    # join path directory with name of doc and creates list
        print(f"TEST MODE: processing only {SINGLE_DOC_TEST}\n")
    elif GOLDEN_ONLY:
        # filter to only the 20 golden dataset documents so i dont spend money on all 176 while still debugging extraction pipeline
        import pandas as pd
        golden_csv = "data/05_retrieval_evaluation/golden_dataset_template.csv"
        golden = pd.read_csv(golden_csv, sep=";")
        golden_names = set(golden["document_name"].str.replace(".json", "").tolist())
        all_files = sorted(glob.glob(os.path.join(RETRIEVAL_DIR, "*.json")))
        files = [f for f in all_files if os.path.basename(f).replace(".json", "") in golden_names]
        print(f"GOLDEN ONLY: {len(files)} documents (out of {len(all_files)} total)\n")
    else:
        files = sorted(glob.glob(os.path.join(RETRIEVAL_DIR, "*.json")))    # glob.glob(...) finds files based on pattern
        print(f"Found {len(files)} documents to process.\n")

    results_log = []

    ##  main for loop
    for filepath in tqdm(files, desc=f"Extracting ({MODEL_KEY}/{MODE})"):   # showing progress bar
        doc_name = os.path.basename(filepath).replace(".json", "")  # name of doc
        output_path = os.path.join(OUTPUT_DIR, f"{doc_name}.json")  # path for output of extraction

        # skip if already extracted (for resuming interrupted runs)
        if os.path.exists(output_path) and not SINGLE_DOC_TEST:
            continue

        # load precomputed retrieval data
        with open(filepath, "r", encoding="utf-8") as f:
            retrieval_data = json.load(f)

        print(f"\n--- {retrieval_data['case_id']} ({doc_name}) ---")

        # run extraction based on mode
        if MODE == "rag":
            result, meta = run_rag_extraction(retrieval_data, MODEL_KEY)
        elif MODE == "fulldoc":
            result, meta = run_fulldoc_extraction(retrieval_data, MODEL_KEY)
        else:
            raise ValueError(f"Unknown MODE: {MODE}. Use 'rag' or 'fulldoc'.")

        if result is None:
            print(f"  FAILED - no result for {doc_name}")
            results_log.append({"doc": doc_name, "status": "FAILED"})
            continue

        # FIX: meta fields (court, case_id, date, ecli) come from REGEX extraction in data_cleaner.py,not from the LLM.
        # Regex is deterministic and  reliable — the LLM sometimes reformats date or gets case number wrong (also saves tokens)
        # I just copy the metadata from  precomputed retrieval JSON (which already has it from the processed JSON).
        doc_metadata = retrieval_data.get("document_metadata", {})
        result["meta"] = {
            "doc_id": doc_name,
            "court_name": doc_metadata.get("court", ""),
            "case_number": doc_metadata.get("case_id", ""),
            "decision_date": doc_metadata.get("date_raw", ""),
            "ecli": doc_metadata.get("ecli", None),
            "document_type": doc_metadata.get("document_type", ""),
        }

        # save extraction result
        output = {
            "extraction_config": {
                "model": MODEL_KEY,
                "mode": MODE,
                "retrieval_config": retrieval_data.get("retrieval_config", {}),
            },
            "api_metadata": meta,
            "result": result,   # extraction result
        }

        # create and write result file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)  # writes output into file with json format

        # calculate total cost and tokens (if rag then call1 + call2)
        if MODE == "rag":
            total_cost = (meta.get("call1", {}).get("cost_usd") or 0) + (meta.get("call2", {}).get("cost_usd") or 0)
            total_tokens = (meta.get("call1", {}).get("prompt_tokens", 0) + meta.get("call1", {}).get("completion_tokens", 0) +
                          meta.get("call2", {}).get("prompt_tokens", 0) + meta.get("call2", {}).get("completion_tokens", 0))
        else:
            total_cost = meta.get("cost_usd") or 0
            total_tokens = (meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0))

        results_log.append({
            "doc": doc_name,
            "status": "OK",
            "tokens": total_tokens,
            "cost": total_cost,
            "flags": result.get("quality_control", {}).get("flags", []),
        })

        print(f"  Saved to {output_path}")
        if result.get("quality_control", {}).get("flags"):
            print(f"  FLAGS: {result['quality_control']['flags']}")

    # summary
    elapsed = time.time() - pipeline_start
    elapsed_min = elapsed / 60
    elapsed_hr = elapsed / 3600



    ####### JUST QUITE NICE LOG INTO TERMINAL #####3
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION DONE — {len(results_log)} documents")

    # count how many documents finished successfully / failed
    ok_count = 0
    fail_count = 0

    # sum total API cost and total token usage across all processed documents
    total_cost = 0
    total_tokens = 0

    for result_info in results_log:
        if result_info["status"] == "OK":
            ok_count += 1
        if result_info["status"] == "FAILED":
            fail_count += 1

        total_cost += result_info.get("cost", 0)
        total_tokens += result_info.get("tokens", 0)

    print(f"OK: {ok_count}  |  Failed: {fail_count}")
    print(f"Total tokens: {total_tokens:,}  |  Total cost: ${total_cost:.4f}")

    # time tracking
    if elapsed_hr >= 1:
        print(f"Total time: {elapsed_hr:.1f} hours ({elapsed_min:.0f} min)")
    else:
        print(f"Total time: {elapsed_min:.1f} minutes ({elapsed:.0f} seconds)")

    if results_log:
        avg_per_doc = elapsed / len(results_log)
        print(f"Average per document: {avg_per_doc:.1f} seconds")

    # collect only documents that have at least one quality-control flag
    flagged = []

    for result_info in results_log:
        if result_info.get("flags"):
            flagged.append(result_info)

    if flagged:
        print(f"\nDocuments with flags:")
        for result_info in flagged:
            print(f"  {result_info['doc']}: {result_info['flags']}")


if __name__ == "__main__":
    main()
