import os
import json
import glob
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from transformers import AutoTokenizer

# --- 1. PATH SETUP ---
# hah run this script from the ROOT project folder (not from inside src/)
# python src/chunking/chunk_processor.py
INPUT_DIR  = "data/02_processed_json"
OUTPUT_DIR = "data/03_chunked_docs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. TOKENIZER SETUP ---
# FIX: my original script used length_function=len which counts CHARACTERS not TOKENS
# i assumed one token = 4 chars which was wrong
# mE5-small has hard limit of 512 tokens and silently truncates without any warning or error
# so i switched to token-based length functions using the exact tokenizer of each model
print("Loading tokenizers...")

# OpenAI tokenizer - tiktoken is official Openai tokenizer library - written in model card
openai_encoding = tiktoken.encoding_for_model("text-embedding-3-small")

# mE5 tokenizer - i want the exact same tokenizer that the embedding model uses internally
# add_special_tokens=False because i want raw text token count only, not with [CLS] [SEP]
#               changed me5_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
me5_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")


def openai_token_len(text: str) -> int:
    return len(openai_encoding.encode(text))


def me5_token_len(text: str) -> int:
    return len(me5_tokenizer.encode(text, add_special_tokens=False))


# --- 3. SEPARATOR LISTS ---

# STANDARD separators - used for all fixed-size token-based strategies
# order matters - langchain tries from top to bottom and only moves to the next
# separator if the chunk is still too big
#
# FIX: removed sentence dot splitter r"(?<=[a-zA-Zá-žÁ-Ž])\.\s+(?=[A-ZÁ-Ž])"
# it fires on JUDr. Mgr. Ing. MUDr. and Z.z. in law citations which all match
# letter + dot + space + uppercase -> created micro-chunks like ["JUDr", "Martin..."]
#
# FIX: removed comma separator r",\s+"
# comma splits destroy sentence context. "Súd dospel k záveru, že pokuta je neprimeraná"
# would split into ["Súd dospel k záveru", "že pokuta je neprimeraná"] - both useless alone
#
# NOTE on (?=\n\s*\d+\.\s+):
# this is a zero-width lookahead - matches a position but consumes no characters
# so keep_separator has no effect on it (nothing to keep)
# the numbered item naturally ends up at the START of the new chunk - correct behavior

SEPARATORS_STANDARD = [
    r"\n\n+",              # 1. paragraph break - strongest boundary, always try first
    r"(?=\n\s*\d+\.\s+)", # 2. before numbered items like "1." "12." (zero-width)
    r"\n",                 # 3. any single newline
    r";\s+",               # 4. semicolons - common in long legal enumerations, safe to split here
    r"\s+",                # 5. whitespace - absolute last resort, should almost never happen
]

# PARAGRAPH separators - only for the paragraph-based OpenAI strategy
# numbered legal points come FIRST because they are the natural unit of Slovak court decisions
# each point = one argument = one chunk
# double newlines come second for cases where one point has multiple sub-paragraphs
SEPARATORS_PARAGRAPH = [
    r"(?=\n\s*\d+\.\s+)", # 1. numbered points - PRIMARY boundary
    r"\n\n+",              # 2. paragraph breaks within a point
    r"\n",                 # 3. single newlines - only if still too long
]


# --- 4. CHUNKING STRATEGIES ---
# i am comparing 5 different strategies to find the best one for this type of text
# the key variables are: which model, how big each chunk is, how much overlap
#
# ME5_200:
#   very fine-grained chunks - each chunk covers roughly half a legal paragraph
#   more chunks = more precise retrieval but more boundary splits
#   overlap 25 tokens = 12.5% - safe margin, 200+25=225 max, well under mE5 512 limit
#
# ME5_380:
#   current baseline - one chunk per legal paragraph roughly
#   overlap 50 = 13.2%, 380+50=430 max tokens, still under 512
#
# OPENAI_200:
#   same chunk size as ME5_200 but measured in OpenAI tokens
#   good for fair comparison between models at the same granularity
#   overlap 25 = 12.5%
#
# OPENAI_500:
#   current baseline for OpenAI - same as what i had before, just renamed
#   overlap 75 = 15%
#
# OPENAI_PARA:
#   paragraph-based strategy - chunk = one numbered point from the court decision
#   this is the most natural split for Slovak legal text because judges write
#   their reasoning as numbered points (1. 2. 3. ...) - each point = one argument
#
#   overlap=0 here! paragraphs are already complete natural units. overlap is a
#   workaround for artificial token-window boundaries. if i add overlap here i would
#   be copying the end of paragraph 3 into paragraph 4 which makes no sense - they
#   are separate legal arguments. so no overlap.
#
#   chunk_size=1500 is just a safety net - some judges write very long single points.
#   in most cases the numbered point separator fires long before we hit 1500 tokens.
#
#   ONLY for OpenAI! typical Slovak legal paragraph is 300-800 tokens in mE5 tokenizer.
#   mE5 has a hard limit of 512 tokens and would silently truncate most paragraphs.

text_splitter_me5_200 = RecursiveCharacterTextSplitter(
    separators=SEPARATORS_STANDARD,
    chunk_size=200,
    chunk_overlap=25,
    length_function=me5_token_len,
    is_separator_regex=True,
    keep_separator="end",
)

text_splitter_me5_380 = RecursiveCharacterTextSplitter(
    separators=SEPARATORS_STANDARD,
    chunk_size=380,
    chunk_overlap=50,
    length_function=me5_token_len,
    is_separator_regex=True,
    keep_separator="end",
)

text_splitter_openai_200 = RecursiveCharacterTextSplitter(
    separators=SEPARATORS_STANDARD,
    chunk_size=200,
    chunk_overlap=25,
    length_function=openai_token_len,
    is_separator_regex=True,
    keep_separator="end",
)

text_splitter_openai_500 = RecursiveCharacterTextSplitter(
    separators=SEPARATORS_STANDARD,
    chunk_size=500,
    chunk_overlap=75,
    length_function=openai_token_len,
    is_separator_regex=True,
    keep_separator="end",
)

text_splitter_openai_para = RecursiveCharacterTextSplitter(
    separators=SEPARATORS_PARAGRAPH,
    chunk_size=1500,    # safety net only - natural paragraph splits come first
    chunk_overlap=0,    # no overlap - paragraphs are complete natural units
    length_function=openai_token_len,
    is_separator_regex=True,
    keep_separator="end",
)


# --- 5. HELPER FUNCTION ---
def process_chunks(splitter, text, metadata, strategy_name, filename, token_len_fn):
    raw_chunks = splitter.split_text(text)
    final_chunks = []

    for i, chunk_txt in enumerate(raw_chunks):
        chunk_txt = chunk_txt.strip()

        if not chunk_txt:
            continue

        # copy() is important so i dont overwrite the original metadata dict!
        # without copy() all chunks would share the same dict reference and chunk_index
        # would be overwritten for all of them on every iteration
        #
        # FIX: use len(final_chunks) instead of i for chunk_index.
        # i comes from enumerate() and does NOT decrement when we skip empty chunks above.
        # if splitter ever produces a whitespace-only chunk, i would jump (0,1,3,4...)
        # and window expansion in evaluate_retrieval.py would silently miss neighbors.
        chunk_meta = metadata.copy()
        chunk_meta["chunk_index"] = len(final_chunks)
        chunk_meta["source_file"] = filename
        chunk_meta["strategy"]    = strategy_name
        chunk_meta["char_len"]    = len(chunk_txt)
        chunk_meta["token_len"]   = token_len_fn(chunk_txt)

        final_chunks.append({
            "page_content": chunk_txt,
            "metadata":     chunk_meta,
        })

    return final_chunks


# --- 6. MAIN PIPELINE ---
def main():
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

    if not json_files:
        print(f"Error: No files found in {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} docs. Starting chunking for all 5 strategies...\n")

    counts = {
        "ME5_200":     0,
        "ME5_380":     0,
        "OPENAI_200":  0,
        "OPENAI_500":  0,
        "OPENAI_PARA": 0,
    }
    skipped_no_reasoning = 0
    failed_exception = 0

    # all 5 strategies in one loop so i only read each file once
    strategies = [
        (text_splitter_me5_200,    "me5_200",    "ME5_200",    me5_token_len),
        (text_splitter_me5_380,    "me5_380",    "ME5_380",    me5_token_len),
        (text_splitter_openai_200, "openai_200", "OPENAI_200", openai_token_len),
        (text_splitter_openai_500, "openai_500", "OPENAI_500", openai_token_len),
        (text_splitter_openai_para,"openai_para","OPENAI_PARA",openai_token_len),
    ]

    for file_path in tqdm(json_files, desc="Processing docs"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            # skip documents without reasoning segment
            # some older decisions dont have it or segmentation failed
            reasoning_text = doc_data.get("segments", {}).get("reasoning", "").strip()
            if not reasoning_text:
                skipped_no_reasoning += 1
                continue

            base_metadata = doc_data.get("metadata", {})
            filename  = doc_data.get("filename", os.path.basename(file_path))
            clean_name = filename.replace(".pdf", "").replace(".json", "")

            for splitter, strategy_name, suffix, token_fn in strategies:
                chunks = process_chunks(
                    splitter, reasoning_text, base_metadata,
                    strategy_name, filename, token_fn
                )
                counts[suffix] += len(chunks)

                out_path = os.path.join(OUTPUT_DIR, f"{clean_name}_chunks_{suffix}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"\nError in {file_path}: {e}")
            failed_exception += 1

    # --- 7. FINAL REPORT ---
    processed = len(json_files) - skipped_no_reasoning - failed_exception
    print("\n=== CHUNKING PROCESS COMPLETED ===")
    print(f"Processed:               {processed} / {len(json_files)}")
    print(f"Skipped (no reasoning):  {skipped_no_reasoning}")
    print(f"Failed (exception):      {failed_exception}")
    print()
    print("Chunks per strategy:")
    for suffix, count in counts.items():
        avg = round(count / processed, 1) if processed > 0 else 0
        print(f"  {suffix:<15} {count:>6} total  (avg {avg} per doc)")


if __name__ == "__main__":
    main()