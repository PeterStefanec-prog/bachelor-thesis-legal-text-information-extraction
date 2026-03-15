import os
import json
import glob
from tqdm import tqdm  # for nice progress bar in terminal

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from transformers import AutoTokenizer

# --- 1. PATH SETUP ---
#  input and output folders.
# hah run this script from the ROOT project folder (not from inside src/)   -  python src/chunking/chunk_processor.py
INPUT_DIR = "data/02_processed_json"
OUTPUT_DIR = "data/03_chunked_docs"

# do output dir if it does not exist yet
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. TOKENIZER SETUP ---
# FIX: my original script used length_function=len which counts CHARACTERS not TOKENS    - just for fast work i assumed that one token is 4 chars
# but it could do wrong stuff because of truncation limit
# mE5-small has hard limit of 512 tokens and silently truncates without any warning or error
# So i switched to token-based length functions using the exact tokenizer of each model.
print("Loading tokenizers...")

# OpenAI tokenizer - tiktoken is  official Openai tokenizer library - written in model card
openai_encoding = tiktoken.encoding_for_model("text-embedding-3-small")

# mE5 tokenizer - i want the exact same tokenizer that the embedding model uses internally
# add_special_tokens=False because i want raw text token count only, not with [CLS] [SEP]
me5_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")


def openai_token_len(text: str) -> int:
    return len(openai_encoding.encode(text))


def me5_token_len(text: str) -> int:
    return len(me5_tokenizer.encode(text, add_special_tokens=False))


# --- 3. SEPARATOR LIST ---
# Shared for both strategies. Order matters - LangChain tries from top to bottom
# and only moves to the next separator if the chunk is still too big.
#
# FIX: I removed the sentence dot splitter that was in the original script:
#   r"(?<=[a-zA-Zá-žÁ-Ž])\.\s+(?=[A-ZÁ-Ž])"
# The idea was good (split at end of sentences) but it also fires on every Slovak
# legal title like JUDr. Mgr. Ing. MUDr. and on Z.z. in law citations,
# because they all match: letter + dot + space + uppercase.
# This created micro-chunks like ["JUDr", "Martin Vladik decided that..."].
# After data_cleaner.py the reasoning text already has \n\n between paragraphs
# so \n\n, \n and ; handle everything without this risky regex.
#
# FIX: also removed comma r",\s+" separator.
# comma splits are too granular and destroy sentence context. example:
# "Súd dospel k záveru, že pokuta je neprimeraná" would become
# ["Súd dospel k záveru", "že pokuta je neprimeraná"] - both chunks are useless alone.
# comma was separator #5 out of 6 anyway, meaning it only triggered when \n\n, \n
# and ; all failed AND chunk was still too big - which basically never happens in
# cleaned legal text. so i removed it - no benefit, only risk.
#
# NOTE on (?=\n\s*\d+\.\s+):
# This is a zero-width lookahead - it matches a position but consumes no characters.
# Because of this, keep_separator has no effect on it (there is nothing to keep).
# The numbered item naturally ends up at the START of the new chunk.
# This is the correct behavior and it does NOT depend on keep_separator.

SEPARATORS = [
    r"\n\n+",                   # 1. Paragraph break - strongest boundary, always try first
    r"(?=\n\s*\d+\.\s+)",       # 2. Before numbered items like "1." "12." (zero-width, so item goes to START of new chunk)
    r"\n",                      # 3. Any single newline
    r";\s+",                    # 4. Semicolons - common in long legal enumerations, safe to split here
    r"\s+",                     # 5. Whitespace - absolute last resort, should almost never happen
]

# --- 4. CHUNKING STRATEGIES ---

# STRATEGY A: For OpenAI API (text-embedding-3-small)
# model limit is 8191 tokens so there is huge room.
# BUT big chunks are bad for retrieval quality! embedding of 2000-token chunk is basically
# a blurry average of everything inside - a specific penalty amount gets completely diluted.
# we want "sharp spears not wide nets" for retrieval.
# chunk_size=500 is much more precise. when retriever finds the right small chunk,
# evaluate_retrieval.py with window expansion (+-1 chunk) pulls in surrounding context anyway
# so LLM still gets the full picture. best of both worlds.
# overlap=75 tokens = 15% of chunk_size -> within spec range (10-20%)
text_splitter_openai = RecursiveCharacterTextSplitter(
    separators=SEPARATORS,
    chunk_size=500,         # in TOKENS (because length_function=openai_token_len)
    chunk_overlap=75,       # in TOKENS
    length_function=openai_token_len,
    is_separator_regex=True,    # Critical: must be True for my regex separators to work
    keep_separator="end",       # Keeps \n\n at the END of the chunk not at START of next one
)

# STRATEGY B: For local mE5-small (open-source model)
# Model has strict 512 token limit and SILENTLY TRUNCATES without any error!
# chunk_size=380 gives reserve of 132 tokens.
# Even chunk + overlap = 380 + 50 = 430 tokens max, still safely under 512.
# overlap=50 tokens = 13.2% of chunk_size -> within spec range (10-20%)
text_splitter_me5 = RecursiveCharacterTextSplitter(
    separators=SEPARATORS,
    chunk_size=380,         # in TOKENS (because length_function=me5_token_len)
    chunk_overlap=50,       # in TOKENS
    length_function=me5_token_len,
    is_separator_regex=True,
    keep_separator="end",
)

# --- 5. HELPER FUNCTION ---
# This takes text, makes chunks and packs each one with metadata.
def process_chunks(splitter, text, metadata, strategy_name, filename, token_len_fn):
    raw_chunks = splitter.split_text(text)
    final_chunks = []

    for i, chunk_txt in enumerate(raw_chunks):
        chunk_txt = chunk_txt.strip()

        # skip empty chunks (can happen after stripping whitespace-only splits)
        if not chunk_txt:
            continue

        # copy() is important so I dont overwrite the original metadata dict!
        # Without copy() all chunks would share the same dict reference and chunk_index
        # would be overwritten for all of them on every iteration.
        chunk_meta = metadata.copy()
        chunk_meta["chunk_index"] = i
        chunk_meta["source_file"] = filename
        chunk_meta["strategy"] = strategy_name  # So I know which model to use later
        chunk_meta["char_len"] = len(chunk_txt)
        chunk_meta["token_len"] = token_len_fn(chunk_txt)  # Store actual token count for debugging

        final_chunks.append({
            "page_content": chunk_txt,
            "metadata": chunk_meta,
        })

    return final_chunks


# --- 6. MAIN PIPELINE ---
def main():
    # Find all my clean json files
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

    if not json_files:
        print(f"Error: No files found in {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} docs. Starting chunking process...\n")

    # Stats for my thesis report
    # FIX: Split into two separate counters instead of one "failed_files".
    # Before I was mixing two very different failure types into one number which
    # made it impossible to tell if files had structural issues or actual errors.
    total_openai_chunks = 0
    total_me5_chunks = 0
    skipped_no_reasoning = 0   # document exists but has no reasoning segment
    failed_exception = 0       # something actually crashed

    # tqdm makes a nice progress bar in terminal
    for file_path in tqdm(json_files, desc="Processing docs"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            # Skip documents without reasoning segment.
            # Some older or shortened decisions don't have it, or segmentation failed.
            # I log this separately so I can track how often it happens in my corpus.
            reasoning_text = doc_data.get("segments", {}).get("reasoning", "").strip()
            if not reasoning_text:
                skipped_no_reasoning += 1
                continue

            base_metadata = doc_data.get("metadata", {})
            filename = doc_data.get("filename", os.path.basename(file_path))

            # Make chunks for both strategies
            chunks_openai = process_chunks(
                text_splitter_openai,
                reasoning_text,
                base_metadata,
                "openai_api_token_based",
                filename,
                openai_token_len,
            )

            chunks_me5 = process_chunks(
                text_splitter_me5,
                reasoning_text,
                base_metadata,
                "me5_local_token_based",
                filename,
                me5_token_len,
            )

            # Update my stats
            total_openai_chunks += len(chunks_openai)
            total_me5_chunks += len(chunks_me5)

            # Clean filename so I dont get weird names like ".pdf_chunks"
            clean_name = filename.replace(".pdf", "").replace(".json", "")

            # Save OpenAI dataset
            path_openai = os.path.join(OUTPUT_DIR, clean_name + "_chunks_OPENAI.json")
            with open(path_openai, "w", encoding="utf-8") as f:
                json.dump(chunks_openai, f, ensure_ascii=False, indent=4)

            # Save mE5 dataset
            path_me5 = os.path.join(OUTPUT_DIR, clean_name + "_chunks_ME5.json")
            with open(path_me5, "w", encoding="utf-8") as f:
                json.dump(chunks_me5, f, ensure_ascii=False, indent=4)

        except Exception as e:
            # Catch errors so one bad file doesnt kill my whole script
            print(f"\nError in file {file_path}: {e}")
            failed_exception += 1

    # --- 7. FINAL REPORT ---
    processed = len(json_files) - skipped_no_reasoning - failed_exception
    print("\n=== CHUNKING PROCESS COMPLETED ===")
    print(f"Processed:               {processed} / {len(json_files)}")
    print(f"Skipped (no reasoning):  {skipped_no_reasoning}")
    print(f"Failed (exception):      {failed_exception}")
    print(f"-> STRATEGY A (OpenAI):  {total_openai_chunks} chunks")
    print(f"-> STRATEGY B (mE5):     {total_me5_chunks} chunks")


if __name__ == "__main__":
    main()