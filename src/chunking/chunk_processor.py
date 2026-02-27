import os
import json
import glob
from tqdm import tqdm  # for nice progress bar in terminal
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. PATH SETUP ---
# My input and output folders.
# Important: I should run this script from the ROOT project folder (not from inside src/)
# So my terminal command should be: python src/chunking/chunk_processor.py
INPUT_DIR = "data/02_processed_json"
OUTPUT_DIR = "data/03_chunked_docs"

# Make output dir if it does not exist yet
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. CHUNKING STRATEGIES ---

# STRATEGY A: For OpenAI API (text-embedding-3-small)
# Big context window (8192 tokens), so I can just split by whole paragraphs (\n\n).
# No need to break sentences.
text_splitter_openai = RecursiveCharacterTextSplitter(
    separators=[r"\n\n"],
    chunk_size=4000,      # Huge size, very safe for API
    chunk_overlap=200,    # Little overlap just in case
    length_function=len,
    is_separator_regex=True,
    keep_separator="end"  # BUGFIX: Keeps \n\n at the end of the chunk, not at the start of the next one
)

# STRATEGY B: For local mE5-small (open-source model)
# It has strict 512 tokens limit. 1200 chars is safe zone for Slovak language.
text_splitter_me5 = RecursiveCharacterTextSplitter(
    separators=[
        r"\n\n",                                  # 1. Paragraph first
        r"\n",                                    # 2. New line
        r"(?<=[a-zA-Zá-žÁ-Ž])\.\s+(?=[A-ZÁ-Ž])",  # 3. Smart dot - cuts only end of sentence! Not "1." or "Z.z."
        r";\s+",                                  # 4. Semicolon for long legal lists
        r",\s+",                                  # 5. Comma - saves decimals like "0,5"
        r"\s+"                                    # 6. Space as last option
    ],
    chunk_size=1200,      # Strict limit for local model!
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=True, # Critical: Must be true for smart regex to work
    keep_separator="end"     # BUGFIX: This forces the dot to stay exactly at the end of the sentence!
)

# --- 3. HELPER FUNCTION ---
# This takes text, makes chunks and packs it with metadata.
def process_chunks(splitter, text, metadata, strategy_name, filename):
    raw_chunks = splitter.split_text(text)
    final_chunks = []

    for i, chunk_txt in enumerate(raw_chunks):
        # copy() is important so I dont overwrite the original metadata dict
        chunk_meta = metadata.copy()
        chunk_meta["chunk_index"] = i
        chunk_meta["source_file"] = filename
        chunk_meta["strategy"] = strategy_name # So I know which model to use later

        final_chunks.append({
            "page_content": chunk_txt,
            "metadata": chunk_meta
        })
    return final_chunks

# --- 4. MAIN PIPELINE ---
def main():
    # Find all my clean json files
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

    if not json_files:
        print(f"Error: No files found in {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} docs. Starting chunking process...\n")

    # Stats for my thesis report
    total_openai_chunks = 0
    total_me5_chunks = 0
    failed_files = 0

    # tqdm makes a nice progress bar
    for file_path in tqdm(json_files, desc="Processing docs"):
        try:
            # Open and read json
            with open(file_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            # Check if document has reasoning section. If not, skip it.
            if "segments" not in doc_data or "reasoning" not in doc_data["segments"]:
                # commented out so it doesnt spam terminal
                # print(f"\nSkipping {file_path}: No reasoning segment.")
                failed_files += 1
                continue

            reasoning_text = doc_data["segments"]["reasoning"]
            base_metadata = doc_data.get("metadata", {})
            filename = doc_data.get("filename", os.path.basename(file_path))

            # Make chunks for both strategies
            chunks_openai = process_chunks(text_splitter_openai, reasoning_text, base_metadata, "openai_api_4000", filename)
            chunks_me5 = process_chunks(text_splitter_me5, reasoning_text, base_metadata, "me5_local_1200", filename)

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
            failed_files += 1

    # --- 5. FINAL REPORT ---
    print("\n=== CHUNKING PROCESS COMPLETED ===")
    print(f"Processed docs: {len(json_files) - failed_files} / {len(json_files)}")
    print(f"Fails: {failed_files}")
    print(f"-> STRATEGY A (OpenAI): Created {total_openai_chunks} chunks.")
    print(f"-> STRATEGY B (mE5): Created {total_me5_chunks} chunks.")

if __name__ == "__main__":
    main()