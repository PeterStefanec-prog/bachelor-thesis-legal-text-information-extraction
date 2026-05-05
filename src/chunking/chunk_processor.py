# INPUT clean reasoning text (data/02_processed_json)
# OUTPUT chunks for retrieval (data/03_chunked_docs)

# cd /Users/stefanec/STU_FIIT/bachelor-thesis-legal-text-information-extraction
# .venv/bin/python src/chunking/chunk_processor.py

# only 20 golden docs
# .venv/bin/python src/chunking/chunk_processor.py --golden-only

import os
import re
import json
import glob
import sys
from tqdm import tqdm # progress bar

from langchain_text_splitters import RecursiveCharacterTextSplitter # splitter from langchain library
    # (python framework for eg loading docs, splitting text on chunks, embeddings, searching in vector DB, connecting to LLM, RAG pipeline,...)
    # i use it for recurstive splitting test - keepin natural text 'borders'
import tiktoken # tokenizer for OpenAi embedding modl
from transformers import AutoTokenizer # tokenizer for me5 model - from hugging face

# fix: need project root in path so i can import from src.candidates
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ####################### 1. PATH SETUP CONFIG ######################
# hah run this script from the ROOT project folder (not from inside src/)
# python src/chunking/chunk_processor.py
INPUT_DIR  = "data/02_processed_json"   # cleaned text, segments, metadata
OUTPUT_DIR = "data/03_chunked_docs"     # flat chunks
#########################################################

os.makedirs(OUTPUT_DIR, exist_ok=True)  # just check, if foldr doesnt exist, it creates new

# ### 2. TOKENIZER SETUP ####
# big fix: my original script used length_function=len which counts CHARACTERS not TOKENS
# for simplificaation i assumed one token = 4 chars which was wrong
# mE5-small has hard limit of 512 tokens and silently truncates without any warning or error  (1000 chars can have 150tokens but 400 as well)
# so i switched to token-based length functions using the exact tokenizer of each model
print("Loading tokenizers..")

# OpenAI tokenizer - tiktoken is official Openai tokenizer library - written in model card
openai_encoding = tiktoken.encoding_for_model("text-embedding-3-small")

# mE5 tokenizer - i want  exact same tokenizer that the embedding model uses internally
# add_special_tokens=False because i want raw text token count only, not with [CLS] [SEP]
#     changed me5_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
me5_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")


# function that returns number of tokens of text
def openai_token_len(text: str) -> int:
    return len(openai_encoding.encode(text))


def me5_token_len(text: str) -> int:
    return len(me5_tokenizer.encode(text, add_special_tokens=False)) # [CLS] and [SEP] are special tokens, which are added by NLP models for their structure
    # [CLS] This is sentence . [SEP]


# ## 3. SEPARATOR LISTS ####

# STANDARD separators - used for all fixed-size token-based strategies
#   order matters - langchain tries from top to bottom and only moves to next separator if the chunk is still too big
#
# FIX: removed sentence dot splitter r"(?<=[a-zA-Zá-žÁ-Ž])\.\s+(?=[A-ZÁ-Ž])" it fires on JUDr. Mgr. Ing. MUDr. and Z.z. in law citations
#   which all match letter + dot + space + uppercase -> created micro-chunks like ["JUDr", "Martin..."]
#
# FIX: removed comma separator r",\s+"
#   comma splits destroy sentence context. "Súd dospel k záveru, že pokuta je neprimeraná"
#       would split into ["Súd dospel k záveru", "že pokuta je neprimeraná"] - both useless alone
#
# NOTE on (?=\n\s*\d+\.\s+):  this is a zero-width lookahead - matches a position but consumes no characters
# so keep_separator has no effect on it (nothing to keep) numbered item naturally ends up at the START of the new chunk - correct

SEPARATORS_STANDARD = [
    r"\n\n+",              # 1. paragraph break - strongest boundary, always try first
    r"(?=\n\s*\d+\.\s+)", # 2. before numbered items like "1." "12." (zero-width)
    r"\n",                 # 3. any single newline
    r";\s+",               # 4. semicolons - common in long legal enumerations, safe to split here
    # r"(?<=[^A-Z])\.\s+(?=[A-Z])",  # sentence.    (but not  JUDr. Mgr.)
    r"\s+",                # 5. whitespace - absolute last resort, should almost never happen
]

# PARAGRAPH separators -  do not have numbered items spilitter (because this is used when already having paragraph)
#           only used as SAFETY NET for splitting extremely long numbered points that exceeds OpenAIs 8192 token embedding context window
#           fix - that exceeds 1500 tokens - more than 8k is really extreme
#  main split is done by split_reasoning_to_paragraphs() which cuts at each numbered point boundary (1. 2. 3. ...).
# this separator list is only needed when a single point is so long that it would be truncated by  embedding API.
# in practice, only 8 out of 3581 points (0.2%) in our dataset need this - talking about 8k
SEPARATORS_PARAGRAPH = [
    r"\n\n+",              # 1. double newline - sub-paragraph breaks within a long point
    r"\n",                 # 2. single newline
    r";\s+",               # 3. semicolons - common in long legal enumerations
    r"\s+",                # 4. whitespace - absolute last resort
]


# ### 4. CHUNKING STRATEGIES ###
# i am comparing 5 different strategies to find  best one for this type of text
#  key variables are: which model, how big each chunk is, how much overlap
#
# ###### ME5_200: ######
#   very finegrained chunks - each chunk covers roughly half  legal paragraph
#   more chunks = more precise retrieval but more boundary splits
#   overlap 25 tokens = 12.5% - safe margin, 200+25=225 max, well under mE5 512 limit
#
# ###### ME5_380: ######
#   current baseline - one chunk per legal paragraph roughly
#   overlap 50 = 13.2%, 380+50=430 max tokens, still under 512
#
# ###### OPENAI_200: ######
#   same chunk size as ME5_200 but measured in OpenAI tokens
#   good for fair comparison between models at the same granularity
#   overlap 25 = 12.5%
#
#  ###### OPENAI_500: ######
#   current baseline for OpenAI - same as what i had before, just renamed
#   overlap 75 = 15%
#
# ###### OPENAI_PARA: ######
#   TRUE paragraph-based chunking - one numbered point = one chunk.
#
#   Slovak court decisions are structured as numbered points (1. 2. 3. ...)
#   where each point is one complete legal argument. this is the most natural  way to split them
#   - i respect the structure the judge wrote
#
#   PREVIOUS VERSION used RecursiveCharacterTextSplitter with chunk_size=1500 which MERGED multiple small points into one chunk
#   (e.g. points 18-25 in one chunk of 1384 tokens).
#   this was wrong because each point has its own topic -
#   merging "plaintiff's appeal" with "court costs" into one chunk creates  mixed  embedding that matches neither query well.
#
#   CURRENT VERSION uses split_reasoning_to_paragraphs() which:
#   1. splits at each "N. " numbered point boundary using regex
#   2. keeps each point as its own chunk, no matter how small (even 50 tokens)
#   3. only splits further if a single point exceeds 8000 tokens (OpenAI 8192 limit)
#   4. for documents without numbered points (34% of dataset), falls back to
#      splitting at double-newline (\n\n) paragraph boundaries
#
#   NO OVERLAP is NEEDED - paragraphs are complete natural units, not artificial windows.
#
#   ONLY for OpenAI embeddings. mE5 has 512 token limit and would truncate most
#   paragraphs (typical Slovak legal paragraph is 300-800 tokens).


##############################################
##############################################
############## EXPERIMETNS ###################
##############################################
text_splitter_me5_200 = RecursiveCharacterTextSplitter( # creating instance
    separators=SEPARATORS_STANDARD,
    chunk_size=200,
    chunk_overlap=25,
    length_function=me5_token_len,
    is_separator_regex=True,
    keep_separator="end",   # if splitting on newline - paste it to the end of chunk (more natural)
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

# Added later for fair embedding-model ablation: same chunk size (380 tok) as
# the mE5-base baseline, so we can compare mE5 vs OpenAI-S vs OpenAI-L at matched
# chunking. mE5-base maxes out at 512 tokens, so 380 is the largest fair size.
text_splitter_openai_380 = RecursiveCharacterTextSplitter(
    separators=SEPARATORS_STANDARD,
    chunk_size=380,
    chunk_overlap=50,
    length_function=openai_token_len,
    is_separator_regex=True,
    keep_separator="end",
)

# ANOTHER ONE ON LINE ~340


# safety-net splitter for extremely long individual points (>8000 tokens)
# OpenAI text-embedding-3-small has 8192 token context window.
# points exceeding this limit would be silent truncated by the API..
# only 8 out of 3581 paragraphs (0.2%) across all documents exceed this limit.
# most are "bod 1" entries that contain a summary of the entire first-instance ruling.
_para_safety_splitter = RecursiveCharacterTextSplitter(
    separators=SEPARATORS_PARAGRAPH,
    chunk_size=8000,    # just below OpenAI 8192 limit, with margin for safety
    chunk_overlap=0,
    length_function=openai_token_len,
    is_separator_regex=True,
    keep_separator="end",
)

# FIX: secondary sub-splitter for oversized OPENAI_PARA chunks (>2000 tokens).
# The original pipeline only split chunks >8000 tokens (OpenAI embedding limit).
# But 137 chunks across the dataset exceed 2000 tokens, and some exceed 5000.
# These mega-chunks contain multiple topics (contract details, § 301 analysis,interest, costs, etc.) all in one numbered point.
# The embedding for such chunk is an AVERAGE of all topics - it scores poorly on any single query.
#
# Example: eval_05 (2Cob/69/2020), chunk_3 = 5011 tokens, contains 7 different topics (penalty, deposit, insurance, set-off...).
# After this fix, it becomes ~3 focused chunks with better embedding precision.
#
# I set chunk_overlap=100 tokens so we dont lose context at the boundaries.
# The separators use SEPARATORS_PARAGRAPH which prefers splitting at sentence endings (". ", ";\n") to keep legal sentences intact.
MAX_PARA_TOKENS = 4000  # threshold for sub-splitting oversized paragraphs
# was 2000 but that was too aggressive - split 137 chunks and lost -1.58% recall
# at top_k=7 because info from 1 para was spread across 2-3 chunks.
# 4000 only splits ~44 truly mega paragraphs (0.8%) while keeping normal long legal points (2000-4000 tokens) intact for better retrieval coverage.
_para_subsplitter = RecursiveCharacterTextSplitter(
    separators=SEPARATORS_PARAGRAPH,
    chunk_size=MAX_PARA_TOKENS,
    chunk_overlap=200,  # raised from 100 - 5% of 4000, gives better boundary coverage
    length_function=openai_token_len,
    is_separator_regex=True,
    keep_separator="end",
)

# regex for splitting reasoning into numbered points (1. 2. 3. ...) also used by the hierarchical chunking below
POINT_SPLIT_RE = re.compile(r"(?m)^\s*(\d+)\.\s+")  # base for paragprah splitting and hierarchical splitting


def split_reasoning_to_paragraphs(reasoning_text):
    """Split reasoning text into true paragraph chunks based on numbered points.
    Each numbered point (1. 2. 3. ...) becomes itss own chunk.
    If no numbered points found, fall back to double-newline paragraph breaks.
    Very long points (>8000 tokens) get further split by the safety-net splitter to avoid exceeding OpenAIs 8192 token embedding limit."""
    matches = list(POINT_SPLIT_RE.finditer(reasoning_text))     # positions where new numbered paragraph starts
    # finditer() scans the whole text and gives  every match one by one
    # It does not return plain string, but  returns match objects.
    #
    # Each match object has: 1. what was found, 2. where it starts, 3. where it ends

    paragraphs = [] # for saving result paragraphs

    # fallback: no numbered points found, i have to split justby double newlines \n\n
    if not matches:
        for block in re.split(r"\n\n+", reasoning_text):
            clean_block = block.strip() # remove whitespaces (also newlines)
            if not clean_block:     # just noise skipping (empty blocks)
                continue
            words = re.findall(r"\S+", clean_block) # finds all words (non empty sequence of chars) (\S - char - not whitespace)
            if len(words) <= 1 and len(clean_block) <= 2:   # filter out things like ".", "," or ":" agter wrong splitting
                continue
            # FIX: lowered threshold from 8000 to MAX_PARA_TOKENS (4000)
            # Before: only split blocks >8000 tokens (almost never happened)
            # Now: also split blocks >4000 tokens which have diluted embeddings  because they cover multiple legal topics in one numbered point.
            if openai_token_len(clean_block) > MAX_PARA_TOKENS:
                paragraphs.extend(_para_subsplitter.split_text(clean_block))
            else:
                paragraphs.append(clean_block)  # just append block
        return paragraphs   # function ends here if paragprahps werent numbered

    # if numbered points found - split at each "number(1,2,3...). " boundary
    first_start = matches[0].start()
    if first_start > 0:
        intro_text = reasoning_text[:first_start].strip() # intro text example is - Odôvodnenie odvolacieho súdu
                                                        #1. Súd zistil...
        if intro_text:  # FIX: also use subsplitter threshold for intro
            if openai_token_len(intro_text) > MAX_PARA_TOKENS:
                paragraphs.extend(_para_subsplitter.split_text(intro_text))
            else:
                paragraphs.append(intro_text)


    ### MAIN LOOP going through all numbered segments ###
    for i, match in enumerate(matches):
        start = match.start()
        if i + 1 < len(matches):
            end = matches[i + 1].start()    # each paragraph goes from starting actual par to starting next par
        else:
            end = len(reasoning_text)   # or the end

        text = reasoning_text[start:end].strip()    # content of one paragrapph
        # for example
        # 2. Odvolací súd dospel k záveru, že ...
        # ....
        # ,,,

        if not text:
            continue    # if block is somehow empty

        # FIX: lowered from 8000 to MAX_PARA_TOKENS (2000).
        # Points exceeding  2000 tokens get sub-split into focused chunks with better embeddings.
        # The original 8000 threshold was only for the OpenAI embedding limit,
        # but even a 3000 token chunk has a diluted embedding that scores poorly on specific queries.
        if openai_token_len(text) > MAX_PARA_TOKENS:
            paragraphs.extend(_para_subsplitter.split_text(text))
        else:
            paragraphs.append(text)

    return paragraphs


# ### 5. HELPER FUNCTION  ### JUST FOR FIXED SIZE STRATEGIES!! NOT PARA!
def process_chunks(splitter, text, metadata, strategy_name, filename, token_len_functiom):
    raw_chunks = splitter.split_text(text) # splits text into chunks based on params of splitter object of type
    # raw_chunks = splitter.split_text(text)

    final_chunks = []

    for i, chunk_txt in enumerate(raw_chunks):
        chunk_txt = chunk_txt.strip()

        if not chunk_txt:
            continue

        # copy() is important so i dont overwrite the original metadata dict!
        # without copy() all chunks would share the same dict reference and chunk_inde would be overwritten for all of them on every iteration
        #
        # FIX: using len(final_chunks) instead of i for chunk_index.
        # i comes from enumerate() and does NOT decrement when we skip empty chunks above.
        # if splitter ever produces  whitespace-only chunk, i would jump (0,1,3,4...)
        # and window expansion in evaluate_retrieval.py would silently miss neighbors (sadly happened to me)
        chunk_meta = metadata.copy()
        chunk_meta["chunk_index"] = len(final_chunks)
        chunk_meta["source_file"] = filename
        chunk_meta["strategy"]    = strategy_name
        chunk_meta["char_len"]    = len(chunk_txt)
        chunk_meta["token_len"]   = token_len_functiom(chunk_txt)

        # after adding metadata wrap into the format
        final_chunks.append({
            "page_content": chunk_txt,
            "metadata":     chunk_meta,
        })

    return final_chunks


# (old main() removed - was superseded by the version below that includes
# hierarchical chunking and the new true paragraph-based OPENAI_PARA strategy)


# ###########################################
# --- 8. HIERARCHICAL PARENT-CHILD CHUNKING ---
# ##########################################
# After running  flat chunk experiments above, i noticed that even with paragraph-based chunking (OPENAI_PARA),
# LLM sometimes gets chopped up pieces of legal arguments.
# The court writes its reasoning as numbered points (1. 2. 3. ...) where each point is one complete legal argument.
# If a point is 800 tokens and chunk_size is 500, it gets split into two half-arguments.
#
# Idea: split reasoning into PARENTS (= numbered legal points, natural boundaries) and then split each parent into small CHILDREN (~220 tokens) for retrieval.
# Retrieve via children (precise), but send the LLM complete parents (full context) - wanted to do this also with 200 tokens, but now implementing formally
#
# This way retrieval should be mrore precise (small children match queries better) but the LLM
# sees full legal arguments (parents), not chopped-up pieces.
#
# i keep this in  same file because it uses the same tokenizer setup and separator logic.
# It just adds a second output directory with the parent/child split.


############# CONFIG ###################
HIER_OUTPUT_DIR = "data/03_chunked_docs_hier"
os.makedirs(HIER_OUTPUT_DIR, exist_ok=True)
#########################################################

# reuse POINT_SPLIT_RE defined above (same regex for both PARA and hier chunking)

# child splitter - same separators as SEPARATORS_STANDARD but smaller chunks
# chunk_size=220 is smaller than OPENAI_PARA because children need to be precise
# chunk_overlap=60 is higher ratio (~27%) because i want children to havem context from neighboring text within the same parent
child_splitter = RecursiveCharacterTextSplitter(
    separators=SEPARATORS_STANDARD,
    chunk_size=220,
    chunk_overlap=60,
    length_function=openai_token_len,
    is_separator_regex=True,
    keep_separator="end",
)


def split_reasoning_to_parents(reasoning_text):        # very similar to splitting into paragraphs
    """Split reasoning into parent chunks based on numbered points.
    If no numbered points found, fall back to paragraph breaks."""
    # quite lot of duplicity with function split_reasoning_to_paragraphs(), but for readability i kepti t separate
    matches = list(POINT_SPLIT_RE.finditer(reasoning_text))
    parents = []

    # fallback: no numbered points, split by double newlines # exactly the same logic as in split_reasoning_to_paragraphs()
    if not matches:
        for block in re.split(r"\n\n+", reasoning_text):
            clean_block = block.strip()
            if not clean_block:
                continue
            # skip noise parents (empty or single-character from broken OCR)
            words = re.findall(r"\S+", clean_block)
            if len(words) <= 1 and len(clean_block) <= 2:
                continue
            parents.append({"text": clean_block, "point_number": None})
        return parents

    # numbered points found - split at each "number. " boundary -  same logic as in  split_reasoning_to_paragraphs()
    first_start = matches[0].start()
    if first_start > 0:
        intro_text = reasoning_text[:first_start].strip()
        if intro_text:
            parents.append({"text": intro_text, "point_number": None})

    for i, match in enumerate(matches):
        start = match.start()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(reasoning_text)
        text = reasoning_text[start:end].strip()
        if not text:
            continue

        point_number = None
        point_match = re.match(r"\s*(\d+)\.", text)
        if point_match:
            point_number = int(point_match.group(1))            # extracting point number - which point number is chunk

        parents.append({"text": text, "point_number": point_number})    # point number better for debugging

    return parents


def split_parent_to_children(parent_text):
    """Split  parent into children using the child_splitter.
    Small parents (under 280 tokens) stay whole - i dont want to split small chunks."""
    if openai_token_len(parent_text) <= 280:       # if parent is small - do not split
        return [parent_text.strip()]

    raw_children = child_splitter.split_text(parent_text)   # splitting with splitter logic above
    final = []
    for c in raw_children:
        s = c.strip()
        if s:
            final.append(s)
    return final or [parent_text.strip()]


# main function for hierarchical pipeline
# segmentation - reasoning - parents- children
# create parent_id - child_id
# structure child - parent
# add roles, anchors, position, lengths
# output in nice json
def process_hierarchical(doc_data, filename, clean_name):
    """Process one document: split reasoning into parents, then parents into children.
    Returns (parents_list, children_list) ready for JSON output."""

    reasoning_text = doc_data.get("segments", {}).get("reasoning", "").strip()
    if not reasoning_text:
        return [], []

    # same odovodnenie prefix strip as above in the flat chunking
    reasoning_text = re.sub(r"^[oó]d[oôó]vodnenie\s*:?\s*", "", reasoning_text, flags=re.IGNORECASE).strip()

    #. split reasoning into parents
    parent_units = split_reasoning_to_parents(reasoning_text)
    case_id = str(doc_data.get("metadata", {}).get("case_id", "unknown"))   # case id from metadata - will be in parents and child

    parents = []
    children = []

    # main loop through parents
    for parent_index, parent_data in enumerate(parent_units):
        text = parent_data["text"]
        position_ratio = round((parent_index + 0.5) / max(len(parent_units), 1), 4) # 0.1 is in the start, 0.5 is in middle, 0.9 is finished
        parent_id = f"{clean_name}_parent_{parent_index:03d}"   # unique id for parent

        parent_meta = {
            "source_file": filename,
            "case_id": case_id,
            "parent_id": parent_id,
            "parent_index": parent_index,
            "section": "reasoning",
            "point_number": parent_data["point_number"] if parent_data["point_number"] is not None else -1,
            "position_ratio": position_ratio,
            "char_len": len(text),
            "token_len": openai_token_len(text),
        }

        parents.append({"page_content": text, "metadata": parent_meta})

        # split this parent into children
        child_texts = split_parent_to_children(text)
        for child_index, child_text in enumerate(child_texts):
            child_meta = {
                "source_file": filename,
                "case_id": case_id,
                "parent_id": parent_id,
                "child_id": f"{parent_id}_child_{child_index:02d}",
                "child_index": child_index,
                "child_count_in_parent": len(child_texts),
                "parent_index": parent_index,
                "section": "reasoning",
                "point_number": parent_meta["point_number"],
                "position_ratio": position_ratio,
                "char_len": len(child_text),
                "token_len": openai_token_len(child_text),
            }

            children.append({"page_content": child_text, "metadata": child_meta})

    return parents, children


def main():
    import argparse # to be able to run it from terminal
    parser = argparse.ArgumentParser(description="Chunk court decision documents for retrieval")
    parser.add_argument("--golden-only", action="store_true",
                        help="only process documents from the golden evaluation dataset (20 docs)") # only docs from golden dataset
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

    if not json_files:
        print(f"Error: No files found in {INPUT_DIR}")
        return

    # --golden-only: filter to only the 20 golden dataset documents
    # i added this so i can quickly re-chunk after changing the PARA strateg  without waiting for all 176 documents to be processed
    if args.golden_only:
        import pandas as pd
        golden_csv = "data/05_retrieval_evaluation/golden_dataset_template.csv"
        golden = pd.read_csv(golden_csv, sep=";")
        golden_names = set(golden["document_name"].tolist())
        json_files = [f for f in json_files if os.path.basename(f) in golden_names]
        print(f"--golden-only: filtered to {len(json_files)} golden dataset documents")

    print(f"Found {len(json_files)} docs. Starting chunking (all strategies)..\n")

    # just initialization of statistics - how much chunks processed
    counts = {
        "ME5_200":     0,
        "ME5_380":     0,
        "OPENAI_200":  0,
        "OPENAI_380":  0,
        "OPENAI_500":  0,
        "OPENAI_PARA": 0,
    }
    hier_parent_count = 0
    hier_child_count = 0
    skipped_no_reasoning = 0
    failed_exception = 0

    # fixed-size strategies use RecursiveCharacterTextSplitter via process_chunks()
    # OPENAI_PARA uses split_reasoning_to_paragraphs() for true paragraph splitting
    fixed_strategies = [
        (text_splitter_me5_200,    "me5_200",    "ME5_200",    me5_token_len),
        (text_splitter_me5_380,    "me5_380",    "ME5_380",    me5_token_len),
        (text_splitter_openai_200, "openai_200", "OPENAI_200", openai_token_len),
        (text_splitter_openai_380, "openai_380", "OPENAI_380", openai_token_len),
        (text_splitter_openai_500, "openai_500", "OPENAI_500", openai_token_len),
    ]

    ### main loop through documents ####
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

            # FIX: strip "odôvodnenie :" prefix before chunking
            # segmentation keeps it as part of reasoning text, but its  section label, not content
            # without this, OPENAI_PARA produces  micro-chunk "odôvodnenie :" (6 tokens)
            # because the numbered-item separator splits before "1." leaving the label alone
            reasoning_text = re.sub(r"^[oó]d[oôó]vodnenie\s*:?\s*", "", reasoning_text, flags=re.IGNORECASE).strip()

            # METADATRA
            base_metadata = doc_data.get("metadata", {})
            filename  = doc_data.get("filename", os.path.basename(file_path))
            clean_name = filename.replace(".pdf", "").replace(".json", "")

            # ### 1. fixed-size strategies (ME5_200, ME5_380, OPENAI_200, OPENAI_500) ###
            for splitter, strategy_name, suffix, token_fn in fixed_strategies:
                # call process reasoninig - return chunks
                chunks = process_chunks(
                    splitter, reasoning_text, base_metadata,
                    strategy_name, filename, token_fn
                )
                counts[suffix] += len(chunks)   # just number of chunks

                # saving json subor liket his - docname_chunks_ME%_200.json
                out_path = os.path.join(OUTPUT_DIR, f"{clean_name}_chunks_{suffix}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=4)


            # ### 2. OPENAI_PARAgraph: true paragraph-based chunking - my own paragraph logic ###
            # each numbered point = one chunk (no merging of small points)
            raw_paragraphs = split_reasoning_to_paragraphs(reasoning_text)  # create chunks
            para_chunks = []
            # creating metadata
            for chunk_txt in raw_paragraphs:
                chunk_txt = chunk_txt.strip()
                if not chunk_txt:
                    continue
                chunk_meta = base_metadata.copy()
                chunk_meta["chunk_index"] = len(para_chunks)
                chunk_meta["source_file"] = filename
                chunk_meta["strategy"]    = "openai_para"
                chunk_meta["char_len"]    = len(chunk_txt)
                chunk_meta["token_len"]   = openai_token_len(chunk_txt)
                para_chunks.append({
                    "page_content": chunk_txt,
                    "metadata":     chunk_meta,
                })

            counts["OPENAI_PARA"] += len(para_chunks) # just for statistics number of chunks

            # save json in way - ..._chunks_OPENAI_PARA.json
            out_path = os.path.join(OUTPUT_DIR, f"{clean_name}_chunks_OPENAI_PARA.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(para_chunks, f, ensure_ascii=False, indent=4)

            # ### 3. HIERARCHICAL PARENT-CHILD CHUNKING ###
            # this runs alongside  flat strategies using the same doc_data
            parents, children = process_hierarchical(doc_data, filename, clean_name)

            # saving parents and children
            if parents:
                parents_path = os.path.join(HIER_OUTPUT_DIR, f"{clean_name}_parents.json")
                children_path = os.path.join(HIER_OUTPUT_DIR, f"{clean_name}_children.json")

                with open(parents_path, "w", encoding="utf-8") as f:
                    json.dump(parents, f, ensure_ascii=False, indent=4)
                with open(children_path, "w", encoding="utf-8") as f:
                    json.dump(children, f, ensure_ascii=False, indent=4)
                hier_parent_count += len(parents)
                hier_child_count += len(children)

        except Exception as e:
            print(f"\nError in {file_path}: {e}")
            failed_exception += 1

    # ### 7. FINAL REPORT ###
    processed = len(json_files) - skipped_no_reasoning - failed_exception
    print("\n=== CHUNKING PROCESS COMPLETED ===")
    print(f"Processed:               {processed} / {len(json_files)}")
    print(f"Skipped (no reasoning):  {skipped_no_reasoning}")
    print(f"Failed (exception):      {failed_exception}")
    print()
    print("Chunks per strategy:")
    for suffix, count in counts.items():
        avg = round(count / processed, 1) if processed > 0 else 0
        print(f"  {suffix:<15} {count:>6} total  (avg {avg} per doc)")  # number of chunks, average per document
    print()
    print("Hierarchical parent-child:")
    avg_p = round(hier_parent_count / processed, 1) if processed > 0 else 0
    avg_c = round(hier_child_count / processed, 1) if processed > 0 else 0
    print(f"  Parents:  {hier_parent_count:>6} total  (avg {avg_p} per doc)")  # number of parents, averag eper document
    print(f"  Children: {hier_child_count:>6} total  (avg {avg_c} per doc)")


if __name__ == "__main__":
    main()