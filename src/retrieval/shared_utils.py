import os
import re
import json
import glob


# #############################################
# Shared utilities for retrieval evaluation
# ############################################
# These were originally inline in evaluate_retrieval.py but i extracted them
# here when i added hierarchical mode - both flat and hierarchical evaluation
# need the same text cleaning, tokenization and normalization functions

# also evalutate_retrieval is too long so tried to shorten it with extracting duplicated functions


# ############################################
# TEXT CLEANING
# ############################################
# If my PDF has "500 \n eur" but golden dataset has "500 eur", normal Python
# substring match says "MISS!". This removes newlines and extra spaces so
# matching works regardless of whitespace differences.

def clean_text(text):
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def safe_avg(lst):
    """Average of a list, returns empty string if list is empty (for CSV output)."""
    return round(sum(lst) / len(lst), 4) if lst else ""


# #######################################################
# TOKEN LENGTH ESTIMATION
# #######################################################
# i use tiktoken (the official OpenAI tokenizer) to count tokens accurately.
# if tiktoken is not installed, fall back to rough whitespace count.

def get_openai_length_function():
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("text-embedding-3-small")

        def _tok_len(text):
            return len(enc.encode(text))

        return _tok_len
    except Exception:
        def _tok_len(text):
            return len(re.findall(r"\S+", text or ""))

        return _tok_len


openai_token_len = get_openai_length_function()


# ############################################
# MIN-MAX NORMALIZATION for score dictionaries
# ############################################

def min_max_normalize(score_dict):
    if not score_dict:
        return {}
    vals = list(score_dict.values())
    lo = min(vals)
    hi = max(vals)
    if lo == hi:
        return {k: 0.5 for k in score_dict}
    normalized = {}
    for k, v in score_dict.items():
        normalized[k] = (v - lo) / (hi - lo)

    return normalized
# for example from {"a": 2, "b": 6, "c": 10} to  {"a": 0.0, "b": 0.5, "c": 1.0}

# ############################################
# HEADER + VERDICT LOADER
# ############################################
# Both pipelines always send header and verdict to the LLM alongside the
# retrieved chunks/parents. This function loads them from the processed JSON
# so evaluation can check golden quotes against header/verdict text too.

_PROCESSED_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "data", "02_processed_json",
)


def load_header_verdict(pdf_filename):
    """Load header and verdict text for a document.
    Returns (header_text, verdict_text). If not found, returns ("", "")."""
    clean_name = pdf_filename.replace(".pdf", "").replace(".json", "")
    pattern = os.path.join(_PROCESSED_DIR, f"{clean_name}*.json")
    matches = glob.glob(pattern)
    if not matches:
        return "", ""
    with open(matches[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    header = data.get("header", "")     # safer than data["header"] - no crash
    verdict = data.get("segments", {}).get("verdict", "")   # 2 level - verdict is under segments in json structure
    return header, verdict


# ############################################
# SLOVAK LEGAL TEXT TOKENIZER (used by BM25 in both pipelines)
# ############################################
# Design decisions:
# 1. PRESERVE NUMBERS & PERCENTAGES: "0,05%" stays as one token
#    Legal texts have specific amounts (15.234,60 EUR) and rates (0,05% rocne)
# 2. PRESERVE LEGAL REFERENCES: "§ 301" -> "§301" as one token
# 3. LEMMATIZATION via simplemma: "pokuty" -> "pokuta"
#    Slovak has 6 cases x 2 numbers = 12+ forms per noun
# 4. STOPWORD REMOVAL: drop common words that add noise

try:
    import simplemma
except Exception:
    simplemma = None

SLOVAK_STOPWORDS = {
    "a", "aj", "ak", "ako", "ale", "alebo", "ani", "áno", "asi",
    "by", "bol", "bola", "bolo", "boli", "buď", "byť", "bez",
    "do", "dňa",
    "ho", "jeho", "jej", "ich",
    "je", "ju",
    "keď", "keďže", "kde", "ku", "ktorý", "ktorá", "ktoré", "ktorej", "ktorým", "ktorom",
    "na", "nad", "nie", "no", "než",
    "od", "ods",
    "po", "pod", "podľa", "pre", "pri", "pred", "preto",
    "sa", "si", "so", "sú",
    "ten", "to", "tá", "tú", "tej", "tom", "tomu", "tým", "tento", "tiež", "tak", "takže",
    "vo", "vz",
    "za", "zo", "že",
}   # i used set because the check whether something is in set is really fast (list is slower)


def tokenize_slovak(text):
    """Tokenize Slovak legal text with number preservation and lemmatization - good for BM25
    INPUT "Zmluvná pokuta vo výške 0,05% denne z nezaplatenej istiny podľa § 301 ObchZ"
    OUTPUT ["zmluvný", "pokuta", "výška", "0,05%", "denný", "nezaplatený", "istina", "§301", "obchz"]
    """

    text = (text or "").lower()     # "Pokuta" must be the same as "pokuta"

    # join paragraph sign with following number: "§ 301" -> "§301"
    text = re.sub(r"§\s*(\d+)", r"§\1", text)

    # Single regex extracts all token types in priority order:
    # 1. Legal refs: §301, §544
    # 2. Numbers: Slovak format (dot thousands, comma decimals, optional %)
    # 3. Long digit sequences (case IDs etc.)
    # 4. Words: letter sequences including Slovak diacritics    pokuta, zanokkinka, dohodnuta
    raw_tokens = re.findall(
        r"§\d+"
        r"|\d{1,3}(?:\.\d{3})*(?:,\d+)?%?"
        r"|\d{6,}"
        r"|[a-záäčďéíľĺňóôŕšťúýžA-ZÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ]+",
        text,
    )

    tokens = []
    for token in raw_tokens:
        if token.startswith("§") or token[0].isdigit(): # if token starts with § or digit - no edit
            tokens.append(token)
            continue

        if len(token) <= 1:
            continue

        # LEMMATIZATION is really important in slovak language
        # change pokuty, pokutu, pokutou to POKUTA  (changing
        # did not use stemming because pokuta is better than pokut
        if simplemma is not None:
            lemma = simplemma.lemmatize(token, lang="sk")
        else:
            lemma = token
            print ("No lemmatization used because do not have simplemma installed - please install and run again")

        if lemma not in SLOVAK_STOPWORDS and len(lemma) > 1:
            tokens.append(lemma)

    return tokens
