"""
Prompt builder for the extraction pipeline.

This module loads prompt templates and fills them with actual data (chunks, header, verdict, etc.)
to create the final prompts for the LLM.
system_prompt.txt + user_prompt_call1.txt + user_prompt_call2.txt + (alt. user_prompt_fulldoc.txt_

I keep the templates as .txt files so i can edit them without touching code.
The curly braces {variable} are replaced with actual content at runtime.
Double curly braces {{}} in the templates are literal braces for JSON examples.
"""

import os
import json

# path to templates directory (relative to this file)
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _load_template(filename):
    """Load prompt template from the templates directory and return as  string"""
    path = os.path.join(TEMPLATE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_chunks_for_prompt(chunks):
    """Format list of chunk dicts into readable string for the LLM.

    Each chunk gets header with its ID so the LLM can reference it in evidence fields
     Chunks are already sorted by chunk_index (document order) from the precompute step.

    Example output:
        [CHUNK 0] (KS_Trenčín_8Cob_64_2011_00_dokument.pdf_chunk_0)
        Okresný súd Bánovce nad Bebravou napadnutým rozsudkom...

        [CHUNK 2] (KS_Trenčín_8Cob_64_2011_00_dokument.pdf_chunk_2)
        Rozhodnutie odôvodnil tým...
    """
    parts = []  # text of each chunk will be saved here
    for chunk in chunks:
        chunk_header = f"[CHUNK {chunk['chunk_index']}] ({chunk['chunk_id']})"
        parts.append(f"{chunk_header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _strip_evidence(obj):
    """Recursively remove all 'evidence' and 'chunk_id' keys from a dict.

    FIX: i use this to clean Call 1 output before sending it to Call 2.
    Without this, the LLM in Call 2 copies chunk_ids from Call 1 prefilled data
    instead of citing from its own chunks. This caused wrong_chunk errors
    in multi-penalty documents.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k == "evidence" or k == "chunk_id":  # if key is not evidence nor chunk_id, leave it
                continue
            cleaned[k] = _strip_evidence(v) # each value recursively clean
        return cleaned
    elif isinstance(obj, list):
        return [_strip_evidence(item) for item in obj]  # each list of objects clean spearately
    return obj  # basic value leave


# this function is jusst for clarity (to call build_system_prompt - but actaully just sending system_prompt.txt :) )
def build_system_prompt():
    """Load  system prompt - which is same for all calls and all modes."""
    return _load_template("system_prompt.txt")


def build_call1_prompt(retrieval_data):
    """Build  user prompt for Call 1 (contract facts + penalty params).

    Arguments:
        retrieval_data: dictionary loaded from data/06_retrieval_results/{doc}.json (it contains metadata, header, verdict, full reasoning, chunks)
    Returns:
        string -  formatted user prompt
    """
    template = _load_template("user_prompt_call1.txt")  # there is something like this HEADER: {header}, ... {verdict}
    chunks_text = format_chunks_for_prompt(retrieval_data["call1_chunks"]) # nice long text of chunks

    return template.format(
        header=retrieval_data["header"],    # i substitute {header} with actual header
        verdict=retrieval_data["verdict"],  # same with verdict
        chunks=chunks_text,     # inserting real chunks
    )


def build_call2_prompt(retrieval_data, call1_result):
    """Build user prompt for Call 2 (moderation analysis).

    Gets the prefilled JSON from call1 so the LLM knows what penalty it needs to analyze.
    This way it doesnt have to re-extract basic facts.

    FIX 3: I added the verdict text here.
    Originally Call 2 only got reasoning chunks + prefilled JSON.
    But the verdict contains thecourts actual decision (e.g. "zamietol žalobu v časti zmluvnej pokuty")
    which is critical for the moderation_analysis.decision field.

    Without it LLM had to guess the outcome from the reasoning alone, which sometimes
    led to wrong decision values (e.g. "unclear" when the verdict clearly
    said "dismissed").

    FIX 4: I also added secured_principal and currency to the prefilled JSON.
    The LLM needs these for factor analysis - e.g. pomer_k_istine (penalty ratio to principal) requires knowing the principal amount.
     Without it, LLM couldnt properly assess whether the court discussed this factor.

    Args:
        retrieval_data: dictionary from precomputed retrieval JSON - data/06_retrieval_results/{doc}.json (it contains metadata, header, verdict, full reasoning, chunks)
        call1_result: dict - parsed JSON output from Call 1
    Returns:
        string - the formatted user prompt
    """
    template = _load_template("user_prompt_call2.txt")  # also contains {header}, {verdict}, ... variables like this
    chunks_text = format_chunks_for_prompt(retrieval_data["call2_chunks"])  # chunks relevatn for call 2

    # I send the full Call 1 output to Call 2 so it has all the context about contract and penalty.
    # But i STRIP the evidence objects (quote + chunk_id)mbefore sending. This did quite lot problems.

    # FIX: "cross-call contamination" — when i sent the full Call 1 JSON including evidence,  LLM in Call 2 saw chunk_ids like "docname_chunk_8" from Call 1
    # and REUSED them for Call 2 evidence, even though those chunks were not inm Call 2 context.
    # The LLM was basically copying chunk_ids from  prefilled data instead of looking at the actual Call 2 chunks.
    # This caused 5 WRONG_CHUNK flags in the 2Cob/69/2020 document (3 penalties, most complex case).
    #
    # The fix is simple: strip evidence from  prefilled JSON.
    # The LLM still sees all  extracted VALUES (breach_type, rate, amounts, interest) which is whatmit actually needs for factor analysis
    # It just doesnt see chunk_ids that could confuse it.
    prefilled_clean = _strip_evidence(call1_result) # so delete evidence, chunk_id elements in josn
    prefilled_json = json.dumps(prefilled_clean, ensure_ascii=False, indent=2)  # change python dicti on json text
    # indent 2 so the json will be well shifted from the left and false ascii for slovak letters
    return template.format(     # call 2 template is appended by
        prefilled_json=prefilled_json,      # json from call 1
        verdict=retrieval_data["verdict"],      # full verdict (can explicitly say - zaloba zamietnuta, potvrdene, zmenene, priznane len z casti
        chunks=chunks_text,     # chunks from call 2
    )


def build_fulldoc_prompt(retrieval_data):
    """Build  user prompt for full-document mode (no RAG, entire text).

    The LLM gets the complete reasoning section instead of selected chunks.
    This is used as  baseline to compare against the RAG approach.

    Args:
        retrieval_data: dictionary from precomputed retrieval JSON
    Returns:
        string - the formatted user prompt
    """
    template = _load_template("user_prompt_fulldoc.txt")    # loading template

    return template.format(     # fillind full header, verdict, and reasoning
        header=retrieval_data["header"],
        verdict=retrieval_data["verdict"],
        reasoning=retrieval_data["reasoning_full"],
    )
