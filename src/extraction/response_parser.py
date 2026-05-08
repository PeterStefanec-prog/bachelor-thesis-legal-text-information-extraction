"""
Response parser for LLM extraction outputs

prompt_builder.py - builds prompt
llm_client - sends prompt to model
response_parser.py - process and control response of model

 LLM returns  JSON string. This module:
1. Parses it into Python dict (control/repair - handles common JSON errors)
2. Merges Call 1 + Call 2 results into  final schema
3. Adds quality_control flags for obvious errors

I kept this simple on purpose -  LLM should do  heavy lifting.
This is just  safety net for format issues.
"""

import json
import re
import copy


def parse_llm_response(response_text):
    """Parse LLM response text into Python dict.

    LLMs sometimes add markdown code fences or explanations around JSON.
    This function tries to extract  JSON even when  format is messy.

    Returns:
        tuple: (parsed_dict, error_message)
        If parsing fails, parsed_dict is None and error_message explains why
        So i can do this in run_extraction.py (for Call1, Call2, Fulldoc extraction):
            result, error = parse_llm_response(...)
            if error:
                ...
    """
    text = response_text.strip()    # for example "\n\n {\"a\": 1} \n" -> "{\"a\": 1}"

    # try direct parse from json to python dict first (works for openai json mode)
    try:
        return json.loads(text), None   # function will return (parsed_dict, None)  - None means no error
    except json.JSONDecodeError:    # json.loads throw this error  whnen json is not valid
        pass

    # some models wrap JSON in markdown code fences like ```json {...} ```
    # i need to strip those before parsing
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)    # this ?: means that it is not capturing group (just group)
    # withou DOTALL it. would not match new lines (they are in json)
    if json_match:
        try:
            return json.loads(json_match.group(1)), None        # group(1) is first catching group (.*?)
        except json.JSONDecodeError:
            pass

    # last try: find the first { and last } and try to parse that
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1]), None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {str(e)}"  # if this fail, it return None and the error

    return None, "No JSON object found in response"


def merge_call1_call2(call1_result, call2_result, chunk_lookup=None):
    """Merge call 1 (facts) and call 2 (moderation) into  final schema

    Call 1 (already parsed) gives us: case_context ... , contractual_penalties (without moderation)
    Call 2 ( parsed) gives us: penalties_moderation array with moderation_analysis for each penalty

    We match them by penalty_id and merge.

    chunk_lookup: optional (was not extracting when debugging) dict (chunk_id -> text) for evidence grounding check.
    If provided, validation will verify that all evidence quotes are real substrings.

    Returns:
        dict: complete extraction result matching  v5.0 schema
    """
    # FIX: i  used call1_result.copy() which is jsut SHALLOW copy.
    # That means when i add moderation_analysis to penalties list below, it also mutates  original call1_result dict
    # (because  inner lists and dicts are shared references, not copies).
    # This wasnt causing bugs because i dont use call1_result after merge, but just to be sure -
    # if i ever wanted to save call1_result separately it would be corrupted.
    # deepcopy makes a completely independent copy of everything - not just refernce to old call1_result (as .copy)
    final = copy.deepcopy(call1_result)

    # build lookup from call2 by penalty_id - helper dictionary from call 2
    moderation_lookup = {}
    if call2_result and "penalties_moderation" in call2_result:     # if there is expected key - penalties_moderation
        for mod in call2_result["penalties_moderation"]:
            pid = mod.get("penalty_id", "")      # i use get so it will not crash
            moderation_lookup[pid] = mod.get("moderation_analysis", {})     # saving moderation (from call 2) for each penaly


    # merge moderation into each penalty
    for penalty in final.get("contractual_penalties", []):
        pid = penalty.get("penalty_id", "")
        if pid in moderation_lookup:
            penalty["moderation_analysis"] = moderation_lookup[pid]
        else:
            # call2 didnt return moderation for this penalty - fill with empty
            penalty["moderation_analysis"] = _empty_moderation()

    # add quality_control if not present
    if "quality_control" not in final:
        final["quality_control"] = {"flags": [], "missing_fields": []}

    # run validation (including evidence grounding if chunk_lookup provided)
    flags, missing = validate_extraction(final, chunk_lookup=chunk_lookup)  # start main validation func
    final["quality_control"]["flags"].extend(flags)
    final["quality_control"]["missing_fields"].extend(missing)

    return final


def _empty_moderation():
    """Return empty moderation_analysis structure when call2 fails."""
    return {
        "decision": {
            "value": "unclear",
            "dismissal_reason": None,
            "evidence": {"quote": None, "chunk_id": None}
        },
        "legal_reasoning_summary": "",
        "key_quotes": [],
        "factors": [
            {"label": label, "sentiment": "not_mentioned", "evidence": {"quote": None, "chunk_id": None}}
            for label in [
                "dobre_mravy", "zabezpecovacia_funkcia", "vyska_skody",
                "pomer_k_istine", "spravanie_dlznika", "kumulacia_s_urokom",
                "spravanie_veritela",
            ]
        ]
    }


def validate_extraction(result, chunk_lookup=None):
    """Run basic validation checks on the extraction result
    I use it in 2 sitations:
        - RAG - through merge_call1_call2
        - fulloc - right in run_extraction.py

    These are deterministic business logic checks - not LLM-based.
    If chunk_lookup is provided (dict: chunk_id -> text), also checks that evidence quotes actually appear in their claimed chunks

    Returns (flags, missing_fields) lists.
    """
    flags = []          # just warnings
    missing = []        # missing fileds

    penalties = result.get("contractual_penalties", [])

    if not penalties:
        flags.append("NO_PENALTIES_EXTRACTED")
        return flags, missing

    ################# loop through all penalties in extraction ##############
    for penalty in penalties:
        pid = penalty.get("penalty_id", "unknown")      # get id of penalty

        ####  check amounts make sense ####
        amounts = penalty.get("amounts", {})    # block with amounts
        claimed = amounts.get("original_claimed", {}).get("value")  # povodnu
        awarded = amounts.get("final_awarded", {}).get("value")

        if claimed is not None and awarded is not None:
            if awarded > claimed:
                # court cant award more than what was claimed (viazanost petitom)
                flags.append(f"MATH_ERROR_AWARDED_EXCEEDS_CLAIMED_{pid}")


        #### check moderation consistency ####
        mod = penalty.get("moderation_analysis", {})
        decision = mod.get("decision", {}).get("value", "")

        if decision == "moderated_301":
            if claimed is not None and awarded is not None and claimed == awarded:  # how could be moderated when awarded penaly is  same
                flags.append(f"LOGIC_ERROR_MODERATION_BUT_SAME_AMOUNT_{pid}")

        if decision == "dismissed" and mod.get("decision", {}).get("dismissal_reason") is None: # if decision is dismissed but dismissal_reason is missing
            flags.append(f"MISSING_DISMISSAL_REASON_{pid}")

        #   ####FIX: check for anonymized amounts. Some court decisions have redacted numbers like "X XXX,XX eur" or "XXX,XX Sk"   ####
        # where  actual digits are replaced with X.
        #  LLM sees these and INVENTS numbers (e.g. returns  600000 when the text says "XXX XXX").
        #  I found this when auditing 40 random documents - 4Cob/122/2011 had 81 XXX markers and LLM hallucinated both claimed and awarded amounts that dont exist anywhere.
        #
        # The check: if ANY evidence quote for amounts contains "XX" (2+ X's in a row),
        # amount is probably from anonymized text and should be flagged.
        for amt_field in ["secured_principal", "original_claimed", "final_awarded"]:
            amt_obj = amounts.get(amt_field, {})
            if amt_obj.get("value") is not None:
                ev_quote = amt_obj.get("evidence", {}).get("quote", "") or ""   # get citation proof
                if re.search(r'X{2,}', ev_quote):       # it finds 2 or more 'X' charrs
                    flags.append(f"ANONYMIZED_AMOUNT_{amt_field}_{pid}")    # llm gave actual number but it is hallucination bcs there is not one

        #   #### FIX: currency cross-check against chunk text.  #### not IMPORTANT
        # i found that GPT-4o sometimes sets currency="EUR" even when  text says "Sk" or "SKK" (slovak koruna, used before 2009).
        # i think the model just defaults to EUR for any european country. so i check if the chunks
        # mention "Sk" amounts and the LLM said EUR - if yes, i flag it.
        if chunk_lookup and amounts.get("currency") == "EUR":
            all_chunk_text = " ".join(chunk_lookup.values())    # merge all chunks into one text
            # look for Sk as currency - pattern: number followed by optional dash/space then "Sk" followed by word boundary
            # This catches "146.162,99 Sk" and "200.000,- Sk" but not "Slovensko" or "diskusia"
            sk_matches = re.findall(r'\d[\d\s.,-]*\s*Sk\b', all_chunk_text)
            skk_matches = re.findall(r'(?i)\bSKK\b', all_chunk_text)
            if len(sk_matches) >= 2 or skk_matches:
                flags.append(f"CURRENCY_MISMATCH_POSSIBLE_SKK_{pid}")   # probably bad response of EUR

        #### check for missing key fields  ###
        if not penalty.get("breach_type", {}).get("value"):
            missing.append(f"{pid}.breach_type")
        if not penalty.get("rate_definition", {}).get("type"):
            missing.append(f"{pid}.rate_definition")
        if amounts.get("original_claimed", {}).get("value") is None:
            missing.append(f"{pid}.amounts.original_claimed")

    # ==========================================
    # FIX 2: EVIDENCE GROUNDING CHECK
    # ==========================================
    # I added this after testing on  first document - i was surprised that 5 out of 13 evidence quotes were NOT exact substrings of their chunk!
    #  LLM was doing  things like changing "v časti" to "V časti" (capitalizing mid-sentence quotes) and cutting off quotes with period
    #   where  original text continued with parenthesis.
    #
    # This is big deal because  whole point of evidence is that lawyer can ctrl+F the quote in the original text and verify it.
    # If the quote is even slightly different, ctrl+F wont find it and the lawyer would lose trust in the system.
    # So i need to catch this.
    #
    # The check has 6 levels:
    #   EXACT match   -> no flag (perfect, ctrl+F would find it)
    #   CASE match    -> flag EVIDENCE_CASE_MISMATCH (LLM changed letter)
    #   NO match      -> flag EVIDENCE_NOT_FOUND (hallucinated or heavily modified

    if chunk_lookup:    # mapping chunk_id on text
        evidence_flags = validate_evidence_grounding(result, chunk_lookup)
        flags.extend(evidence_flags)

    return flags, missing


def _collect_evidence_fields(result):
    """Walk through  extraction result and collect all evidence objects - which has quote and chunk_id

    Returns list of (field_path, quote, chunk_id) tuples - ("pokuta_1.amounts.final_awarded", "zmluvná pokuta 400 eur", "chunk_11")
    I need this to check every single evidence quote in one loop.
    """
    evidence_list = []

    # loop through all penalties
    for penalty in result.get("contractual_penalties", []):
        pid = penalty.get("penalty_id", "?")

        # call1 fields with evidence
        for field_name in ["breach_type", "rate_definition"]:
            ev = penalty.get(field_name, {}).get("evidence", {})
            if ev.get("quote") and ev.get("chunk_id"):          # only if it quote and chunk_id is provided
                evidence_list.append((f"{pid}.{field_name}", ev["quote"], ev["chunk_id"]))

        # amount fields with evidence   - amounts
        for amt_name in ["secured_principal", "original_claimed", "final_awarded"]:
            ev = penalty.get("amounts", {}).get(amt_name, {}).get("evidence", {})
            if ev.get("quote") and ev.get("chunk_id"):
                evidence_list.append((f"{pid}.amounts.{amt_name}", ev["quote"], ev["chunk_id"]))

        # associated interest
        ev = penalty.get("associated_interest", {}).get("evidence", {})
        if ev.get("quote") and ev.get("chunk_id"):
            evidence_list.append((f"{pid}.associated_interest", ev["quote"], ev["chunk_id"]))

        # moderation fields
        # FIX (2026-04-16): mod was indented 4 extra spaces, making it part of the `if`
        # block above. When associated_interest evidence was missing (common for docs
        # without interest info, e.g. 5Cob/11/2022), mod would never be assigned and
        # next line would throw UnboundLocalError.
        mod = penalty.get("moderation_analysis", {})

        # decision evidence
        ev = mod.get("decision", {}).get("evidence", {})
        if ev.get("quote") and ev.get("chunk_id"):
            evidence_list.append((f"{pid}.decision", ev["quote"], ev["chunk_id"]))

        # key_quotes
        for i, kq in enumerate(mod.get("key_quotes", [])):
            if kq.get("quote") and kq.get("chunk_id"):
                evidence_list.append((f"{pid}.key_quote_{i}", kq["quote"], kq["chunk_id"]))

        # factors
        for factor in mod.get("factors", []):
            ev = factor.get("evidence", {})
            if ev.get("quote") and ev.get("chunk_id"):
                label = factor.get("label", "?")
                evidence_list.append((f"{pid}.factor.{label}", ev["quote"], ev["chunk_id"]))

    return evidence_list


def _normalize_whitespace(text):
    """Remove spaces before punctuation marks.

    FIX: PDF extraction sometimes leaves spaces before periods and commaslike "potvrdzuje ." instead of "potvrdzuje." -
    LLM "fixes" this in its quotes which breaks exact match. I normalize both sides before
    comparing so this preprocessing artifact doesnt cause false NOT_FOUND flags.
    """
    return re.sub(r'\s+([.,;:!?\)\]])', r'\1', text)


def _lemmatize_text(text):
    """Lemmatize text for fuzzy comparison of Slovak inflected forms.

    Slovak has 6 grammatical cases x 2 numbers = 12+ forms per noun.
     LLM often changes the case when quoting: "zmluvnej pokuty" (genitiv) becomes "zmluvnú pokutu" (accusativ).
     Both mean the same thing but exact match fails. Lemmatization normalizes both to "zmluvný pokuta".

    I already use simplemma in the BM25 tokenizer (shared_utils.py) (can be spacy also )
    """
    try:
        import simplemma
        words = re.findall(r'[a-záäčďéíľĺňóôŕšťúýžA-ZÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ]+|\S+', text.lower())    # tokenizing text (words)
        return " ".join(simplemma.lemmatize(w, lang="sk") for w in words) # each tokken is lemmatized
    except Exception:   # any error
        return text.lower()


def _find_best_match_in_chunk(quote, chunk_text, min_overlap=0.65):
    """Try to find the best matching substring in chunk_text for  quote.

    This handles the case where  LLM paraphrases slightly - drops word like "v žalobe" or adds "že" at the beginning.
    The quote is ALMOST in chunk but not an exact substring.

    How it works:
    1. Split both quote and chunk into word lists
    2. Slide  window of quote-length across the chunk words
    3. For each window position, count how many quote words appear in it
    4. If the best overlap is >= min_overlap (65%), we found a match

    Returns (best_overlap_ratio, matched_text) or (0.0, None) if no match.

    I set min_overlap=0.65 (not higher) because the LLM sometimes drops 2-3
    words from a 15-word quote, which is ~80% overlap.
    """
    # split into words (keep punctuation as separate tokens for accuracy)
    quote_words = re.findall(r'\S+', quote.lower()) # divide citation on tokens (based on whitespace)
    chunk_words = re.findall(r'\S+', chunk_text.lower())    # divide chunk on tokens

    if not quote_words or not chunk_words:
        return 0.0, None

    quote_len = len(quote_words)    # length of citation in tokens
    # window slightly bigger (3 tokens more) than quote to catch added words but not bigger than chunk
    window_size = min(quote_len + 3, len(chunk_words))

    best_overlap = 0.0
    best_start = 0

    quote_set = set(quote_words)

    for start in range(len(chunk_words) - window_size + 1):
        window = chunk_words[start:start +  window_size] # choose the window
        window_set = set(window)    # move to set (no duplicity, no order)

        # count how many quote words appear in this window
        overlap = len(quote_set & window_set)   # how much words from citation is in chunk
        ratio = overlap / len(quote_set)

        if ratio > best_overlap:    # save the best ratio
            best_overlap = ratio
            best_start = start

    if best_overlap >= min_overlap:
        # reconstruct the matched text from the original (non-lowered) chunk
        orig_words = re.findall(r'\S+', chunk_text)
        matched = " ".join(orig_words[best_start:best_start + window_size])
        return best_overlap, matched

    return 0.0, None


def validate_evidence_grounding(result, chunk_lookup):
    """Check every evidence quote against  claimed chunk text.

    This is my hallucination detector.
    After the first test i found that 5 out of 13 quotes were wrong.
    After many iterations the check now has 6 levels, from strictest to most lenient:

    1. EXACT match           -> perfect, ctrl+F would find it
    2. WHITESPACE normalized -> preprocessing spaces before punctuation
    3. CASE-INSENSITIVE      -> LLM capitalized first letter
    4. LEMMATIZED            -> LLM changed grammatical case (Slovak inflection)
    5. FUZZY (word overlap)  -> LLM dropped/added word or two
    6. NOTHING matched       -> hallucinated or completely rewritten

    For levels 2-5, i flag the issue but its still considered "usable" evidence.
    Only level 6 is  real failure.

    If level 5 (fuzzy) finds match, i also REPAIR the quote in the result by replacing it with the actual text from the chunk.
    This way the output JSON has valid ctrl+F-able quote even when  LLM paraphrased slightly.

    chunk_lookup: dict mapping chunk_id -> full chunk text.
    Returns list of flag strings.
    """
    flags = []
    evidence_list = _collect_evidence_fields(result)    # get all evidence fields  from response

    ### go through every evidence
    for field_path, quote, chunk_id in evidence_list:
        chunk_text = chunk_lookup.get(chunk_id, "")

        if not chunk_text:
            flags.append(f"EVIDENCE_CHUNK_NOT_FOUND:{field_path}:{chunk_id}")
            continue

        # level 1: exact substring match - best case
        if quote in chunk_text:
            continue

        # level 2: normalize whitespace before punctuation on both sides.
        # catches preprocessing artifacts like "potvrdzuje ." vs "potvrdzuje."
        norm_quote = _normalize_whitespace(quote)
        norm_chunk = _normalize_whitespace(chunk_text)
        if norm_quote in norm_chunk:
            flags.append(f"EVIDENCE_WHITESPACE_NORMALIZED:{field_path}")
            continue

        # level 3: case-insensitive match
        if quote.lower() in chunk_text.lower():
            flags.append(f"EVIDENCE_CASE_MISMATCH:{field_path}")
            continue

        # level 4: lemmatized match. Catches Slovak inflection changes like "zmluvnej pokuty" (genitive) vs "zmluvnú pokutu" (accusative)
        # Both lemmatize to "zmluvný pokuta" so they match.
        lemma_quote = _lemmatize_text(quote)
        lemma_chunk = _lemmatize_text(chunk_text)
        if lemma_quote in lemma_chunk:
            flags.append(f"EVIDENCE_INFLECTION_MISMATCH:{field_path}")
            continue

        # level 5: fuzzy word overlap. Catches LLM dropping/adding words like "Žalobca v žalobe jednoznačne" -> "žalobca jednoznačne" (dropped "v žalobe").
        # If i find >= 65% word overlap, i consider it a match and REPAIR the quote.
        overlap, matched_text = _find_best_match_in_chunk(quote, chunk_text)
        if matched_text:
            flags.append(f"EVIDENCE_QUOTE_REPAIRED:{field_path}")
            # repair the quote in the result so the output has a valid substring
            _repair_quote_in_result(result, field_path, matched_text, chunk_id)
            continue

        # level 6: nothing matched - either hallucinated or too heavily modified
        # i still check first 40 chars as a last heuristic for truncation
        short = quote[:40].lower()
        if short in chunk_text.lower():
            flags.append(f"EVIDENCE_TRUNCATED:{field_path}")
        else:
            flags.append(f"EVIDENCE_NOT_FOUND:{field_path}:{chunk_id}")

    return flags


def _repair_quote_in_result(result, field_path, new_quote, chunk_id):
    """Replace paraphrased quote with the actual text found in the chunk.

    field_path looks like "pokuta_1.breach_type" or "pokuta_1.factor.dobre_mravy"
    i need to navigate the result dict to find the right evidence object and update its quote field.
    Result is the whole response

    This is best-effort repair - if the path doesnt match for some reason, i just skip it .
     But flag is already set so we know it happened.
    """
    parts = field_path.split(".") # just divide path based on .
    if len(parts) < 2: # must have at least 2 parts
        return

    pid = parts[0]  # id of penalty - e.g. "pokuta_1"

    # going through all penalties
    for penalty in result.get("contractual_penalties", []):
        if penalty.get("penalty_id") != pid:
            continue

        # navigate to the evidence object based on field_path
        try:
            if parts[1] in ("breach_type", "rate_definition"):
                penalty[parts[1]]["evidence"]["quote"] = new_quote

            elif parts[1] == "amounts" and len(parts) >= 3:
                penalty["amounts"][parts[2]]["evidence"]["quote"] = new_quote

            elif parts[1] == "associated_interest":
                penalty["associated_interest"]["evidence"]["quote"] = new_quote

            elif parts[1] == "decision":
                penalty["moderation_analysis"]["decision"]["evidence"]["quote"] = new_quote

            elif parts[1].startswith("key_quote"):
                idx = int(parts[1].split("_")[-1])
                if idx < len(penalty["moderation_analysis"]["key_quotes"]):
                    penalty["moderation_analysis"]["key_quotes"][idx]["quote"] = new_quote

            elif parts[1] == "factor" and len(parts) >= 3:
                label = parts[2]
                for factor in penalty["moderation_analysis"]["factors"]:
                    if factor["label"] == label:
                        factor["evidence"]["quote"] = new_quote
                        break
        except (KeyError, IndexError, TypeError):
            pass  # couldnt repair - thats ok, the flag is already set
        break
