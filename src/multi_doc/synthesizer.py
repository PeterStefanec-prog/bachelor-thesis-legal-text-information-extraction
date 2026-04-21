"""
Stage 5: LLM Synthesis — generate lawyer-friendly answer from analytics and precedents.

This is the last stage. The LLM gets:
1.  lawyer's original query
2. Python-computed analytics (stats, factor lifts, rate analysis)
3. Top 5 penalty records with their structured data and key quotes

The LLM's job is to EXPLAIN  data, not COMPUTE anything.
All numbers come from Python (Stage 4).
LLM just puts them into coherent Slovak text with citations.

I have different prompts depending on intent:
- safe_rate: recommend rate range, explain risks, cite precedents
- defense_args: list strongest arguments, cite cases where factors helped
- general_precedent: summarize findings, cite key precedents
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.multi_doc.analytics import format_for_llm, format_for_lawyer


# #########################################
# LLM CONFIG
# #########################################

MODEL_NAME = "gemini-2.5-flash"

# singleton client — same pattern as in query_analyzer.py
_client = None

def _get_client():
    """Get or create Gemini client. (again singleton)"""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()
    return _client


# #########################################
# SYSTEM PROMPTS (different for each intent)
# #########################################
# core rules are the same: cite cases as [case_number], answer in Slovak,
# and NEVER compute numbers — only explain what Python computed

_BASE_SYSTEM = """\
Si pravny analytik specializovany na zmluvne pokuty v slovenskom prave.
Na zaklade poskytnutych statistik a relevantnych sudnych rozhodnuti odpovedz na otazku.

PRAVIDLA:
- Cituj konkretne rozhodnutia v hranatych zatvorkach, napr. [14Cob/109/2017].
- NEPOCITAJ ziadne cisla — vsetky statistiky su uz spocitane a poskytnutne v sekcii ANALYZA.
- Len vysvetli a interpretuj poskytnutne udaje.
- Odpoved v slovencine.
- Pouzivaj formulacie ako "v analyzovanom korpuse", "na zaklade precedensov" —
  nie "s istotou" alebo "zarucene", pretoze vzorka nie je uplna.
"""

SYSTEM_PROMPTS = {
    "safe_rate": _BASE_SYSTEM + """
ULOHA: Pravnik tvori zmluvu a chce vediet aku sadzbu pokuty nastavit.
- Odporuc rozsah sadzob s nizsim rizikom moderacie na zaklade dat.
- Vysvetli ake faktory zvysuju riziko znizenia (napr. vysoka sadzba, kumulacia s urokom).
- Cituj 2-3 konkretne precedensy kde sud pokutu potvrdil a preco.
- DOLEZITE: Upozorni aj na rizika — aj standardna sadzba moze byt znizena ak sa celkova
  suma akumuluje nepriemerne, ak dlznik nebol zloumyselny, alebo ak sa kumuluje s urokom.
  Pouzi data o moderacii z ANALYZY (napr. miera moderacie percent_denne, meritorny pomer).
- Odpoved ma mat 3-5 odsekov.
""",

    "defense_args": _BASE_SYSTEM + """
ULOHA: Pravnik zastupuje klienta a hlada argumenty na moderaciu pokuty podla §301.
- Zoznam najsilnejsich argumentov pre moderaciu (zoradenych podla sily — pouzi LIFT data).
- Pre kazdy argument cituj konkretny precedens kde tento faktor zavazil.
- Vysvetli ake okolnosti zvysuju sancu na uspesnu moderaciu.
- Odpoved ma mat 3-5 odsekov.
""",

    "general_precedent": _BASE_SYSTEM + """
ULOHA: Pravnik hlada vseobecny prehlad precedensov.
- Zhrnni hlavne zistenia z analytiky (rozlozenie rozhodnuti, klucove faktory).
- Cituj 2-3 najrelevantnejsie rozhodnutia.
- Odpoved ma mat 2-4 odsekov.
""",
}


# #########################################
# LOW-N GUARD
# #########################################
# when cohort has less than 5 cases, LLM should NOT write confident-sounding answer
# like it does for n=50. With n=1 or n=3 any "trend" or "recommendation" is
# basically making things up - i saw it in demo: Q4 had 1 case and LLM still
# wrote 4 paragraphs of analysis as if it was statistically meaningful.
#
# I append this to system prompt when n<5 to force LLM into cautious mode.

_LOW_N_GUARD = """

KRITICKE UPOZORNENIE — NIZKY POCET PRIPADOV:
V kohorte je menej ako 5 podobnych pripadov. Toto znamena:
- NEFORMULUJ ziadne vseobecne zavery ani trendy ("sudy casto", "vacsina", "tendencia").
- NEDAVAJ odporucania co ma pravnik urobit (ani ohladom vysky sadzby ani argumentov).
- NEEXTRAPOLUJ z jednotlivych pripadov na pravnu prax ("toto naznacuje", "z precedensu vyplyva").
- Miesto toho: opis kazdy pripad jednotlivo — aky mal kontext, ako sud rozhodol, preco.
- Uved expicitne ze je malo dat: "Pre zadane kriteria mame len {n} pripadov, co je malo
  na vseobecne zavery. Nizsie opisujem konkretne rozhodnutia — pravnik musi sam posudit
  ich prenositelnost na svoj pripad."
- Odpoved kratka — max 2-3 odseky. Ziadne odpovede typu "na zaklade analyzy..."
"""


# #########################################
# FORMAT PRECEDENT CARD (for LLM prompt)
# #########################################

def _format_precedent_card(rec, rank):
    """Format one penalty record as text card for  LLM prompt.
    rec - penalty record
    rank - rank of precedens in top results

    I include structured data + reasoning summary + key quotes so LLM has enough context to cite the case meaningfully.
    """
    lines = []

    # header line — court, case number, date
    court = rec.get("court_name", "")
    case_num = rec.get("case_number", "")
    date = rec.get("decision_date", "")
    lines.append(f"[{rank}] {court} {case_num} ({date})")

    # contract info
    ct = rec.get("contract_type", "?")
    rt = rec.get("relationship_type", "?")
    lines.append(f"    Zmluva: {ct} | {rt}")

    # breach
    bt = rec.get("breach_type", "?")
    lines.append(f"    Porusenie: {bt}")

    # rate
    rate_raw = rec.get("rate_value_raw", "")
    if rate_raw:
        lines.append(f"    Sadzba: {rate_raw}")

    # amounts — build list of available amounts
    amounts_parts = []
    currency = rec.get("currency", "EUR")
    if rec.get("secured_principal") is not None:
        val = f"{rec['secured_principal']:,.0f}".replace(",", " ")
        amounts_parts.append(f"istina: {val} {currency}")
    if rec.get("original_claimed") is not None:
        val = f"{rec['original_claimed']:,.0f}".replace(",", " ")
        amounts_parts.append(f"pozadovane: {val} {currency}")
    if rec.get("final_awarded") is not None:
        val = f"{rec['final_awarded']:,.0f}".replace(",", " ")
        amounts_parts.append(f"priznane: {val} {currency}")
    if amounts_parts:
        lines.append(f"    Sumy: {', '.join(amounts_parts)}")

    # interest
    if rec.get("interest_awarded") == "yes":
        rate_val = rec.get("interest_rate", "")
        if rate_val:
            lines.append(f"    Urok z omeskania: ano ({rate_val})")
        else:
            lines.append(f"    Urok z omeskania: ano")

    # decision
    decision_labels = {
        "awarded_full": "POTVRDENA v plnej vyske",
        "moderated_301": "ZNIZENA podla §301",
        "dismissed": "ZAMIETNUTA",
        "returned": "VRATENA",
        "unclear": "NEJASNE",
    }
    dec = rec.get("decision", "unclear")
    lines.append(f"    Rozhodnutie: {decision_labels.get(dec, dec)}")

    # factors (only those court actually mentioned)
    factors = rec.get("factors", [])
    factor_strs = []
    for f in factors:
        sentiment = f.get("sentiment")
        if sentiment is not None and sentiment != "not_mentioned":
            factor_strs.append(f"{f['label']}({sentiment})")
    if factor_strs:
        lines.append(f"    Faktory: {', '.join(factor_strs)}")

    # reasoning summary — the court's argumentation
    reasoning = rec.get("legal_reasoning_summary", "")
    if reasoning:
        lines.append(f"    Odovodnenie: {reasoning}")

    # key quotes — exact citations from original document
    key_quotes = rec.get("key_quotes", [])
    for i, q in enumerate(key_quotes[:2]):  # max 2 quotes
        quote_text = q.get("quote", "")
        if quote_text:
            lines.append(f"    Citacia {i+1}: \"{quote_text}\"")

    return "\n".join(lines)


# #########################################
# MAIN SYNTHESIS FUNCTION
# #########################################

def synthesize(query_text, intent, analytics, top_penalties):
    """Generate answer using LLM.

    Takes query, intent from Stage 1, analytics from Stage 4, and top ranked penalties from Stage 3.

    Returns dict with statistics_text, llm_answer, precedent_cards, metadata.
    """
    from google.genai import types

    # i have 2 versions of stats:
    # - stats_for_llm: full detail (lift + raw counts both groups) for grounding
    # - stats_for_lawyer: plain language frequencies, no jargon, shown in output
    stats_for_llm = format_for_llm(analytics)
    stats_for_lawyer = format_for_lawyer(analytics)

    # format top 5 precedent cards for LLM prompt
    cards_lines = []
    for i in range(len(top_penalties[:5])):
        rec = top_penalties[i][0]       # record dict
        card = _format_precedent_card(rec, i + 1)
        cards_lines.append(card)
    cards_text = "\n\n".join(cards_lines)

    # build user prompt — query + full analytics (for LLM) + precedents
    intent_type = intent.get("intent", "general_precedent")
    n = analytics.get("n", 0)

    user_prompt = f"""OTAZKA PRAVNIKA:
{query_text}

ANALYZA (vypocitane z {n} podobnych pripadov):
{stats_for_llm}

TOP PRECEDENSY:
{cards_text}
"""

    # pick system prompt based on intent
    system_prompt = SYSTEM_PROMPTS.get(intent_type, SYSTEM_PROMPTS["general_precedent"])

    # if cohort is too small, append strict guard instructions so LLM doesnt
    # hallucinate authoritative-sounding answers from 1-4 cases.
    # i use 5 as threshold to match MIN_SAMPLE_SIZE in analytics.py
    if n < 5:
        system_prompt = system_prompt + _LOW_N_GUARD.format(n=n)

    # call LLM with retry for 503 errors (Gemini overload)
    client = _get_client()
    start = time.time()

    max_retries = 4
    response = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,          # a bit of creativity for natural text
                    max_output_tokens=4096,
                    # disable thinking — synthesis doesnt need it
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            break  # success
        except Exception as e:
            error_str = str(e)
            is_overload = "503" in error_str or "UNAVAILABLE" in error_str
            is_last_attempt = (attempt >= max_retries - 1)
            if is_overload and not is_last_attempt:
                wait = 10 * (attempt + 1)
                print(f"    Gemini overloaded, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

    latency = round(time.time() - start, 2)

    # get the answer text
    llm_answer = ""
    if response and response.text:
        llm_answer = response.text

    # build display cards for output (simpler than LLM cards)
    display_cards = []
    for i in range(len(top_penalties[:5])):
        rec = top_penalties[i][0]       # record
        score = top_penalties[i][1]     # score

        card = {
            "rank": i + 1,
            "penalty_id": rec.get("penalty_id", "pokuta_1"),
            "case_number": rec.get("case_number", ""),
            "court_name": rec.get("court_name", ""),
            "decision_date": rec.get("decision_date", ""),
            "contract_type": rec.get("contract_type", ""),
            "relationship_type": rec.get("relationship_type", ""),
            "breach_type": rec.get("breach_type", ""),
            "rate_value_raw": rec.get("rate_value_raw", ""),
            "decision": rec.get("decision", ""),
            "score": round(score, 3),
            "authority_level": rec.get("authority_level", 1),
        }
        display_cards.append(card)

    return {
        "statistics_text": stats_for_lawyer,  # plain language, no jargon
        "llm_answer": llm_answer,
        "precedent_cards": display_cards,
        "_metadata": {
            "model": MODEL_NAME,
            "latency_seconds": latency,
        },
    }
