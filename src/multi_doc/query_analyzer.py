"""
Stage 1: Query Understanding — parse lawyer's query into structured filters.

This is the entry point of the multi-doc pipeline.
Lawyer types something like "Robim zmluvu o dielo za 50k EUR, aku pokutu za omeskanie?" and this module turns it into structured filters:
  {
  "contract_type": "dielo",
  "breach_type": "late_payment",
  "decision_interest": null,
  "factor_interest": [],
  "amount_hint": 50000,
  "intent": "safe_rate",
  "semantic_query": "zmluvna pokuta za omeskanie pri zmluve o dielo"
}


I use Gemini 2.5 Flash with structured output for this.
I originaly considered regex but Slovak has 6 grammatical cases (zmluva/zmluvy/zmluvou/zmluve...) which makes regex not really working.
LLM handles morphology natively and costs about $0.003 per query.

The LLM also classifies  intent — is the lawyer writing a contract (safe_rate), defending a client (defense_args), or just searching (general_precedent)?
This changes what output the system produces in later stages.
"""

import os
import sys
import time
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.multi_doc.schemas import (
    get_query_understanding_schema,
    QUERY_UNDERSTANDING_SYSTEM_PROMPT,
)


# #########################################
# CONFIG
# #########################################
# gemini 2.5 flash — fast and cheap, good at classification tasks
# flash-lite would also work but flash handles nuanced Slovak better

MODEL_NAME = "gemini-2.5-flash"


# #########################################
# GEMINI CLIENT (singleton — same pattern as extraction llm_client.py)
# #########################################
# i create client once and reuse it for all queries in the session
# google.genai.Client() reads GEMINI_API_KEY or GOOGLE_API_KEY from env

_client = None

def _get_client():
    """Get or create Gemini client (created once, reused after)."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()
        # _client = genai.Client(api_key="XX XX")

    return _client


# #########################################
# MAIN FUNCTION
# #########################################

def analyze_query(query_text):
    """Parse a lawyer's query into structured filters using LLM.

    Takes raw query string in Slovak, sends it to Gemini Flash with
    structured output schema, returns dict with contract_type, breach_type,
    decision_interest, factor_interest, amount_hint, intent, semantic_query.

    Any filter field can be None — means "dont filter on this".
    """
    from google.genai import types

    client = _get_client()
    schema = get_query_understanding_schema()

    start = time.time()

    # retry on 503 errors — Gemini sometimes gets overloaded
    # same retry pattern as in my extraction llm_client.py
    max_retries = 4
    response = None

    for attempt in range(max_retries):
        try:
            # CALLING GEMINI
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=query_text,
                config=types.GenerateContentConfig(
                    system_instruction=QUERY_UNDERSTANDING_SYSTEM_PROMPT,
                    temperature=0.0,          # deterministic — same query = same filters
                    max_output_tokens=1024,   # structured output is small
                    # i disable thinking (budget=0) — for simple classification
                    # thinking just wastes tokens without improving results
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_schema=schema,
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

    # parse JSON response from Gemini
    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"  WARNING: Failed to parse query analysis response: {e}")
        print(f"  Raw response: {response.text[:300]}")
        # fallback — return empty intent so pipeline continues without filters
        result = {
            "contract_type": None,
            "breach_type": None,
            "decision_interest": None,
            "factor_interest": [],
            "amount_hint": None,
            "intent": "general_precedent",
            "semantic_query": query_text,
        }

    # add metadata about the LLM call
    result["_metadata"] = {
        "model": MODEL_NAME,
        "latency_seconds": latency,
    }

    return result


# #########################################
# QUICK TEST
# #########################################
# runing this file directly to test query analysis:
#   python src/multi_doc/query_analyzer.py

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)

    test_queries = [
        "Robim zmluvu o dielo za 50 000 EUR. Aku pokutu za omeskanie platieb nastavit?",
        "Klient dostal pokutu 0,5% denne z najomnej zmluvy. Ake argumenty na moderaciu?",
        "Ake faktory sud najviac zvazuje pri moderacii podla §301?",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        result = analyze_query(q)
        for key, value in result.items():
            if key != "_metadata":
                print(f"  {key}: {value}")
        print(f"  (latency: {result['_metadata']['latency_seconds']}s)")
