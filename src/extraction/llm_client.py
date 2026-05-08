"""
LLM client wrapper for my extraction pipeline (translator between extraction system and different LLM models).
- i can call just call_llm(model_key, system_prompt, user_prompt, call_type="call1") ant it handles:
    - choosing right model
    - setting client
    - sendingp prompt
    - enforcing json format based on my schema
    - retry when error occurs
    - measeuring toknes, price and time

Provide unified interface for calling different LLM APIs:
- OpenAI (GPT-4o) -  baseline, should have best structured output support
- Google Gemini (2.5 Pro) - huge context window, good for full-doc mode (ofc that even opeani context is enough)
- Qwen 3.5 397B (throguh openRouter) - open-source (Apache 0.2), MoE

- # changed Ollama (Llama 3.1 70B) - local/open-source, privacy-friendly # changes

Each model returns JSON string. The caller is responsible for parsing it.

ive tested prompts in these playgrounds before writing this code:
- OpenAI: https://platform.openai.com/playground (free tier has some credits)
- Google AI Studio: https://aistudio.google.com (free tier for Gemini)
- Qwen (tried slovak version at first Qwen3-14B-sk-Q6_K, but was repeating itself and was halucinating a lot):
        just run locally with 'ollama run hf.co/tomg42/Qwen3-14B-sk-Q6_K.gguf

"""

import os
import json
import time

# 3 types of outputs when calling llm (just call1, call2 or full doc)
from src.extraction.schemas.output_schemas import (
    get_call1_schema, get_call2_schema, get_fulldoc_schema,
)


###########################################
# MODEL CONFIGS
# ##########################################
# i keep these here so i can easily change model versions later.
# the model name  is what gets passed to the API.
#
# TEMPERATURE = 0.0 for all models. For extraction i want deterministic output -
# the same document should always produce the same JSON (even though i know thats not 100% possible without seed).
# Higher temperature adds randomness which is good for creative writing but bad for extracting
# facts from legal text. I also dont set top_p because at temperature=0
# the model always picks the most probable token, so top_p has no effect.
#
# MAX_OUTPUT_TOKENS: simple documents are usually under 1000 output tokens per call,
# but multi-penalty cases with evidence quotes can be much longer. I use 8192 for
# Call 1 and Call 2, and 16384 for full-doc mode where the model returns the
# complete schema in one response.
#
# With structured outputs the model CANT be verbose - it has to fill the schema
# fields and stop. It cant add explanations or padding outside the JSON structure.
# So max_output_tokens is really just a safety limit, not a behavior control.
# The free-text fields (dispute_summary, legal_reasoning_summary) are constrained by prompt instructions
# ("2-3 sentences", "3-5 sentences").


# dict with all models i have used
MODEL_CONFIGS = {
    "gpt-4o": {
        "provider": "openai",
        "model_name": "gpt-4o",     # real API name of model
        "temperature": 0.0,
    },
    "gemini-2.5-flash": {
        "provider": "google",
        # FIX: i originally wanted gemini-2.5-pro but free tier has 0 quota
        # for it. Flash only has 20 req/day which is too slow for 176 docs.
        # So i use flash for the 20 golden docs (high quality comparion)
        # and flash-lite for all 176 docs (enough quota: 1000 req/day).
        "model_name": "gemini-2.5-flash",
        "temperature": 0.0,
    },
    "gemini-2.5-flash-lite": {
        "provider": "google",
        # 1000 req/day on free tier
        # Its less accurate but fast and has enough quota for all 176 docs
        # in one night (352 calls for RAG mode). I use it for the full
        # corpus extraction while Flash handles golden (not the same golden but chosen for extraction) dataset.
        "model_name": "gemini-2.5-flash-lite",
        "temperature": 0.0,
    },
    # === OPEN-SOURCE MODEL: Qwen 3.5 397B via OpenRouter ===
    # I want to compare commercial models (GPT-4o, Gemini) with  open-source one.
    # Qwen 3.5 397B-A17B is currently  best open-weight model (Apache 2.0 license).
    # Its a MoE (Mixture of Experts) - 397B total params but only 17B active per
    # token, so its efficient despite being huge - also has 262k token context.
    #
    # I use OpenRouter (openrouter.ai) to access it via API because:
    #   - I cant run 397B locally on my MacBook Air M2 24GB (needs ~200GB+ VRAM)
    #   - OpenRouter has OpenAI-compatible API so i can reuse  same SDK
    #   - Its cheap: $0.39/M input + $2.34/M output (6x cheaper than GPT-4o)
    #   - OpenRouter pick the best provider automatically (DeepInfra, Together, etc.)
    #
    # For the thesis argument: the model is OPEN-SOURCE (anyone can download weights
    # from HuggingFace and run it on their own server). Im just using API access
    # because i dont have enough RAM (Mac studio ultra 3 512GB RAM would do it :)
    # But law firm with privacy concerns could self-host this on their own infrastructure
    #

    # API key: export OPENROUTER_API_KEY="sk-or-..."
    "qwen3.5-397b": {
        "provider": "openrouter",
        "model_name": "qwen/qwen3.5-397b-a17b",
        "temperature": 0.0,
    },

    # === OLD did not go as expected: Slovak fine-tuned Qwen (LOCAL, via Ollama on macbook air m2) ===
    # I originally tried running a small Slovak model locally for the privacy
    # argument ("data never leaves MY machine"). I chose Qwen3-14B-sk which
    # is fine-tuned for Slovak by SAV and TU Košice. It fits on my m2 mac just with Q6_K quantization (12.1 GB).
    #
    # RESULT: total failure. 0/20 documents - the model cant do structured JSON
    # extraction at all. It was fine-tuned on Slovak text but NOT on instruction following or structured output
    # It just repeats itself in loop and never produces valid JSON.
    # The base Qwen3-14B might workbetter but i switched to Qwen 3.5 397B via OpenRouter instead - much more capable and still open-source.
    #
    # I keep this config here for documentation. If someone wants to try it:
    #   1. Install Ollama: brew install ollama
    #   2. Download model: ollama run hf.co/tomg42/Qwen3-14B-sk-Q6_K.gguf
    #   3. Set NO_PROXY: export NO_PROXY="localhost,127.0.0.1"
    #   4. Uncomment this config and comment out qwen3.5-397b above
    #
    # "qwen3-14b-sk": {
    #     "provider": "ollama",
    #     "model_name": "hf.co/tomg42/Qwen3-14B-sk-Q6_K.gguf",
    #     "temperature": 0.0,
    #     # MODEL: slovak-nlp/Qwen3-14B-sk (HuggingFace)
    #     # GGUF:  tomg42/Qwen3-14B-sk-Q6_K.gguf (quantized for Ollama)
    #     # Also available: ericek111/Qwen3-14B-sk-Q4_K_M-GGUF (smaller, 9GB)
    # },
}

# max output tokens per call type - different because full-doc returns more data
MAX_TOKENS_PER_CALL = {
    # FIX: bumped call1 from 4096 to 8192. Gemini was truncating on 2Cob/69/2020
    # (3-penalty case): raw response reached 11095 chars (~4000 tokens) and cut
    # mid-sentence before closing brackets, causing JSON parse errors.
    # 8192 gives enough headroom for 3-5 penalties with full evidence quotes.
    "call1": 8192,    # contract facts + penalty definitions (multi-penalty can be 6k+ tokens)
    # FIX: also bumped call2 to 8192. Multi-penalty moderation analysis with 7
    # factors per penalty × 3 penalties = 21 factor objects with sentiment+evidence.
    "call2": 8192,    # moderation analysis (multi-penalty 7-factor analysis can be 5k+ tokens)
    # FIX: bumped fulldoc from 8192 to 16384 for multi-penalty cases (e.g. 2Cob/69/2020
    # with 3 penalties produced 6400+ tokens of formatted JSON).
    "fulldoc": 16384, # everything at once - needs headroom for multi-penalty cases
}


# ##########################################
# LAZY-INITIALIZED CLIENTS
# ##########################################
# FIX: I originally created new OpenAI() client inside _call_openai() every  time
# For 20 docs x 2 calls = 40 client instantiations.
# Worked fine but was wasteful - each instantiation reads env vars, sets up HTTP session, etc.
# Now i create  client once on first use and reuse it for all subsequent calls.
# Same for Gemini client.

# I use a dict instead of global variables because its cleaner and i can reset it in tests
_clients = {}   # ict is created in first usage (then reusing)


def _get_openai_client():
    """Get or create the OpenAI client (singleton )."""
    if "openai" not in _clients:
        from openai import OpenAI   # client object created by callin constructor openai() through openai class
        _clients["openai"] = OpenAI()  # tries to find OPENAI_API_KEY from env variables (export OPENAI_API_KEY="sk-....")
        # also can do like that - client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]

#  this is how i can see env variables (which has process)
# import os     print(os.environ.get("OPENAI_API_KEY"))

# ########### CHECK if singleton works as i expect ##########
# if __name__ == "__main__":
#     print("Initial _clients:", _clients)
#
#     c1 = _get_openai_client()
#     print("After first _get_openai_client():", _clients)
#
#     c2 = _get_openai_client()
#     print("After second _get_openai_client():", _clients)
#
#     print("c1 is c2:", c1 is c2)
#     print("'openai' in _clients:", "openai" in _clients)
# ######################


def _get_google_client():
    """Same as above - get or create the Google GenAI client (singleton )."""
    if "google" not in _clients:
        from google import genai
        _clients["google"] = genai.Client()  # reads GEMINI_API_KEY or GOOGLE_API_KEY
    return _clients["google"]


def _get_openrouter_client(): # same library, different endpoint, different API key
    """Again just for openrouter - get or create the OpenRouter client (singleton ).

    OpenRouter has OpenAI-compatible API so i justr reuse the OpenAI SDK - just change
    base_url to openrouter.ai. This way i dont need a separate library.
    The API key is different tho - its OPENROUTER_API_KEY, not OPENAI_API_KEY.
    """
    if "openrouter" not in _clients:
        from openai import OpenAI
        # explicitly loading the key - because openai() can automatically find just OPENAI_API_KEY
        api_key = os.environ.get("OPENROUTER_API_KEY", "")      # oepnai would search for its standard key OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/settings/keys"
            )
        _clients["openrouter"] = OpenAI(        # have to change url and api key
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _clients["openrouter"]


# just helper function
def _strip_additional_properties(obj):
    """Recursively remove 'additionalProperties' from a JSON schema dict

    FIX: OpenAI strict mode REQUIRES additionalProperties:false on every object.
    But Gemini doesnt support this field at all and returns 400 INVALID_ARGUMENT if it sees it, and bcs of that i remove it
    So i strip it before sending to Gemini (resurcisvely because it needs to be in each level of json)
    Schema still work because Gemini enforces  structure through response_schema without needing this flag - it just doesnt allow extra fields by default.
    """
    if isinstance(obj, dict):   # if obj is dict, create new dict
        new_obj = {}
        for k, v in obj.items():    # go through key-valude paires
            if k != "additionalProperties":     # skip this key
                new_obj[k] = _strip_additional_properties(v)    # other insert into dict and process recursively
        return new_obj
    elif isinstance(obj, list): # if obj is list, process each element
        return [_strip_additional_properties(item) for item in obj]
    return obj


def call_llm(model_key, system_prompt, user_prompt, call_type="call1"):
    """Call  LLM and return the response text + metadata

    Arguments:
        model_key: one of  keys in MODEL_CONFIGS (e.g. "gpt-4o")
        system_prompt:  system message (extraction rules)
        user_prompt:  user message (chunks + schema)
        call_type: "call1", "call2", or "fulldoc" - determines which JSON schema to use for structured output and how many output tokens to allow

    Returns:
        dict with:
            "response_text": raw text from the model (should be JSON)
            "model": model name used
            "provider": openai/google/ollama
            "prompt_tokens": input token count (approximate for some providers)
            "completion_tokens": output token count
            "latency_seconds": time taken for the API call
            "cost_usd": estimated cost (null for free/local models (gemini 2.5 flash, qwen 3.5)
    """
    config = MODEL_CONFIGS[model_key]   # load configs which is defined in the start of file
    provider = config["provider"]       # openai, google, openrouter
    max_tokens = MAX_TOKENS_PER_CALL.get(call_type, 4096) # getting output tokens limit based on call

    # pick right JSON schema for this call type
    schema_map = {"call1": get_call1_schema,
                  "call2": get_call2_schema,
                  "fulldoc": get_fulldoc_schema} # schemamap holds functions

    output_schema = schema_map.get(call_type, get_call1_schema)()   # this calls the function, based on call or type

    ################# startin timer  #################
    start_time = time.time()

    # dispatcher - this call_llm is single entry point - now call concrete implementation based on model
    if provider == "openai":
        result = _call_openai(config, system_prompt, user_prompt, max_tokens, output_schema)
    elif provider == "google":
        result = _call_google(config, system_prompt, user_prompt, max_tokens, output_schema)
    elif provider == "openrouter":
        result = _call_openrouter(config, system_prompt, user_prompt, max_tokens, output_schema)
    elif provider == "ollama":
        result = _call_ollama(config, system_prompt, user_prompt, max_tokens, output_schema)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    result["latency_seconds"] = round(time.time() - start_time, 2)
    result["model"] = config["model_name"]
    result["provider"] = provider

    return result


# ##########################################
# OPENAI (GPT-4o)
# ##########################################
# IMPORTANT: I use response_format with "json_schema" (Structured Outputs), NOT older "json_object" (JSON Mode)
# The difference is huge:
#
# - json_object (old): guarantees valid JSON syntax but the model can still
#   skip fields, add extra fields, or use wrong types. I used this at first
#   and the model sometimes forgot the evidence field.
#
# - json_schema (new, Structured Outputs): guarantees the output EXACTLY matches
#   my schema. The model literally cannot skip a required field or return a wrong
#   type. OpenAI says this gets 100% schema adherence. This is much better for
#   extraction because i need every field to be there for validation to work.
#
# API playground: https://platform.openai.com/playground
# Pricing: https://openai.com/api/pricing/
# As a student I got $5 free credits when I signed up. After that its pay-as-you-go.
# GPT-4o costs about $2.50/1M input tokens + $10/1M output tokens (as of early 2026).
# For 176 documents thats roughly $5-15 total depending on document length.

def _call_openai(config, system_prompt, user_prompt, max_tokens, output_schema):
    """Provider specific wrapper for opneAI, which call model, send system_user prompt, force json output, retry wher rate limit, get response """
    client = _get_openai_client()   # will call clients function, which sends HTTP request to openai server

    # FIX: retry on rate limit (429) errors. When processing 176 documents
    # back to back, OpenAI sometimes throttles me after ~10-15 rapid requests.
    # Simple retry with increasing wait time fixes this. I originally didnt
    # have this and the script crashed at document 12 of 176.
    import time as _time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config["model_name"],     # "gpt-4o" in my case
                messages=[
                    {"role": "system", "content": system_prompt},   # general rules - what should model do, what constraints are, etc...
                    {"role": "user", "content": user_prompt},   # specific work - document text (chunks)
                ],
                temperature=config["temperature"],  # less creativity, more consistency
                max_tokens=max_tokens,  # limit of outuptut tokens
                # Structured outputs - forces the model to follow my exact JSON schema.
                response_format={
                    "type": "json_schema",  # json_schema (valid json with strict structure) / json_object (just give me valid json)
                    "json_schema": output_schema,
                },
            )
            break  # success, exit retry loop
        except Exception as e:      # when too many requests - openai can rate limit
            if "429" in str(e) and attempt < max_retries - 1:   # HTTP 429 = too many requests
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                _time.sleep(wait)
            else:
                raise  # not  rate limit error or last attempt, let it crash

    usage = response.usage  # how many toknes was in input and how many on outputs
    # cost calculation based on openai pricing (gpt-4o as of 2026)
    cost = (usage.prompt_tokens * 2.5 + usage.completion_tokens * 10.0) / 1000000

    return {
        "response_text": response.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": round(cost, 4),
    }


# ##########################################
# OPENROUTER (Qwen 3.5 397B)
# ##########################################
# OpenRouter (openrouter.ai) is API aggregator - it gives one endpoint to access 300+ models from different providers (DeepInfra, Together, Fireworks)
# The API is OpenAI-compatible so i just reuse the OpenAI SDK with different base_url.
# The code looks almost identical to _call_openai().
#
# I use it for Qwen 3.5 397B-A17B which is  current SOTA open-source model (multilingual)
# Its a MoE model (397B total params, 17B active) so its fast despite being huge.
# OpenRouter supports structured outputs (json_schema) for this model, same as OpenAi
#
# Pricing: 0.39/1M input + 2.34/1M output - about 6x cheaper than GPT-4o.
# For my 176 documents the total cost should be around 1-3 dollars

#   export OPENROUTER_API_KEY="sk-or-v1-..."

# openrouter pricing per million tokens (just for cost tracking)
_OPENROUTER_QWEN_INPUT_COST_PER_M = 0.39
_OPENROUTER_QWEN_OUTPUT_COST_PER_M = 2.34

def _call_openrouter(config, system_prompt, user_prompt, max_tokens, output_schema):
    client = _get_openrouter_client()

    # retry logic - same as OpenAI. OpenRouter can return 429 when the underlying
    # provider is overloaded, or 502/503 for temporary failures
    import time as _time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config["temperature"],
                max_tokens=max_tokens,
                # structured outputs - OpenRouter passes this to the provider
                # same format as OpenAI because the API is compatible
                response_format={
                    "type": "json_schema",
                    "json_schema": output_schema,
                },
                # FIX: Qwen 3.5 is a "thinking" model - it spends tokens on internal reasoning before outputting JSON
                # Without this, simple 20-token JSON response costs 1200+ tokens because the model thinks first
                # For extraction i dont need thinking - just fill the JSON schema
                # OpenRouter has unified "reasoning" parameter that works across all thinking models (Qwen, DeepSeek, etc). effort="none" disables it.
                extra_body={"reasoning": {"effort": "none"}},
            )
            break
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "502" in err_str or "503" in err_str) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    OpenRouter error, waiting {wait}s (attempt {attempt+1}/{max_retries}): {err_str[:80]}")
                _time.sleep(wait)
            else:
                raise

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    cost = (prompt_tokens * _OPENROUTER_QWEN_INPUT_COST_PER_M +
            completion_tokens * _OPENROUTER_QWEN_OUTPUT_COST_PER_M) / 1000000

    return {
        "response_text": response.choices[0].message.content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 4),
    }


# ##########################################
# GOOGLE GEMINI (2.5 Pro)
# ##########################################
# Gemini has very big context window (1M+ tokens) which makes it perfect for full-document experiment where isend the entire court decision
#
# I use response_mime_type="application/json" together with response_schema to get structured output
# Gemini also supports JSON schema constraints similar to OpenAI, but through different parameter name
#
# Google AI Studio: https://aistudio.google.com (free playground, no code needed)
#
# FIX: I now estimate the cost even though i use the free tier. If i ever exceed free limits, google charges per token. The pricing (as of 2026) is :
# - input:  1.25/1M tokens (prompts <= 200k tokens)
# - output: 10.00/1M tokens
# For my 20 golden docs this would be ~1.50 if i had to pay
# But free tier covers it easily (1500 requests/day, i need 40 for RAG mode).
#
#  API uses google-genai library (pip install google-genai)

# gemini pricing per million tokens (for cost estimation even on free tier)
_GEMINI_INPUT_COST_PER_M = 1.25
_GEMINI_OUTPUT_COST_PER_M = 10.00

def _call_google(config, system_prompt, user_prompt, max_tokens, output_schema):
    from google.genai import types

    client = _get_google_client()

    # FIX: strip additionalProperties - Gemini doesnt support it and returns (implemented helper function for this)
    # 400 INVALID_ARGUMENT if its there - OpenAI needs it, Gemini doesnt.
    gemini_schema = _strip_additional_properties(output_schema.get("schema", {}))

    # FIX: gemini 2.5 flash has a "thinking" mode that eats up output token before producing the JSON
    # This caused 30% of my calls to return emptyn responses ("No JSON object found")
    # Setting thinking_budget=0 disables internal reasoning and makes it output JSON directly
    # Also: retry with longer waits for rate limits AND 503 UNAVAILABLE (overloaded model).
    # Free tier has only 20 RPD for Flash, paid tier is betterbut still has RPM limits.
    import time as _time
    max_retries = 10  # raised from 6 - 503 overload errors can persist for 5-10 min
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config["model_name"],
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=config["temperature"],
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",  # response has to be json (little bit different than openai)
                    response_schema=gemini_schema,
                    # disable thinking - it wastes tokens on internal reasoning
                    # and often returns empty JSON as a result
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            # check for empty response - gemini sometimes returns None
            if response.text and response.text.strip():
                break
            elif attempt < max_retries - 1:
                print(f"    Empty response, retrying (attempt {attempt+1}/{max_retries})...")
                _time.sleep(5)
            else:
                break  # give up, parser handles empty
        except Exception as e:
            err_str = str(e)
            # FIX: 503 UNAVAILABLE happens when the model is overloaded ("experiencing high demand").
            # These are transient - the server tells us to try again later.
            # Also retry 500, 502, 504 (server-side errors) + network errors.
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
            is_server_error = "503" in err_str or "UNAVAILABLE" in err_str or "500" in err_str or "502" in err_str or "504" in err_str or "INTERNAL" in err_str
            is_network = "connection" in err_str.lower() or "timeout" in err_str.lower() or "reset" in err_str.lower()

            if (is_rate_limit or is_server_error or is_network) and attempt < max_retries - 1:
                if is_rate_limit:
                    wait = 60 * (attempt + 1)  # rate limit: 60s, 120s, 180s, ...
                    reason = "Rate limited"
                elif is_server_error:
                    wait = min(15 * (2 ** attempt), 300)  # server overload: 15s, 30s, 60s, 120s, 240s, 300s
                    reason = "Server overloaded (503/UNAVAILABLE)"
                else:
                    wait = min(10 * (2 ** attempt), 120)
                    reason = "Network error"
                short_err = err_str[:100].replace('\n', ' ')
                print(f"    {reason}, waiting {wait}s (attempt {attempt+1}/{max_retries}): {short_err}")
                _time.sleep(wait)
            else:
                raise

    # gemini usage metadata
    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count if usage else 0
    completion_tokens = usage.candidates_token_count if usage else 0

    # FIX: estimate cost even on free tier - useful for thesis cost comparison
    # table and for knowing how much it WOULD cost if i had to pay
    cost = (prompt_tokens * _GEMINI_INPUT_COST_PER_M +
            completion_tokens * _GEMINI_OUTPUT_COST_PER_M) / 1_000_000

    return {
        "response_text": response.text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 4),
    }


# ==========================================
# OLLAMA (Qwen3-14B-sk - local Slovak model)
# ==========================================
# I ran Qwen3-14B-sk LOCALLY on my MacBook Air M2 24GB via Ollama
# This is  open-source / privacy option for my thesis - no data leaves my machine
#
# The model is a Slovak fine-tune of Qwen3-14B by SAV and TU Košice.
# I use Q6_K quantization (12.1 GB) which leaves about 6GB for the OS and inference context
# I keep num_ctx=4096 to be safe on RAM - our RAG chunks are well under 4k tokens so this works for RAG mode

# For fulldoc some very long documents (120k chars = 30k tokens) wont fit in 4k context, so fulldoc results may be incomplete for those
# (never mind because model did not work as expected)
#
# If OLLAMA_API_KEY is set, the code falls back to Ollama Cloud
# (for the Qwen 3.5 397B cloud model etc). Without it, goes to localhost.
#
# Since Ollama v0.5  supports structured output through the "format" parameter - grammar-based constrained decoding guarantees  output
# matches the JSON schema.
# schema is also in the prompt text as recommended by Ollama docs (belt and suspenders approach)

def _call_ollama(config, system_prompt, user_prompt, max_tokens, output_schema):
    import requests

    # decide: cloud or local? if OLLAMA_API_KEY is set, use cloud endpoint.
    # otherwise fall back to localhost for local model
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if api_key:
        base_url = "https://ollama.com"
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        base_url = "http://localhost:11434"     # ollama defaultly listens here 127.0.0.1:11434
        headers = {}

    # FIX: retry on timeout or connection errors. Cloud can have rate limits,
    # local can have timeouts on large models.
    import time as _time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post( # Ollama API is imple HTTP API
                f"{base_url}/api/chat",
                headers=headers,
                json={
                    "model": config["model_name"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": config["temperature"],
                        "num_predict": max_tokens,
                        # FIX: set context window explicitly. Ollama default is 2048 which is  too small for our prompts
                        # i originally set 4096 but after checking actual prompt sizes i found that 18/20 evaluation docs  exceed 4k tokens
                        # Even 8k only fits 10/20. So i set 16384 which fits all 20 docs.
                        # This uses more RAM (~4GB extra for KV cache) but on my M2 24GB with Q6 model (12GB) i still have about 8GB left which should be enough. (little bit slower)
                        "num_ctx": 16384,
                    },
                    # structured output - ollama uses grammar-based constrained decoding
                    "format": output_schema.get("schema", {}),
                },
                timeout=1800,  # 30 min timeout - local 14B model on M2 is SLOW
            )
            response.raise_for_status()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    Ollama error, retrying in {wait}s (attempt {attempt+1}/{max_retries}): {str(e)[:80]}")
                _time.sleep(wait)
            else:
                raise

    data = response.json()
    message = data.get("message", {})

    # ollama returns token counts in different fields
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    return {
        "response_text": message.get("content", ""),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": 0.0,  # local = free
    }
