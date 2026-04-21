"""
JSON schemas for multi-document precedent search.

I have 2 things here — enum values and the JSON schema for query understanding.
The schema forces LLM (Gemini Flash) to return exactly  fields i need when arsing a lawyer's query.
Same idea as in my extraction pipeline (src/extraction/schemas/output_schemas.py) where i force GPT-4o to return exact JSON structure.

example: lawyer writes "zmluva o dielo za 50k, omeskanie" and the LLM returns:
  {contract_type: "dielo", breach_type: "late_payment", amount_hint: 50000}

I use Gemini structured output (response_schema parameter) so it  cannot return something outside my enum lists.
Thats the same approach i used in extraction — without it the model sometimes skipped fields.
"""


# #########################################
# ENUM VALUES
# ########################################
# same enums as in extraction schema (src/extraction/schemas/extraction_schema.json)
# i copy them here so this multi_doc module doesnt depend on extraction code

CONTRACT_TYPES = [  # typ zmluvy
    "uver", "pozicka", "najom", "dielo", "kupna",
    "dodavka_sluzieb", "sprostredkovatelska", "telekom",
    "preprava", "mandatna", "ine", "nezname",
]

BREACH_TYPES = [    # typ porusenia zmluvy
    "late_payment",
    "non_monetary_performance",
    "early_termination",
    "breach_of_confidentiality",
    "other",
]

DECISION_TYPES = [
    "awarded_full",
    "moderated_301",
    "dismissed",
    "returned",
    "unclear",
]

# these are  7 legal factors that courts consider when moderating penalty
# i extracted them in my extraction pipeline for each penalty
FACTOR_LABELS = [
    "dobre_mravy",
    "zabezpecovacia_funkcia",
    "vyska_skody",
    "pomer_k_istine",
    "spravanie_dlznika",
    "kumulacia_s_urokom",
    "spravanie_veritela",
]

# what the lawyer wants — this determines what kind of answer we generate
INTENT_TYPES = [
    "safe_rate",           # lawyer is writing contract, wants to know safe penalty rate
    "defense_args",        # lawyer is defending client on court, wants moderation arguments
    "general_precedent",   # just searching for similar cases
]


# #########################################
# QUERY UNDERSTANDING SCHEMA (for Stage 1)
# #########################################
# this schema tells the LLM what to extract from the lawyer's query
# all filter fields are nullable — if lawyer doesnt mention contract type,
# LLM returns null (not guess). Null = "dont filter on this"

def get_query_understanding_schema():
    """Build  JSON schema that Gemini uses for structured output.

    I define  schema as nested dicts — same approach as in my extractiom output_schemas.py.
    Gemini reads this and guarantees  response matches.
    """
    schema = {
        "type": "OBJECT",
        "properties": {

            # --- structured filters (nullable — null if not in query) ---

            "contract_type": {
                "type": "STRING",
                "nullable": True,
                "enum": CONTRACT_TYPES,
                "description": "Typ zmluvy ak je zrejmy z query. Null ak nie je jasne.",
            },
            "breach_type": {
                "type": "STRING",
                "nullable": True,
                "enum": BREACH_TYPES,
                "description": "Typ porusenia ak je zrejmy z query. Null ak nie je jasne.",
            },
            "decision_interest": {
                "type": "STRING",
                "nullable": True,
                "enum": DECISION_TYPES,
                "description": "Ake rozhodnutie sudu zaujima pouzivatela. Null ak nepyta na konkretne rozhodnutie.",
            },
            "factor_interest": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                    "enum": FACTOR_LABELS,
                },
                "description": "Ktore pravne faktory pouzivatela zaujimaju. Prazdne pole ak nespomina ziadne.",
            },
            "amount_hint": {
                "type": "NUMBER",
                "nullable": True,
                "description": "Priblizna suma zmluvy/istiny v EUR ak je zrejma z query. Null ak nie je.",
            },

            # --- intent classification ---

            "intent": {
                "type": "STRING",
                "enum": INTENT_TYPES,
                "description": (
                    "Co chce pouzivatel: "
                    "safe_rate = hlada bezpecnu sadzbu pokuty (tvori zmluvu), "
                    "defense_args = hlada argumenty na obranu/moderaciu (zastupuje na sude), "
                    "general_precedent = vseobecne hladanie precedensov."
                ),
            },

            # --- semantic query for embedding search ---

            "semantic_query": {
                "type": "STRING",
                "description": (
                    "Preformuluj povodnu otazku do kratkeho textu (max 2 vety) "
                    "ktory zachytava podstatu hladania — typ zmluvy, typ porusenia, "
                    "klucove faktory. Tento text sa pouzije na semanticke vyhladavanie."
                ),
            },
        },
        "required": [
            "contract_type", "breach_type", "decision_interest",
            "factor_interest", "amount_hint", "intent", "semantic_query",
        ],
    }
    return schema


# #########################################
# QUERY UNDERSTANDING PROMPT
# #########################################
# system prompt for the LLM that parses lawyer's query
# i keep it here next to the schema so everything about query understanding is in one file

QUERY_UNDERSTANDING_SYSTEM_PROMPT = """\
Si system na analyzu pravnych dotazov o zmluvnych pokutach v slovenskom prave.

Tvoja uloha: z dotazu pravnika extrahuj strukturovane filtre pre vyhladavanie v databaze sudnych rozhodnuti.

PRAVIDLA:
- Ak nieco nie je jasne z dotazu, vrat null (nehadaj).
- contract_type urcuj podla obsahu zmluvy, nie podla presneho slova.
  Napriklad "IT zakazka" moze byt "dielo" alebo "dodavka_sluzieb".
- breach_type: "omeskanie", "neplatenie", "nezaplatil" = late_payment.
  "nesplnil", "nedodal", "neodovzdal" = non_monetary_performance.
- intent:
  * safe_rate — ak pravnik tvori zmluvu a chce vediet aku sadzbu nastavit
  * defense_args — ak pravnik zastupuje klienta a chce argumenty na znizenie/moderaciu
  * general_precedent — vsetko ostatne (hladanie precedensov, statistik, prehladov)
- semantic_query: preformuluj dotaz do 1-2 viet zachytavajucich podstatu hladania.

TYPY ZMLUV: uver, pozicka, najom, dielo, kupna, dodavka_sluzieb, sprostredkovatelska, telekom, preprava, mandatna, ine, nezname
TYPY PORUSENIA: late_payment, non_monetary_performance, early_termination, breach_of_confidentiality, other
ROZHODNUTIA: awarded_full (potvrdena), moderated_301 (znizena §301), dismissed (zamietnuta), returned (vratena), unclear
FAKTORY: dobre_mravy, zabezpecovacia_funkcia, vyska_skody, pomer_k_istine, spravanie_dlznika, kumulacia_s_urokom, spravanie_veritela
"""
