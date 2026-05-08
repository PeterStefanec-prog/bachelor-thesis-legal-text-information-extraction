"""
JSON Schema definitions for LLM structured output.

These schemas tell the LLM exactly what structure to return. I use them with:
- OpenAI: response_format={"type": "json_schema", "json_schema": {...}}
- Gemini: response_schema parameter in GenerateContentConfig
- Ollama: format parameter in the chat API

Why structured output instead of just asking nicely in the prompt?
Before I added these schemas,  LLM sometimes:
  - forgot fields (no breach_type in the output)
  - added extra fields I didnt ask for
  - used wrong types (string "180" instead of number 180)
  - returned inconsistent structure between documents

With structured output the API FORCES the model to follow  schema exactly.
OpenAI calls this "Structured Outputs" and claims 100% schema adherence.
Gemini and Ollama use grammar-based constrained decoding to achieve  same.

I have 3 different schemas because the 3 call types return different things:
  - call1: case_context + penalties (without moderation)
  - call2: just the moderation analysis for each penalty
  - fulldoc: everything in one shot

Meta is not returned by the LLM schemas anymore.
It is added later in run_extraction.py from deterministic regex/preprocessing metadata.
"""


# ==========================================
# HELPER FUNCTIONS
# ==========================================
# these make the schemas shorter and less repetitive.
# OpenAI strict mode requires every nullable field to use "anyOf" with explicit null type,
# and every object needs "additionalProperties: false"
# and ALL properties must be in "required".
# its very strict but thats the point.

def _nullable_string():
    """A string field that can also be null."""
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _nullable_number():
    """A number field that can also be null."""
    return {"anyOf": [{"type": "number"}, {"type": "null"}]}


def _evidence_obj():
    """Evidence object: quote + chunk_id. Both nullable because sometimes
    the LLM cant find a supporting quote (e.g. contract_type is inferred,
    not explicitly stated in one chunk)."""
    return {
        "type": "object",
        "properties": {
            "quote": _nullable_string(),
            "chunk_id": _nullable_string(),
        },
        "required": ["quote", "chunk_id"],
        "additionalProperties": False,
    }


def _factor_obj():
    """One factor in the moderation analysis.
    There are always exactly 7 factors with sentiment and evidence.
    I added spravanie_veritela (creditor passivity) after reading NS SR decisions - courts sometimes consider whether the creditor
    just sat and waited for the penalty to grow instead of enforcing performance."""
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": [
                    "dobre_mravy", "zabezpecovacia_funkcia", "vyska_skody",
                    "pomer_k_istine", "spravanie_dlznika", "kumulacia_s_urokom",
                    "spravanie_veritela",
                ],
            },
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "not_mentioned"],
            },
            "evidence": _evidence_obj(),
        },
        "required": ["label", "sentiment", "evidence"],
        "additionalProperties": False,
    }


def _meta_obj():
    """Document metadata extracted from the header. Not used - bcs i get them with regexes"""
    return {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string"},
            "court_name": {"type": "string"},
            "case_number": {"type": "string"},
            "decision_date": _nullable_string(),
            "ecli": _nullable_string(),
        },
        "required": ["doc_id", "court_name", "case_number", "decision_date", "ecli"],
        "additionalProperties": False,
    }


def _case_context_obj():
    """Case context - summary + classification."""
    return {
        "type": "object",
        "properties": {
            "dispute_summary": {"type": "string"},
            "verdict_summary": {"type": "string"},
            "contract_type": {
                "type": "string",
                "enum": [
                    "uver", "pozicka", "najom", "dielo", "kupna",
                    "dodavka_sluzieb", "sprostredkovatelska", "telekom",
                    "preprava", "mandatna", "ine", "nezname",
                ],
            },
            "relationship_type": {
                "type": "string",
                "enum": ["B2B", "B2C", "C2C", "unknown"],
            },
        },
        "required": ["dispute_summary", "verdict_summary", "contract_type", "relationship_type"],
        "additionalProperties": False,
    }


def _amount_with_evidence():
    """An amount field (number) with evidence.
    Used for secured_principal, original_claimed, final_awarded."""
    return {
        "type": "object",
        "properties": {
            "value": _nullable_number(),
            "evidence": _evidence_obj(),
        },
        "required": ["value", "evidence"],
        "additionalProperties": False,
    }


def _amounts_obj():
    """All monetary amounts for one penalty."""
    return {
        "type": "object",
        "properties": {
            "currency": {
                "type": "string",
                "enum": ["EUR", "SKK", "CZK", "unknown"],
            },
            "secured_principal": _amount_with_evidence(),
            "original_claimed": _amount_with_evidence(),
            "final_awarded": _amount_with_evidence(),
        },
        "required": ["currency", "secured_principal", "original_claimed", "final_awarded"],
        "additionalProperties": False,
    }


def _moderation_analysis_obj():
    """The moderation analysis for one penalty - decision, reasoning, factors."""
    return {
        "type": "object",
        "properties": {
            "decision": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "enum": ["awarded_full", "moderated_301", "dismissed", "returned", "unclear"],
                    },
                    "dismissal_reason": {
                        "anyOf": [
                            {
                                "type": "string",
                                "enum": [
                                    "contract_invalidity", "clause_invalidity",
                                    "unproven_breach", "procedural", "other",
                                ],
                            },
                            {"type": "null"},
                        ],
                    },
                    "evidence": _evidence_obj(),
                },
                "required": ["value", "dismissal_reason", "evidence"],
                "additionalProperties": False,
            },
            "legal_reasoning_summary": {"type": "string"},
            "key_quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {"type": "string"},
                        "chunk_id": {"type": "string"},
                    },
                    "required": ["quote", "chunk_id"],
                    "additionalProperties": False,
                },
            },
            "factors": {
                "type": "array",
                "items": _factor_obj(),
            },
        },
        "required": ["decision", "legal_reasoning_summary", "key_quotes", "factors"],
        "additionalProperties": False,
    }


def _penalty_call1_obj():
    """One penalty as returned by Call 1 (no moderation_analysis)."""
    return {
        "type": "object",
        "properties": {
            "penalty_id": {"type": "string"},
            "related_claim_ref": _nullable_string(),
            "breach_type": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "enum": [
                            "late_payment", "non_monetary_performance",
                            "early_termination", "breach_of_confidentiality", "other",
                        ],
                    },
                    "evidence": _evidence_obj(),
                },
                "required": ["value", "evidence"],
                "additionalProperties": False,
            },
            "rate_definition": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "percent_denne", "percent_mesacne", "percent_rocne",
                            "percent_jednorazovo", "percent_z_ceny", "percent_z_dlznej_sumy",
                            "fixna_suma_denne", "fixna_suma_mesacne", "fixna_suma_jednorazovo", "ine",
                        ],
                    },
                    "value_raw": _nullable_string(),
                    "evidence": _evidence_obj(),
                },
                "required": ["type", "value_raw", "evidence"],
                "additionalProperties": False,
            },
            "amounts": _amounts_obj(),
            "associated_interest": {
                "type": "object",
                "properties": {
                    "awarded": {
                        "type": "string",
                        "enum": ["yes", "no", "unclear"],
                    },
                    "rate_value": _nullable_string(),
                    "evidence": _evidence_obj(),
                },
                "required": ["awarded", "rate_value", "evidence"],
                "additionalProperties": False,
            },
        },
        "required": [
            "penalty_id", "related_claim_ref", "breach_type",
            "rate_definition", "amounts", "associated_interest",
        ],
        "additionalProperties": False,
    }


def _penalty_fulldoc_obj():
    """One penalty as returned by full-doc mode (includes moderation_analysis)."""
    # same as call1 but with moderation_analysis added
    base = _penalty_call1_obj()
    base["properties"]["moderation_analysis"] = _moderation_analysis_obj()
    base["required"].append("moderation_analysis")
    return base


# ==========================================
# TOP-LEVEL SCHEMAS FOR EACH CALL TYPE
# ==========================================

def get_call1_schema():
    """Schema for Call 1 response - facts about contract and penalty.
    Does NOT include moderation_analysis (thats Call 2).
    FIX: removed meta from here - meta fields (court, date, case_id, ecli) are
    already extracted by regex in data_cleaner.py and are more reliable there.
    No point asking the LLM to re-extract them and risk getting the date wrong."""
    return {
        "name": "extraction_call1",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "case_context": _case_context_obj(),
                "contractual_penalties": {
                    "type": "array",
                    "items": _penalty_call1_obj(),
                },
            },
            "required": ["case_context", "contractual_penalties"],
            "additionalProperties": False,
        },
    }


def get_call2_schema():
    """Schema for Call 2 response - moderation analysis for each penalty."""
    return {
        "name": "extraction_call2",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "penalties_moderation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "penalty_id": {"type": "string"},
                            "moderation_analysis": _moderation_analysis_obj(),
                        },
                        "required": ["penalty_id", "moderation_analysis"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["penalties_moderation"],
            "additionalProperties": False,
        },
    }


def get_fulldoc_schema():
    """Schema for full-doc mode - everything in one shot.
    FIX: removed meta - same as call1, regex handles it better."""
    return {
        "name": "extraction_fulldoc",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "case_context": _case_context_obj(),
                "contractual_penalties": {
                    "type": "array",
                    "items": _penalty_fulldoc_obj(),
                },
                "quality_control": {
                    "type": "object",
                    "properties": {
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "missing_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["flags", "missing_fields"],
                    "additionalProperties": False,
                },
            },
            "required": ["case_context", "contractual_penalties", "quality_control"],
            "additionalProperties": False,
        },
    }
