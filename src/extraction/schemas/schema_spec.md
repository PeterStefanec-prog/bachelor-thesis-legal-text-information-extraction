```json
{
  "schema_version": "4.0",

  // 1. METADATA A ENTITY ( Regex / NER / Python)
  "meta": {
    "doc_id": "string",
    "decision_metadata": {
      "court_name": "string",
      "case_number": "string",
      "decision_date": "YYYY-MM-DD",
      "ecli": "string|null"
    }
  },


  //  2. KONTEXT SPORU (LLM, parties probably with NER)
  "case_context": {
    // rychly zaver pre pravnika 
    "dispute_summary": {
      "value": "string (max 2-3 vety o predmete sporu) (napr. 'Žaloba o zaplatenie faktúr za stavebné práce, kde sa žalovaný dostal do omeškania kvôli druhotnej platobnej neschopnosti.')",
      "evidence": [{ "quote": "string", "chunk_id": "string" }]
    },
    
    // rychly vysledok pre zoznam
    "verdict_summary": {
       "value": "string (napr. 'Súd nárok na pokutu priznal len čiastočne, zvyšok zamietol pre neprimeranosť.')",
       "evidence": [{ "quote": "string", "chunk_id": "string" }]
    },
    
    "contract_type": {
      "value": "uver|pozicka|najom|dielo|kupna|dodavka_sluzieb|telekom|preprava|mandatna|ine|nezname"
    },

    "parties": {
      "plaintiff": {"value": "string|null", "evidence": []},
      "defendant": {"value": "string|null", "evidence": []},
      "relationship_type": {
        "value": "B2B|B2C|C2C|unknown|unclear"
      }
    },
    
  },


  // 3. ZMLUVNE POKUTY (Jadro prace)
  // Je to POLE [], lebo v jednom rozhodnuti moze byt viac pokut (napr. za 3 faktury)
  "contractual_penalties": [
    {
      "penalty_internal_id": "string (napr. 'pokuta_1')",
      "related_claim_ref": {
        "value": "string (napr. 'Faktúra č. 10/2020' alebo 'Omeškanie za január')",
        "evidence": [{ "quote": "string", "chunk_id": "string" }]
      },

      // A) AKA BOLA POKUTA V ZMLUVE?
      "rate_definition": {
        "type": "percent_denne|percent_mesacne|fixna_suma_mesacne|percent_rocne|percent_jednorazovo|fixna_suma_denne|fixna_suma_jednorazovo|pausal|percent_z_ceny|percent_z_dlznej_sumy|ine",
        "value_raw": "string (napr. '0.05% denne')",
        "evidence": [{ "quote": "string", "chunk_id": "string" }]
      },

      // B) SUMY (Povodna vs. Priznana)
      "amounts": {
        "currency": "EUR|SKK|CZK|unknown",
        "original_claimed": {
          "amount": null,
          "evidence": [{ "quote": "string", "chunk_id": "string" }]
        },
        "final_awarded": {
          "amount": null,
          "evidence": [{ "quote": "string", "chunk_id": "string" }]
        },
        "calculation_logic_summary": {
          "value": "Žalobca počítal 5% týždenne z ceny etapy (15 573,90 EUR). Súd výpočet neuznal pre neurčitosť základu.",
          "evidence": [{ "quote": "string", "chunk_id": "string" }]
        }
      },

      // C) PRISLUSENSTVO (Urok pri pokute)
      "associated_interest": {
        "awarded": "yes|no",
        "applies_to": "penalty|principal|both|unclear",
        "rate_value": "string (napr. '9,0 % ročne' alebo 'zákonný úrok z omeškania')",
        "evidence": [{ "quote": "string", "chunk_id": "string" }]
      },

      
      // D) PRECO SUD ROZHODOL TAKTO? (Analyza moderacie)
      "moderation_analysis": {
        "decision_on_penalty": {
            "value": "awarded_full (priznaná v plnej výške)|awarded_reduced (znížená)|dismissed (zamietnutá)|unclear",
            "evidence": [{ "quote": "string", "chunk_id": "string" }]
        },
        "moderation_applied": {
            // 'not_applicable' = súd zamietol nárok z iného dôvodu (neplatnosť, premlčanie) a moderáciu neriešil
            "value": "yes (súd znížil)|no (neznížil)|nepriznana_uplne (zamietol)|not_applicable (zamietol z iného dôvodu)|unclear",
            "evidence": [
            { "quote": "string", "chunk_id": "string" }
          ]
        },
        // new - Textove vysvetlenie pre pravnika
        "legal_reasoning_summary": {
            "value": "string (napr. 'Súd uviedol, že sadzba 1% denne je v rozpore s dobrými mravmi, pretože výrazne prevyšuje bežné úrokové miery bánk a pre žalovaného by bola likvidačná.')",
            "evidence": [{ "quote": "string", "chunk_id": "string" }]
        },

        // Faktory (Enums pre jednoduchu analyzu v grafoch)
        "factors": [
          {
            "label": "dobre_mravy",
            "sentiment": "positive (súd súhlasí s pokutou)|negative (rozpor s mravmi)|neutral",
            "evidence": [{ "quote": "string", "chunk_id": "string" }]
          },
          {
            "label": "zabezpecovacia_funkcia",
            "sentiment": "positive (funkcia zachovaná)|negative (funkcia popretá)|neutral",
            "evidence": [{ "quote": "string", "chunk_id": "string" }]
          },
          {
            "label": "vyska_skody",
            "sentiment": "positive (škoda vznikla)|negative (škoda žiadna/malá)|neutral",
            "evidence": [{ "quote": "string", "chunk_id": "string" }]
          },
          {
            "label": "pomer_k_istine",
            "sentiment": "positive (pomer OK)|negative (neprimerane vysoká)|neutral",
            "evidence": [{ "quote": "string", "chunk_id": "string" }]
          }
        ]
      }
    }
  ],

  // 4. KONTROLA (Generuje Python skript, nie LLM)
  "quality_control": {
    "flags": ["string"],
    "missing_fields": ["string"]
  }
}
```