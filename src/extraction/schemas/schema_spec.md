# Extraction Schema v5.0 – Specification

> **File:** `src/extraction/schemas/extraction_schema.json`

---

## What the schema extracts

The goal is to extract structured info about contractual penalties (zmluvna pokuta)
from Slovak court decisions, specifically to help lawyers calibrate penalty clauses
in new contracts. The key question is always: was the penalty moderated, why,
and what rate/amount was involved.

---

## Structure overview

```
meta                          - doc_id, court name, case number, date, ecli
                                (flat - no nesting)
case_context
  dispute_summary             - 2-3 sentence summary of what the dispute was about
                                (o com bol spor a co zalobca pozadoval)
  verdict_summary             - 1-2 sentences on overall outcome
                                (celkovy vysledok konania)
  contract_type               - enum (typ zmluvy)
  relationship_type           - B2B / B2C / C2C / unknown

contractual_penalties[]       - list because one decision can have multiple penalties
                                (zoznam lebo jedno rozhodnutie moze riesit viac pokut)
  penalty_id                  - pokuta_1, pokuta_2 ...
  related_claim_ref           - which invoice/claim this relates to (null if just one)
  breach_type                 - type of breach + evidence
  rate_definition             - type + raw value + evidence
  amounts
    currency                  - EUR / SKK / CZK / unknown
    secured_principal         - base amount the penalty was calculated from + evidence
    original_claimed          - what plaintiff asked for + evidence
    final_awarded             - what court actually gave + evidence (nullable)
  associated_interest         - just enough to not confuse interest with penalty rate
  moderation_analysis
    decision                  - single merged field: outcome + dismissal_reason + evidence
    legal_reasoning_summary   - plain string: how numbers were calculated + why court moderated
    key_quotes[]              - 1-2 anchor quotes from the reasoning
    factors[]                 - 7 specific legal arguments + sentiment + evidence each

quality_control
  flags, missing_fields
```

---

## Complete enum reference

All allowed values for each enum field, as defined in the JSON schema.

### case_context.contract_type
| value | meaning |
|-------|---------|
| `uver` | zmluva o uvere (credit agreement) |
| `pozicka` | zmluva o pozicke (loan) |
| `najom` | najomna zmluva (lease/rental) |
| `dielo` | zmluva o dielo (contract for work) |
| `kupna` | kupna zmluva (purchase agreement) |
| `dodavka_sluzieb` | zmluva o dodavke sluzieb (service delivery) |
| `sprostredkovatelska` | sprostredkovatelska zmluva (brokerage/mediation) |
| `telekom` | telekomunikacna zmluva (telecom) |
| `preprava` | prepravna zmluva (transport/shipping) |
| `mandatna` | mandatna zmluva (mandate contract) |
| `ine` | iny typ zmluvy (other) |
| `nezname` | typ sa neda urcit z textu (cannot determine) |

### case_context.relationship_type
`B2B` | `B2C` | `C2C` | `unknown`

### breach_type.value
| value | meaning |
|-------|---------|
| `late_payment` | oneskorena platba (omeskanie s platbou) |
| `non_monetary_performance` | neplnenie nepenaznej povinnosti |
| `early_termination` | predcasne ukoncenie zmluvy |
| `breach_of_confidentiality` | porusenie mlcanlivosti |
| `other` | ine porusenie |

### rate_definition.type
| value | meaning |
|-------|---------|
| `percent_denne` | % denne z nejakej sumy |
| `percent_mesacne` | % mesacne |
| `percent_rocne` | % rocne |
| `percent_jednorazovo` | jednorazove % (napr. 10% z ceny diela) |
| `percent_z_ceny` | % z celkovej ceny |
| `percent_z_dlznej_sumy` | % z dlznej sumy |
| `fixna_suma_denne` | fixna suma za kazdy den (napr. 100 EUR/den) |
| `fixna_suma_mesacne` | fixna suma za kazdy mesiac |
| `fixna_suma_jednorazovo` | jednorazova fixna suma |
| `ine` | iny sposob vypoctu |

### amounts.currency
`EUR` | `SKK` | `CZK` | `unknown`

### associated_interest.awarded
`yes` | `no` | `unclear`

### moderation_analysis.decision.value
| value | meaning |
|-------|---------|
| `awarded_full` | sud priznal pokutu v plnej vyske |
| `moderated_301` | sud znizil pokutu podla §301 ObchZ |
| `dismissed` | sud zamietol narok na pokutu |
| `returned` | vec vratena na dalsie konanie |
| `unclear` | nie je jasne z textu |

### moderation_analysis.decision.dismissal_reason
Only used when decision.value = `dismissed`. Otherwise null.

| value | meaning |
|-------|---------|
| `contract_invalidity` | neplatnost zmluvy |
| `clause_invalidity` | neplatnost dojednania o pokute |
| `unproven_breach` | nepreukázanie porusenia povinnosti |
| `procedural` | procesny dovod (napr. premlcanie, nedostatok aktivnej legitimacie) |
| `other` | iny dovod zamietnutia |
| `null` | not dismissed, or reason unknown |

### factors[].label
7 fixed factors, always present in the array:
| label | meaning |
|-------|---------|
| `dobre_mravy` | rozpor s dobrymi mravmi / poctivym obchodnym stykom |
| `zabezpecovacia_funkcia` | hodnota a vyznam zabezpecovanej povinnosti |
| `vyska_skody` | pomer pokuty k skutocnej skode |
| `pomer_k_istine` | pomer pokuty k istine / celkovej cene |
| `spravanie_dlznika` | miera zavinenia, spravanie dlznika |
| `kumulacia_s_urokom` | kumulacia pokuty s urokom z omeskania |
| `spravanie_veritela` | pasivita veritela - ci veritel necakoval namiesto toho aby vymahal plnenie |

### factors[].sentiment
`positive` | `negative` | `neutral` | `not_mentioned`

- positive = factor supports the penalty (sud suhlasi s pokutou)
- negative = factor argues against the penalty (dovod na znizenie)
- neutral = court mentioned it but took no clear stance
- not_mentioned = court did not discuss this factor at all

---

## Where evidence is and where it is not

| field | evidence? | reason |
|-------|-----------|--------|
| dispute_summary | no | synthesis from multiple chunks |
| verdict_summary | no | synthesis from multiple chunks |
| contract_type | no | inferred from overall context |
| relationship_type | no | inferred from overall context |
| breach_type | yes | verifiable, affects case comparability |
| rate_definition | yes | specific number, must be checkable |
| secured_principal | yes (nullable) | specific number, null if not explicit in text |
| original_claimed | yes | specific number |
| final_awarded | yes (nullable) | specific number, nullable for dismissed cases |
| associated_interest | yes (nullable) | rate value must be verifiable |
| decision | yes | legal conclusion, must be anchored in text |
| legal_reasoning_summary | no (key_quotes sibling) | synthesis, anchored by key_quotes next to it |
| key_quotes | yes (1-2 quotes) | anchor quotes for auditability |
| factors (each) | yes (nullable) | interpretive claim about court's argument |

---

## Changes from v4.1 to v4.3

### removed fields

**`verdict_summary` evidence**
verdict_summary is a model-generated synthesis from multiple paragraphs. there is no single sentence
that "proves" it. added evidence would just be a random cherry-picked quote, which is misleading.
kept the field itself (as a plain string) because it is still useful - sometimes the court never
even reaches the penalty question (zamietol pre neplatnost zmluvy) and without verdict_summary
you wouldnt know that from decision_on_penalty alone.

**`dispute_summary` evidence**
same reason - it is a synthesis not an extraction. plain string, no evidence.

**`parties.plaintiff` and `parties.defendant` names**
Slovak decisions are anonymized (O. R., XX. B. XXXX). model would hallucinate real names.
names are useless for penalty calibration anyway. kept `relationship_type` (B2B/B2C) because
that one actually affects how strictly courts apply §301.

**`breach_duration` (trvanie porusenia)**
almost never stated explicitly in decisions. model would derive it from the calculation context
and write it as a fact - that is a hallucination. when duration is relevant it shows up
naturally in `legal_reasoning_summary`.

**`calculation_logic_summary`**
merged into `legal_reasoning_summary`. both fields were pulling quotes from the same paragraphs -
court describes calculation and immediately explains why it accepted or rejected it. two separate
evidence lists for overlapping content confused the model and wasted tokens.

**`associated_interest.applies_to`**
hard to determine reliably. field exists only to prevent confusing 9% p.a. interest with
0.05% daily penalty rate. for that you only need `awarded` + `rate_value`.

### added / changed (v4.1 to v4.3)

**`breach_type` now has evidence**
in v4.1 it had no evidence. breach_type determines comparability of cases
(late_payment vs non_monetary_performance behave very differently in courts). should be verifiable.

**`factors` got `not_mentioned` as 4th sentiment option**
before: positive / negative / neutral. now: positive / negative / neutral / not_mentioned.
this matters for thesis statistics. if the court never mentioned a factor, it should not be
counted as `neutral`. otherwise chapter 9 analysis would show artificially high neutral counts.

**`legal_reasoning_summary` uses `key_quotes` instead of `evidence`**
it is a synthesis from multiple paragraphs, so a single evidence list would be misleading.
replaced with `key_quotes` (1-2 anchor quotes) which signals that this is not a single
extracted fact but a summarized interpretation.

**all examples in Slovak**
previous version had English examples. the model reads Slovak legal texts and should produce
Slovak output. English examples would cause the model to mix languages in output.

---

## Changes from v4.3 to v5.0

### structural changes

**`meta` flattened (zrusene vnorenie)**
removed the `decision_metadata` nesting. `court_name`, `case_number`, `decision_date`, `ecli`
are now directly under `meta`. the extra level was unnecessary.

**`penalty_internal_id` renamed to `penalty_id` (premenovanie)**
shorter, cleaner. same purpose (pokuta_1, pokuta_2 ...).

**`decision_on_penalty` + `moderation_applied` merged into `decision` (zlucenie)**
this was the biggest change. before we had two separate fields - one for the outcome
(awarded_full/awarded_reduced/dismissed) and one for whether moderation was applied
(yes/no/not_applicable). in practice these overlap: if the court moderated under §301 ObchZ,
the outcome is always "awarded_reduced". having both caused confusion - sometimes the model
would say decision_on_penalty=awarded_reduced but moderation_applied=no which doesnt make sense.

new `decision.value` options: awarded_full, moderated_301, dismissed, returned, unclear.

**`decision.dismissal_reason` added (nove pole)**
when decision=dismissed, this says why. matters because zamietnutie pre neplatnost zmluvy vs
nepreukázanie porusenia su completely different situations for lawyers.

**`legal_reasoning_summary` simplified to plain string (zjednodusenie)**
was an object with `poznamka`, `value`, and `key_quotes` inside. now just a plain string.
the poznamka was an instruction for the LLM, not data.

**`key_quotes` moved to top level of moderation_analysis (presunutie)**
was nested inside legal_reasoning_summary. now a direct child of moderation_analysis,
same level as decision and factors. cleaner structure, easier to parse programmatically.

### value changes

**`contract_type` - added options, removed `unclear`**
added: `sprostredkovatelska` (brokerage), `telekom`, `preprava`, `mandatna`, `pozicka`, `kupna`,
`dodavka_sluzieb`. removed `unclear`, using `nezname` instead (more consistent).

**`rate_definition.type` - removed `pausal`**
"pausal" (flat fee) was never used in our court decisions. all penalties are either
percentage-based or fixed amount per time unit. removed to reduce options.

**`factors` - expanded to 7 fixed factors**
I added `kumulacia_s_urokom` because courts sometimes consider whether the creditor is
already getting urok z omeskania on top of zmluvna pokuta. I also added
`spravanie_veritela`, because some decisions consider whether the creditor waited
passively and let the penalty grow.

**`factors[].sentiment` - removed inline explanations**
was: `"positive (sud suhlasi s pokutou)|negative (rozpor s dobrymi mravmi)|..."`
now: `"positive|negative|neutral|not_mentioned"`
the parenthetical explanations were LLM instructions, not part of the schema.

### minor cleanup

- `associated_interest.poznamka` removed (was an LLM instruction, not data)
- `secured_principal.value` inline instruction removed (goes in prompt, not schema)
- `final_awarded.evidence` now nullable (if court dismissed, there is no "awarded" quote)
