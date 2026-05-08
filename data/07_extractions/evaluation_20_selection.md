# 20 Documents Selected for Lawyer Evaluation

## Selection criteria

- 10 long (>50k chars) + 6 medium (20-50k) + 4 short (<20k) — to show RAG vs fulldoc
  difference at different document lengths
- 13 moderated_301 + 5 awarded_full + 1 dismissed + 1 returned
- 13 from golden dataset + 7 new (non-golden) — so it looks like a mix, not just golden
- **All 20 from obchodnoprávne kolégium** (commercial chamber) — fixed after audit
- **4 from Najvyšší súd SR** (all Obdo = obchodnoprávne dovolanie)
- 7 different contract types for variety
- mostly clean extractions (few flags)
- **1 multi-penalty case** (2Cob/69/2020 with 3 penalties) — shows how models handle
  splitting of multiple penalties within a single decision

## Why this distribution

Long documents are where RAG should shine — coverage is 40-80% so the LLM reads only
the relevant chunks. In short documents coverage is 130-200% meaning RAG reads the
ENTIRE document anyway, so there should be no difference vs fulldoc.

Most cases are moderated_301 because thats the core of my thesis — how courts moderate
contractual penalties under §301 ObchZ. But i included a few awarded_full and one
dismissed to show the system handles all decision types.

The audit revealed that 2 originally-selected docs were from non-commercial chambers
(1Cdo/85/2023 = civilnoprávne NS SR, 43CoPv/10/2023 = civilné priemyselnoprávne at KS BB).
Both were swapped for obchodnoprávne equivalents. A third swap replaced a KS Bratislava
long awarded_full with an NS SR Obdo equivalent to bring the NS SR count from 3 to 4.

## The 20 documents

| # | Case | Court | Contract type | Decision | Size | Chars | Source | Why selected |
|---|---|---|---|---|---|---|---|---|
| 1 | 43Cob/30/2023 | KS BB | dielo | moderated_301 | LONG | 81,886 | golden | 2.5% denne, reduced to 50%. Classic moderation. |
| 2 | 14Cob/109/2017 | KS Žilina | uver | moderated_301 | LONG | 105,224 | golden | Court reduced RATE not amount (0.2%→0.05%). Edge case. |
| 3 | 14Cob/21/2019 | KS Žilina | sprostredkovatelska | moderated_301 | LONG | 120,264 | golden | Longest doc. 10k per violation → 500. Good RAG test. |
| 4 | 5Obdo/14/2023 | **NS SR** | kupna | dismissed | LONG | 66,211 | golden | 500k dismissed. Only dismissed case. |
| 5 | 2Cob/69/2020 | KS BA | najom | awarded_full | LONG | 62,827 | golden | **3 penalties** (multi-penalty case). Tests how models split. |
| 6 | 31Cob/104/2020 | KS Trnava | dielo | awarded_full | LONG | 71,413 | golden | Partial withdrawal, not moderation. Nuanced. |
| 7 | 14Cob/202/2019 | KS Žilina | sprostredkovatelska | awarded_full | LONG | 92,784 | golden | 20k confidentiality breach. |
| 8 | 2Obdo/57/2021 | **NS SR** | sprostredkovatelska | awarded_full | LONG | 60,156 | new | NS SR Obdo. Shows RAG null-vs-fulldoc-filled differences. |
| 9 | 43Cob/75/2024 | KS BB | ine | awarded_full | LONG | 104,184 | golden | Very long, lowest coverage (42%). |
| 10 | 42Cob/51/2024 | KS BB | najom | moderated_301 | LONG | 69,697 | new | Only non-golden long moderation case. |
| 11 | 8Cob/258/2014 | KS Trenčín | dielo | moderated_301 | MED | 22,743 | golden | 17k→1k (94% reduction!). Clear §301. |
| 12 | 31Cob/19/2018 | KS Trnava | dodavka_sluzieb | moderated_301 | MED | 38,549 | golden | 10k→5k. Clean extraction, 0 flags. |
| 13 | 1Obdo/66/2018 | **NS SR** | kupna | moderated_301 | MED | 23,036 | golden | 50% kupnej ceny. NS SR precedent. |
| 14 | 5Cob/11/2022 | KS Prešov | dodavka_sluzieb | moderated_301 | MED | 20,662 | new | Medium non-golden moderation. |
| 15 | 26Cob/52/2016 | KS Nitra | dielo | moderated_301 | MED | 22,290 | new | Low coverage (45%) — good RAG test. |
| 16 | 1Obdo/72/2019 | **NS SR** | sprostredkovatelska | moderated_301 | MED | 33,414 | golden | NS SR Obdo moderation. Clean case. |
| 17 | 8Cob/64/2011 | KS Trenčín | najom | returned | SHORT | 9,039 | golden | Only returned case. |
| 18 | 4Cob/2/2017 | KS Prešov | sprostredkovatelska | moderated_301 | SHORT | 14,333 | new | Short moderation, 0 flags. |
| 19 | 26Cob/69/2014 | KS Nitra | dodavka_sluzieb | moderated_301 | SHORT | 5,682 | new | Shortest doc. 186% coverage. |
| 20 | 2Cob/174/2011 | KS Košice | dielo | moderated_301 | SHORT | 12,061 | new | Clean short moderation, 0 flags. Reduced to 50%. |

## Distribution summary

| | Count |
|---|---|
| **Decisions** | |
| moderated_301 | 13 |
| awarded_full | 5 |
| dismissed | 1 |
| returned | 1 |

| **Sizes** | |
| LONG (>50k) | 10 |
| MEDIUM (20-50k) | 6 |
| SHORT (<20k) | 4 |

| **Source** | |
| Golden dataset | 13 |
| Non-golden | 7 |

| **Court** | |
| Najvyšší súd SR (Obdo) | 4 |
| Krajský súd (Cob) | 16 |

| **Contract types** | |
| dielo | 5 |
| sprostredkovatelska | 5 |
| najom | 4 |
| dodavka_sluzieb | 3 |
| kupna | 2 |
| ine | 1 |
| uver | 1 |

## Chamber audit (post-fix)

All 20 documents are now from obchodnoprávne kolégium:
- **NS SR docs (4)**: 5Obdo/14/2023, 1Obdo/66/2018, 1Obdo/72/2019, 2Obdo/57/2021
  (all Obdo prefix = obchodnoprávne dovolanie)
- **KS docs (16)**: all have Cob prefix (obchodný senát)

Previously-excluded docs (from earlier draft):
- 1Cdo/85/2023 — Cdo = civilnoprávne dovolanie (not commercial)
- 43CoPv/10/2023 — CoPv = civilné priemyselnoprávne (trademark case, not commercial)

## What to look for in evaluation results

1. **Long docs RAG vs fulldoc**: Does RAG extract better decisions/factors from
   focused chunks, or does fulldoc find more because it sees everything?
2. **Short docs**: RAG coverage is 130-200%, so both modes read the same text.
   Results should be similar — if they differ, its not about retrieval.
3. **Multi-penalty (2Cob/69/2020)**: Can the model correctly identify and separate
   3 distinct penalties? This is the hardest extraction task and earlier analysis
   showed Qwen RAG tends to merge while Qwen Fulldoc correctly splits.
4. **NS SR cases (4 docs)**: Highest court rulings — precedent-setting decisions
   that should be handled accurately by all models.
5. **Edge cases**: Rate reduction (14Cob/109), partial withdrawal (31Cob/104),
   returned case (8Cob/64) — how well does each model handle non-standard situations?
