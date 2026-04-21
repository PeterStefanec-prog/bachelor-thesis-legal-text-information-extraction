# Legal Precedent Search System — Design Document

## 1. Motivacia a pouzitie

### Problem
Pravnik pisuci zmluvu o dielo za 50 000 EUR potrebuje vediet: "Aku zmluvnu pokutu
za omeskanie mozem nastavit, aby ju sud neznizil?" Na zodpovedanie tejto otazky
musi nastudovat desiatky sudnych rozhodnuti — rucne hladanie trva hodiny.

Z druhej strany, pravnik zastupujuci firmu kde pokuta uz bola uplatnena chce vediet:
"Ake faktory sud najviac berie do uvahy pri moderacii podla §301? Existuju precedensy
kde sud znizil pokutu pri podobnej zmluve?"

### Riesenie
System ktory kombinuje strukturovane udaje (z extraction pipeline) so semantickym
vyhladavanim. Na rozdiel od naivneho multi-document RAG, ktory len hlada podobne
chunky napriec dokumentmi, nas system:

1. Filtruje pokuty podla strukturovanych atributov (typ zmluvy, typ porusenia) —
   toto je mozne LEN vdaka extraction pipeline
2. Rankuje filtrovane pokuty podla multi-dimenzionalnej podobnosti
3. Spocita statistiky napriec relevantnymi pripadmi (meritorny pomer, factor lift)
4. LLM syntetizuje odpoved s citaciami na konkretne precedensy

### Definicia pristupu
**Structured-Augmented Multi-Document Retrieval with Analytics** — kombinacia troch
paradigiem:

- **Multi-Meta-RAG** (Zareieh et al., 2024): filtrovanie podla LLM-extrahovanych
  metadat pred semantickym retrievalom
- **RAS — Retrieval And Structuring** (KDD 2025 survey): extraction ako "structuring
  step" pred retrievalom
- **Analytical RAG**: agregacia statistik napriec dokumentmi, co klasicky RAG nedokaze

Pre thesis:
> "System na vyhladavanie pravnych precedensov postaveny na principe
> Structured-Augmented Retrieval (Multi-Meta-RAG), kde LLM-extrahovane
> strukturovane data sluzia ako prvy stupen filtrovania a agregacie,
> nasledovany rankingom v prefiltrovanch pripadoch a LLM syntezou
> odpovede s citaciami."


## 2. Dostupne data

### Extraction JSONy (data/07_extractions/gpt-4o_rag/)
176 dokumentov s kompletne extrahovanymi strukturovanymi udajmi. Pre kazdy dokument:

```
result.case_context:
  contract_type    — 12 enumov (dielo, najom, uver, kupna, ...)
  relationship_type — B2B | B2C | C2C | unknown

result.contractual_penalties[] (1-5 per dokument):
  breach_type.value           — 5 enumov
  rate_definition.type        — 10 enumov (percent_denne, fixna_suma_jednorazovo, ...)
  rate_definition.value_raw   — "0,05% denne z dlznej sumy"
  amounts.currency            — EUR | SKK | CZK | unknown
  amounts.secured_principal   — cislo alebo null
  amounts.original_claimed    — cislo alebo null
  amounts.final_awarded       — cislo alebo null
  associated_interest.awarded — yes | no | unclear
  moderation_analysis:
    decision.value              — awarded_full | moderated_301 | dismissed | returned
    legal_reasoning_summary     — 2-5 viet
    key_quotes[]                — [{quote, chunk_id}]
    factors[] (vzdy 7):
      label     — dobre_mravy | zabezpecovacia_funkcia | vyska_skody |
                  pomer_k_istine | spravanie_dlznika | kumulacia_s_urokom |
                  spravanie_veritela
      sentiment — positive | negative | neutral | not_mentioned

result.quality_control.flags — ANONYMIZED_AMOUNT, MATH_ERROR, EVIDENCE_NOT_FOUND, ...
result.meta — doc_id, court_name, case_number, decision_date, ecli
```

### Penalty-level index (data/10_penalty_index/)
Build-script `build_penalty_index.py` pretvori 176 JSONov na:
- `penalty_index.jsonl` — 1 riadok = 1 pokuta (~202 zaznamov)
- `embeddings.npy` — numpy matica [N, 1536] penalty card embeddingov

**Preco penalty-level a nie document-level:** 8.5% dokumentov ma viac pokut s roznymi
atributmi. Dokument-level filtrovanie by vytvaralo false matches (napr. filter na
`late_payment AND awarded_full` by matchol dokument kde pokuta_1 je
`late_payment+moderated` a pokuta_2 je `non_monetary+awarded` — kombinacia ktora
v dokumente REALNE NEEXISTUJE).


## 3. Architektura — 5 stupnov

```
QUERY: "Robim zmluvu o dielo za 50k EUR. Aku pokutu za omeskanie dat?"
  |
  v
STAGE 1: Query Understanding ─────────────────────────────────────────
  LLM (Gemini 2.5 Flash) so structured output schema.
  Vstup: prirodzeny text query
  Vystup: {
    contract_type: "dielo",
    breach_type: "late_payment",
    decision_interest: null,
    factor_interest: [],
    amount_hint: 50000,
    intent: "safe_rate",
    semantic_query: "zmluvna pokuta za omeskanie zmluva o dielo"
  }
  |
  v
STAGE 2: Penalty-Level Filtering ─────────────────────────────────────
  In-memory Python filtering nad 202 penalty records.
  Hard filtre: contract_type (s near-miss mapou), breach_type.
  Vstup: 202 pokut
  Vystup: ~49 pokut kde (contract_type IN {dielo, dodavka_sluzieb}) AND
          (breach_type = late_payment)
  |
  v
STAGE 3: Ranking ─────────────────────────────────────────────────────
  4-dimenzionalne skorovanie:
  - Embedding similarity (40%) — cosine(query_emb, penalty_card_emb)
  - Amount proximity (25%) — ratio blizkej/vzdialenej sumy
  - Bonus attributes (20%) — decision_interest + factor_interest match
  - Authority (15%) — NS SR > KS > OS
  Vstup: ~49 pokut
  Vystup: top 15 zoradenych podla relevantnosti (top 5 ide do Stage 5)
  |
  +-------------------------------+
  |                               |
  v                               v
STAGE 4: Analytics              [TOP 5 penalties]
  n<5: len precedensy             Metadata, sumy, rozhodnutie, 7 faktorov,
  n>=5: merit ratio + factors     reasoning_summary, key_quotes.

                                  Pri safe_rate intente: zabezpec ze
  Pocita len 2 veci:              aspon 1 moderated case je v top 5
  - meritorny pomer (moderacia    (injection v pipeline.py).
    medzi awarded + moderated)
  - factor LIFT per faktor
    (s raw counts)
  |                               |
  +-------------------------------+
  |
  v
STAGE 5: LLM Synthesis ───────────────────────────────────────────────
  Jeden LLM call (Gemini 2.5 Flash) s intent-aware system prompt:
  - safe_rate: odporuc sadzbu + varuj pred rizikami
  - defense_args: najsilnejsie argumenty podla LIFT
  - general_precedent: zhrnutie + citacie
  
  Pri n<5: prida sa low-n guard instrukcia (zakazuje LLM formulovat
  vseobecne zavery, opisuje len konkretne pripady).
  
  LLM dostane: query + formatovane analytics + top 5 precedent cards
  LLM NEPOCITA ziadne cisla — len interpretuje Python-computed data.
  |
  v
VYSTUP:
  1. STATISTIKY — Python-computed, deterministicke (plain language)
  2. LLM ODPOVED — synteza s citaciami [case_number]
  3. PRECEDENSY — top 5 s metadatami a skorom
```


## 4. Detailny navrh kazdeho stage-u

### Stage 1: Query Understanding (query_analyzer.py, ~170 riadkov)

Metoda: Gemini 2.5 Flash so structured output (`response_schema`).

**Preco LLM a nie regex:** Slovencina ma 6 padov × 2 cisla —
"zmluva/zmluvy/zmluvou/zmluve o dielo" by vyzadovalo desiatky patternov. Ani tak
by regex nezachytil "IT zakazka" = zmluva o dielo. Gemini Flash to zvladne za
~$0.003 a jeden code path je jednoduchsi ako dva.

Schema enforcuje (`schemas.py:get_query_understanding_schema()`):
- `contract_type`: nullable enum (12 hodnot)
- `breach_type`: nullable enum (5 hodnot)
- `decision_interest`: nullable enum (5 hodnot)
- `factor_interest`: array of factor labels (0-7)
- `amount_hint`: nullable number
- `intent`: enum (safe_rate | defense_args | general_precedent)
- `semantic_query`: string (preformulovany query pre embedding)

Retry logika: 503/UNAVAILABLE → wait 10s/20s/30s/40s. Pri uplnom zlyhani fallback
na prazdny intent (pipeline pokracuje bez filtrov).

### Stage 2: Penalty-Level Filtering (penalty_index.py, ~150 riadkov)

Trieda `PenaltyIndex` — `__init__` nacita JSONL + numpy embeddingy (raz pri starte),
`filter(intent)` aplikuje filtre.

Hard filtruje len na:
- `contract_type` (s `SIMILAR_CONTRACT_TYPES` mapou pre near-miss: "dielo" matchne
  aj "dodavka_sluzieb", "uver" matchne "pozicka")
- `breach_type` (exact match)

`decision_interest` a `factor_interest` NIE SU hard filtre — idu do Stage 3
ako soft ranking signaly. Dovod: hard-filter na `decision=awarded_full` by dal
analytics "100% potvrdenych" (trivialne pravda, filtroval som na to). A filter
na konkretne faktory by znemoznil factor lift (treba aj moderated aj awarded
na porovnanie).

**Preco in-memory a nie databaza:** 202 zaznamov (v buducnosti ~1500 pri 1000
dokumentoch) je maly objem. Python dict filtering je rychlejsi a jednoduchsi ako
SQLite. Od ~10000 zaznamov by SQLite davalo zmysel.

### Stage 3: Ranking (ranker.py, ~244 riadkov)

4-dimenzionalne skorovanie s vahami:

**A) Embedding similarity (40%):**
- Query embedding: OpenAI `text-embedding-3-small` na `semantic_query`
- Cosine similarity so vsetkymi filtered penalty card embeddings naraz
- numpy `np.dot()` — vektorizovane, ~1ms pre 49 pokut

**B) Amount proximity (25%):**
- Ratio `min(query_amount, penalty_amount) / max(...)`
- 50k vs 48k → 0.96, 50k vs 500k → 0.10
- Prednostne `secured_principal`, fallback `original_claimed`
- Ak chyba data: neutralne 0.5 (nepenalizujem)

**C) Bonus attributes (20%):**
- +2.0 ak decision match (ak decision_interest je specifikovany)
- +1.0 per matching factor (ak factor_interest je specifikovany)
- Normalizovane na [0, 1]
- Neutral 0.5 ak nic nespecifikovane

**D) Authority (15%):**
- NS SR (level 3) = 1.0, KS (level 2) = 0.67, OS (level 1) = 0.33
- Jednoduche one-line skore: `auth_level / 3.0`

Finalne skore: `0.4×emb + 0.25×amt + 0.2×bonus + 0.15×auth`

Output: zoradeny zoznam `(record, score)` tuple, sortovany podla skore desc.

### Stage 4: Analytics (analytics.py, ~380 riadkov)

Pure Python agregacia. Ziadne ML, ziadne API calls.

Analytika pocita len 2 veci, obe obhajitelne pred pravnikom:

**1. Meritorny pomer** — moderacia len medzi awarded + moderated pripadmi.
Dismissed/returned pripady zlyhali z formalnych dovodov (neplatnost zmluvy,
procesne chyby) — nie kvoli vyske sadzby. Ak su 49 pokut rozdelenych ako 15
awarded / 9 moderated / 11 dismissed / 14 returned, "miera moderacie 18.4% (9/49)"
je zavadzajuce. Realna miera moderacie medzi meritornymi rozhodnutiami je
9/(15+9) = 37.5%.

**2. Factor LIFT** — ktory zo 7 faktorov najsilnejsie predpoveda moderaciu.

```
lift = P(factor negative | moderated) / P(factor negative | awarded)
```

Napriklad pomer_k_istine: 68% moderated vs 3% awarded → lift 26.4x = silny
prediktor. Faktor ktory sa spomina v 83% moderated AJ 70% awarded ma lift 1.2x —
spomina sa vsade, nie je prediktivny.

Edge case: ak faktor je negativny v moderated ale NIKDY v awarded, lift je
matematicky nekonecny ("only_in_moderated"). Zobrazujem ako "inf" s raw counts
(napr. "4/34 vs 0/78") — najsilnejsi signal.

**Raw counts namiesto percent:** Zobrazujem "8/9 vs 0/15" nie "89% vs 0%". Pri
malej vzorke pravnik vidi ze "0/15" je maly sample; percenta to skryvaju.

**Sample size threshold:** `MIN_SAMPLE_SIZE = 5`. Pri n<5 analytika nevracia
ziadne statistiky, len note ze je malo dat.

**Dataset bias caveat:** Vzdy zobrazena poznamka ze korpus nie je reprezentativny
(zbieran cez keywords "zmluvna pokuta" + §301).

**Dva formatery pre rozne audience:**

- `format_for_lawyer()` — plain language, ziadny jargon ("lift", "inf"). Zobrazuje
  len frekvencie z moderated cases (napr. "vyska skutocnej skody: 8 z 9 pripadov").
  Toto ide na displej pravnikovi.

- `format_for_llm()` — full detail s lift + raw counts pre oba groups. LLM to
  pouziva ako grounding aby pisal presne vety (napr. "pomer k istine bol problem
  v 8 z 9 znizenych pripadov, zatial co v potvrdenych len v 1 z 15").

### Stage 5: LLM Synthesis (synthesizer.py, ~330 riadkov)

Metoda: Jeden Gemini 2.5 Flash call s temperature 0.3.

**Intent-aware system prompts:**
- **safe_rate**: "Odporuc sadzbu s nizsim rizikom. Upozorni na rizika — aj
  standardna sadzba moze byt znizena ak je kumulacia s urokom alebo suma
  narasta neprimerane."
- **defense_args**: "Zoznam najsilnejsich argumentov podla LIFT dat. Pre kazdy
  cituj precedens."
- **general_precedent**: "Zhrn zistenia, cituj 2-3 rozhodnutia."

**Low-n guard (pri n<5):** Pridava sa striktna instrukcia do system promptu:
zakazuje LLM formulovat vseobecne zavery, davat odporucania alebo extrapolovat
z malej vzorky. Namiesto toho LLM opisuje kazdy pripad jednotlivo a explicitne
uvedie ze je malo dat. Bez tejto instrukcie LLM pri n=1 alebo n=3 pisal 4 odseky
"analyzy" ako keby to bola statisticky vyznamna vzorka.

**Safe_rate moderated injection:** Pri `intent=safe_rate` ranker prirodzene
promuje awarded cases (embedding "bezpecna sadzba" ma vyssie similarity
s potvrdenymi). Bez intervencie by LLM videl top 5 same potvrdene a nevaroval
by pred rizikami. V pipeline.py: ak top 5 neobsahuje ziadny moderated case,
najdem najlepsi z ranked a vlozim ho (nahradim posledny slot). LLM tak vidi aj
pripad kde pokuta bola znizena.

**LLM dostane:**
1. Povodny query
2. Formatovane analytics (format_for_llm — full detail)
3. Top 5 precedent cards: metadata + sumy + decision + factors +
   legal_reasoning_summary + key_quotes

**LLM nepocita cisla** — len interpretuje Python-computed udaje. Vsetky statistiky
su pripravene, LLM robi len natural-language formulaciu s citaciami `[case_number]`.

**Retry logika:** 503/UNAVAILABLE → wait 10s/20s/30s/40s (rovnake ako Stage 1).

Rychlost: ~4s per query.

### Orchestrator (pipeline.py, ~200 riadkov)

Trieda `PrecedentSearchPipeline`. `__init__()` nacita PenaltyIndex. `search(query)`
vykona stages 1-5 a vrati kompletny output.

Error handling:
- Stage 1 failure → prazdny intent, pipeline pokracuje bez filtrov
- Stage 2 nic nenamatchuje → fallback na vsetky zaznamy
- Stage 5 overload → retry s exponencialnym backoff

Meria cas kazdeho stage — pre debug a pre demo (pravnik vidi "celkovo 6s:
query 1s, filter 0ms, rank 1s, analytics 5ms, synthesis 4s").


## 5. Vystup systemu

Pre kazdu query vracia 3 casti:

### Cast 1: Statistiky (Python-computed, bez LLM)
```
ZHRNUTIE (49 relevantnych pripadov)
============================================================

V 37.5% meritornych pripadov sud znizil pokutu (9 z 24).

Najcastejsie dovody znizeni (v 9 znizenych pripadoch):
  - vyska skutocnej skody.............. 8 z 9 pripadov
  - pomer pokuty k istine.............. 8 z 9 pripadov
  - zabezpecovacia funkcia pokuty...... 5 z 9 pripadov
  - dobre mravy........................ 4 z 9 pripadov
  - kumulacia s urokom z omeskania..... 3 z 9 pripadov
  - spravanie dlznika.................. 2 z 9 pripadov
  - spravanie veritela................. 2 z 9 pripadov

Poznamka: Statistiky su zalozene na korpuse... Nezastupuju vsetky slovenske sudy.
```

### Cast 2: LLM odpoved (Stage 5)
```
Na zaklade analyzovaneho korpusu 49 relevantnych pripadov je pri nastavení
sadzby zmluvnej pokuty pre zmluvu o dielo s hodnotou 50 tisic EUR moznost
odporucit sadzbu v rozsahu 0,05 az 0,1% denne z dlznej sumy. V precedensoch
[4Obdo/96/2021] a [4Cob/216/2015] sud potvrdil takuto sadzbu ako primeranu.

Avsak je potrebne upozornit na rizika — v pripade [4Cob/95/2012] bola
pokuta znizena napriek standardnej sadzbe, pretoze celkova suma narastla
neprimerane k okolnostiam...
```

### Cast 3: Precedensy (top 5 zo Stage 3)
```
  [1] [NS SR] Najvyssi sud 4Obdo/96/2021 (30.09.2022)
      dielo | late_payment | 25,- eur za kazdy den omeskania
      POTVRDENA (score: 0.722)
  [2] Krajsky sud Bratislava 4Cob/216/2015 (03.11.2016)
      dielo | late_payment | 0,05 % denne zo sumy 79.107,11 eur
      POTVRDENA (score: 0.705)
  ...
  [5] Krajsky sud Bratislava 4Cob/95/2012 (15.06.2013)
      dielo | late_payment | X,X% denne z dlznej sumy
      ZNIZENA (s301) (score: 0.569)
```


## 6. Porovnanie s naivnym multi-doc RAG

|                        | Naivny multi-doc RAG          | Nas system                           |
|------------------------|-------------------------------|--------------------------------------|
| Query                  | embedding celeho textu        | rozlozene na filtre + semantiku      |
| Priestor hladania      | 5300+ chunkov naraz           | 202 penalty records → ~49 filtered   |
| Statistiky             | NEMOZNE                       | meritorny pomer, factor LIFT         |
| Vysvetlitelnost        | "score 0.82"                  | "matchol: same type + similar amount"|
| Vystup                 | chunky + LLM odpoved          | statistiky + odpoved + precedensy    |
| Rychlost               | pomaly (global search)        | rychly (prefiltrovaný priestor)      |
| Presnost               | shumove chunky v top          | len relevantne typy zmluv            |
| Zavislost na extraction| ZIADNA                        | UPLNA — toto je klucovy argument     |

Posledny riadok je najdolezitejsi: cely system EXISTUJE len preto, ze mame
kvalitne extrahovane strukturovane data. Bez extraction pipeline by sme mohli
robit len naivny RAG.


## 7. Subory a rozsah

| Subor                           | Riadky | Popis                              |
|---------------------------------|--------|------------------------------------|
| src/multi_doc/schemas.py        | ~174   | Enum values + JSON schema + prompt |
| src/multi_doc/build_penalty_index.py | ~452 | Preprocessing (one-time)         |
| src/multi_doc/query_analyzer.py | ~171   | Stage 1 — LLM query understanding  |
| src/multi_doc/penalty_index.py  | ~154   | Stage 2 — filtering                |
| src/multi_doc/ranker.py         | ~244   | Stage 3 — 4-dim ranking            |
| src/multi_doc/analytics.py      | ~379   | Stage 4 — merit ratio + factor lift|
| src/multi_doc/synthesizer.py    | ~333   | Stage 5 — LLM synthesis            |
| src/multi_doc/pipeline.py       | ~199   | Orchestrator                       |
| src/multi_doc/demo.py           | ~189   | Demo script (5 queries)            |
| **CELKOM**                      | **~2295** |                                 |


## 8. Referencie

1. Zareieh, S., Yein, B.K., & Kostiuk, Y. (2024).
   Multi-Meta-RAG: Improving RAG for Multi-Hop Queries using Database Filtering
   with LLM-Extracted Metadata.
   arXiv:2406.13213 — NAJPODOBNEJSI nasmu pristupu

2. KDD 2025 Survey.
   A Survey on Retrieval And Structuring Augmented Generation with Large
   Language Models.
   arXiv:2509.10697 — definuje paradigmu "structuring before retrieval"

3. Sarthi, P. et al. (2024).
   RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
   ICLR 2024. arXiv:2401.18059 — hierarchicky retrieval

4. Metadata-Driven RAG for Financial Question Answering. (2024).
   arXiv:2510.24402 — rovnaky princip v inej domene (finance)

5. Graph Retrieval-Augmented Generation: A Survey.
   ACM Transactions on Information Systems.
   DOI:10.1145/3777378 — knowledge graph + RAG

6. Towards Comprehensive Legal Document Analysis: A Multi-Round RAG Approach.
   ICMR 2025. DOI:10.1145/3731715.3733451 — legal domain RAG

7. LRAGE: Legal Retrieval Augmented Generation Evaluation Tool. (2025).
   arXiv:2504.01840 — evaluacia legal RAG systemov


## 9. Klucove designove rozhodnutia

### Preco penalty-level a nie document-level index
8.5% dokumentov ma viac pokut s roznymi atributmi. Document-level filtrovanie by
vytvaralo false cross-penalty kombinacie. Overene na realnych datach (napr.
KS_Trencin_8Cob_38_2012 ma pokuta_1=dismissed+non_monetary a
pokuta_2=moderated+early_termination — ziadna kombinacia dismissed+early_termination
v dokumente realne neexistuje).

### Preco LLM query understanding a nie regex
Slovencina ma 6 padov × 2 cisla = 12+ tvarov per podstatne meno. Regex pre 12 typov
zmluv by vyzadoval desiatky patternov. "IT zakazka" = zmluva o dielo, ale regex
to nezachyti (slovo "dielo" v texte nie je). Gemini Flash stoji ~$0.003 a zvladne
morfologiu nativne.

### Preco hard filter len na contract_type + breach_type
Hard filter na decision by sposoboval tautologicke statistiky (filter na
awarded_full → 100% awarded, duh). Hard filter na factors by znemoznil factor
lift (treba obe skupiny na porovnanie). Preto su decision_interest a
factor_interest soft ranking signaly v Stage 3, nie hard filtre v Stage 2.

### Preco factor LIFT a nie frequency
Frekvencia je zavadzajuca ak faktor sa spomina vo vacsine pripadov. Lift ukazuje
SKUTOCNU prediktivnu silu: P(negative | moderated) / P(negative | awarded).
V datasete napriklad pomer_k_istine ma lift 26.4x (silny prediktor) ale
dobre_mravy len 12.6x (slabsi, hoci sa spomina casto).

### Preco meritorny pomer
Zamietnute a vratene pripady zlyhali z inych dovodov (neplatnost zmluvy, procesne
chyby) — NIE kvoli vyske sadzby. Miera moderacie medzi meritornymi rozhodnutiami
(awarded + moderated only) je pre pravnika relevantnejsia ako celkova distribucia
s dismissed/returned percentami.

### Preco numpy a nie ChromaDB v multi-doc
Pri 202 (buducnost ~1500) zaznamoch je brute-force numpy dot product rychlejsi
ako ANN search v ChromaDB. Bez zavislosti na databaze. ChromaDB ma zmysel od
milionov vektorov.

### Preco ziadny Deep Retrieval (chunk-level)
Extraction pipeline uz poskytuje ~300 tokenov textoveho kontextu per pokuta
(reasoning_summary + key_quotes + factor evidence). Deep retrieval by vracal
chunky z tych istych paragrafov. Cista redundancia. Bez neho: jednoduchsi kod,
-1.5s latency, silnejsi argument pre obhajobu.

### Preco safe_rate moderated injection
Pri intent=safe_rate ranker promuje awarded cases (query "bezpecna sadzba" ma
vyssie embedding similarity s potvrdenymi). Bez intervencie LLM nevidi ziadny
moderated case a nevaroval by pred rizikami. Injection v pipeline.py zabezpecuje
ze top 5 obsahuje aspon 1 moderated case.

### Preco low-n guard v synthesis prompte
Pri n=1 alebo n=3 LLM bez guardu pise 4-odsekovu analyzu ako keby to bola
statisticky vyznamna vzorka. Guard (pridany k system promptu pri n<5) zakazuje
vseobecne zavery, odporucania a extrapolaciu — LLM opisuje kazdy pripad
jednotlivo a explicitne uvedie ze je malo dat.

### Preco jeden LLM call v Stage 5
Vsetky data su pripravene — statistiky Python-computed, top 5 precedensov
s textovym kontextom vybranych. LLM len syntetizuje do slovenciny s citaciami.
Dalsi call by pridal latency a cost bez pridanej hodnoty.

### Preco dva formatery analytiky (for_lawyer + for_llm)
Pravnik nechce citat "lift 26.4x" — chce "pomer k istine bol problem v 8 z 9
pripadov". LLM naopak potrebuje plny kontrast (moderated vs awarded counts) aby
vedel pisat presne vety. Preto mam dve formatovacie funkcie: `format_for_lawyer`
pre displej, `format_for_llm` pre grounding v prompte.
