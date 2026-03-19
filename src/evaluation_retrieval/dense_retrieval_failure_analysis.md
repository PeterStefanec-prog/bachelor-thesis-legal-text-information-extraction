# Why Dense Retrieval Fails — Analysis

> Script: `src/evaluation_retrieval/diagnose_retrieval_mE5_failures.py`

---

## The problem in one sentence

The mE5 model retrieves chunks about **legal argumentation about the contract** (právna argumentácia o zmluve) instead of chunks that **factually describe the contract** (skutkový opis zmluvy). Both types of text look almost identical to the embedding model.

---

## What the script does

For each document where `q_context` consistently fails, it:
1. ranks ALL chunks in the document by cosine similarity to the query
2. finds where the correct chunks (containing golden quotes) actually rank
3. shows what the model retrieved instead
4. checks how rare the distinctive keyword is — to preview what BM25 would do

---

## Root cause — explained first

Every Slovak court decision (súdne rozhodnutie) has the same structure:

**Section 1 — skutkový stav** (factual background): who signed what contract, what the obligation was, what breach occurred. Short, dense with specific names and contract numbers.

**Section 2 — právne posúdenie** (legal assessment): court's reasoning about the penalty proportionality, citing §301 ObchZ, discussing dobré mravy, zabezpečovacia funkcia, výška škody. Long, full of legal argumentation.

The query `"Čo bolo predmetom zmluvy medzi stranami? Akú povinnosť si dlžník prevzal..."` should retrieve Section 1. But both sections contain the same vocabulary: *zmluva, povinnosť, žalovaný, porušenie, zmluvná pokuta*. The mE5 embedding sees these words and scores both sections almost equally. Section 2 is longer and appears more times in the document, so it often wins.

---

## Results document by document

### KS_Bratislava_1Cob_40 (leasingová zmluva)

**What model retrieved (top 5):**
- Chunk 22: *"Žalobca má za to, že v prípade žalovaného bola dojednaná zmluvná pokuta zjavne neprimeraná..."* — odvolacia argumentácia o primeranosti
- Chunk 13: *"Žalovaný v odvolaní ďalej uviedol, že odstúpenie od zmluvy podľa § 351 ods..."* — námietky žalovaného

All 5 retrieved chunks are from the právne posúdenie section. Makes sense to the model — they all discuss zmluva and povinnosť.

**Where is the actual answer:**
- Rank **#15/31** — *"V dôsledku neplatenia leasingových splátok žalovaným v 1. rade žalobca dňa 15.4.2009 od zmluvy odstúpil"*
- Rank **#18/31** — *"predmetom bol finančný leasing motorového vozidla IVECO Daily 35 S 10 V"*
- Rank **#22/31** — *"vyúčtovanie pri predčasnom ukončení leasingovej zmluvy"*

These are in the skutkový opis section — factual, short, specific. The model underranks them because they are less "discussive" about contracts.

**BM25 fix:** `"leasingovú zmluvu č. 9587/25"` — číslo zmluvy, appears in **1/31 chunks**. BM25 would rank that chunk #1 immediately. Dense model ranked it #18.

---

### KS_Trnava_32Cob_3_2021 (abonentná zmluva, telekomunikačné služby)

**What model retrieved:**
- Chunk 19: *"Dohoda o zmluvnej pokute musí podľa názoru odvolacieho súdu vyhovieť testu primeranosti..."*
- Chunk 15: *"Žalobca tvrdí, že zmluvné ustanovenia týkajúce sa zmluvných pokút odrážajú..."*

Again — právna diskusia o pokute, nie opis čo bolo predmetom zmluvy.

**Where is the actual answer:**
- Rank **#9/38** — *"Sporové strany ako podnikateľské subjekty uzavreli dňa 21.01.2019 tzv. abonentnú zmluvu"*
- Rank **#10/38** — *"strany medzi sebou uzavreli v ten istý deň aj zmluvu o servisnej službe"*

Score difference between rank #5 (retrieved) and rank #9 (correct): **only 0.005**. That tiny margin is the difference between a PERFECT HIT and a MISS.

**BM25 fix:** `"abonentnú zmluvu"` appears in **2/38 chunks** — both are the skutkový opis chunks. BM25 → rank #1 and #2 immediately.

---

### KS_Banská_Bystrica_43CoPv_10_2023 (zákaz konkurencie, ochranná známka)

Hardest case. 61 chunks total (long appellate decision), correct chunks at **#14, #41, #47**.

The document is an IP/trademark case (ochranná známka, nekalosúťažné konanie) and the model keeps retrieving chunks about constitutional court precedents and competition law arguments — all legally sophisticated text that semantically matches the query.

**BM25 caveat:** `"metabolic balance"` appears **10/61 chunks** — it's the disputed brand name, referenced throughout the whole decision. BM25 still helps (narrows candidates) but not as dramatically as in the other cases. This is the exception — hybrid scoring would help but won't be a single-keyword fix.

---

### NS_SR_1Obdo_72_2019 (realitná agentúra, sprostredkovateľská zmluva)

**What model retrieved:**
- Chunk 1: *"proti ktorému žalovaná podala odvolanie, ale na zdôraznenie jeho správnosti uviedol..."*
- Chunk 3: *"22. K tvrdeniu súdu, že žalovaná podľa zmluvy bola povinná akúkoľvek novú pribratú..."*

These are all about the obligations under the contract — correct topic, wrong context level.

**Where is the actual answer:**
- Rank **#20/42** — *"žalobkyňa a žalovaná ako podnikateľky uzavreli dňa 17.04.2012 Zmluvu o poskytovaní služieb, na základe ktorej žalovaná vykonávala sprostredkovanie predaja, prenájmu a kúpy nehnuteľností"*

**BM25 fix:** `"sprostredkovanie predaja, prenájmu a kúpy nehnuteľností"` — very specific phrase, appears in **1/42 chunks**. BM25 → rank #1. Dense → rank #20.

---

### KS_Bratislava_2Co_380_2011 (zmluva o dielo, rodinný dom)

Partial success — one correct chunk reached rank #1, but the remaining 4 needed chunks ranked **#6, #7, #11, #13**. For a PERFECT HIT all golden quotes must be found, so this still counts as a MISS.

**BM25 fix:** `"rodinný dom"` in **1/28 chunks** — the chunk describing the construction contract subject. Dense ranked it #13. BM25 → #1.

---

## Summary table

| Document | Correct chunk rank | Score gap to top-5 | BM25 keyword frequency | BM25 would fix? |
|---|---|---|---|---|
| KS_BA_1Cob_40 | #15, #18, #22 / 31 | -0.022 | 1/31 | ✓ yes |
| KS_Trnava | #9, #10 / 38 | -0.005 | 2/38 | ✓ yes |
| KS_Banska | #14, #41, #47 / 61 | -0.009 | 10/61 | ~ partially |
| NS_SR_1Obdo_72 | #15, #20 / 42 | -0.018 | 1/42 | ✓ yes |
| KS_BA_2Co_380 | #6–#13 / 28 | -0.008 | 1/28 | ✓ yes |

---

## Expected impact of BM25 hybrid

- `q_context` (skutkový opis zmluvy): **4/5 failures** would be fixed — these are exactly the cases where contract-specific keywords (číslo zmluvy, predmet zmluvy, meno strán) appear only once or twice in the whole document
- `q_penalty` (parametre zmluvnej pokuty): same logic — specific penalty amounts (0,05 % denne, 10.000 EUR, §9.2 zmluvy) are unique per document
- `q3_combined` (moderácia podľa §301): already works well with dense — moderačná argumentácia is semantically distinctive enough, BM25 on terms like "moderačné oprávnenie", "§ 301 Obchodného zákonníka" would be a small bonus

Expected overall recall improvement: **~10-15 percentage points**, concentrated on q_context and q_penalty where dense consistently underperforms.