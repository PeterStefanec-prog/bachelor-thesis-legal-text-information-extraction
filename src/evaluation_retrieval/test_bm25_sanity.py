"""
BM25 sanity check script.
Run from project root:  python src/evaluation_retrieval/test_bm25_sanity.py

Tests:
  1. simplemma lemmatization quality on key legal terms
  2. tokenize_slovak on real legal sentences
  3. BM25 query tokens vs corpus tokens overlap (will they actually match?)
  4. BM25 scoring on a mini-corpus (does the right chunk win?)
  5. Edge cases: empty text, single word, numbers only
"""

import re
import simplemma
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Copy of SLOVAK_STOPWORDS and tokenize_slovak from evaluate_retrieval.py
# so this script is standalone and testable without imports
# ---------------------------------------------------------------------------
SLOVAK_STOPWORDS = {
    "a", "aj", "ak", "ako", "ale", "alebo", "ani", "áno", "asi",
    "by", "bol", "bola", "bolo", "boli", "buď", "byť", "bez",
    "do", "dňa",
    "ho", "jeho", "jej", "ich",
    "je", "ju",
    "keď", "keďže", "kde", "ku", "ktorý", "ktorá", "ktoré", "ktorej", "ktorým", "ktorom",
    "na", "nad", "nie", "no", "než",
    "od", "ods",
    "po", "pod", "podľa", "pre", "pri", "pred", "preto",
    "sa", "si", "so", "sú",
    "ten", "to", "tá", "tú", "tej", "tom", "tomu", "tým", "tento", "tiež", "tak", "takže",
    "vo", "vz",
    "za", "zo", "že",
}


def tokenize_slovak(text):
    text = text.lower()
    text = re.sub(r'§\s*(\d+)', r'§\1', text)
    raw_tokens = re.findall(
        r'§\d+'
        r'|\d{1,3}(?:\.\d{3})*(?:,\d+)?%?'
        r'|[a-záäčďéíľĺňóôŕšťúýžA-ZÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ]+',
        text
    )
    tokens = []
    for t in raw_tokens:
        if t.startswith("§") or t[0].isdigit():
            tokens.append(t)
        elif len(t) > 1:
            lemma = simplemma.lemmatize(t, lang='sk')
            if lemma not in SLOVAK_STOPWORDS and len(lemma) > 1:
                tokens.append(lemma)
    return tokens


BM25_QUERIES = {
    "q_breach":    "zmluva predmet porušenie povinnosť záväzok dlžník zhotoviteľ objednávateľ žalovaný neuhradil nesplnil",
    "q_rate":      "zmluvná pokuta výška sadzba percentá ročne denne omeškanie eur suma",
    "q_principal": "istina suma pohľadávka dlh záväzok eur faktúra cena dielo",
    "q_outcome":   "zmluvná pokuta priznaná znížená zamietnutá moderácia §301 neplatnosť vrátená",
    "q_reasoning": "primeranosť neprimeraná dôvod posúdenie zníženie argumenty súd záver",
    "q_factors":   "dobré mravy škoda pomer istina úrok omeškanie kumulácia zabezpečovacia funkcia",
}

passed = 0
failed = 0


def check(test_name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  OK   {test_name}")
        passed += 1
    else:
        print(f"  FAIL {test_name}  --  {detail}")
        failed += 1


# ===== TEST 1: simplemma lemmatization on key legal terms =====
print("\n" + "=" * 70)
print("TEST 1: simplemma lemmatization quality")
print("=" * 70)

# (inflected_form, expected_lemma_or_acceptable_set)
LEMMA_TESTS = [
    # Nouns - case inflections
    ("pokuty",        {"pokuta"}),               # genitive sg
    ("pokút",         {"pokuta"}),               # genitive pl
    ("pokutou",       {"pokuta"}),               # instrumental sg
    ("pokutám",       {"pokuta"}),               # dative pl
    ("záväzku",       {"záväzok"}),              # genitive sg
    ("záväzkov",      {"záväzok"}),              # genitive pl
    ("zmluvy",        {"zmluva"}),               # genitive sg
    ("zmlúv",         {"zmluva", "zmlúv"}),      # genitive pl (tricky)
    ("pohľadávky",    {"pohľadávka"}),           # genitive sg
    ("pohľadávok",    {"pohľadávka"}),           # genitive pl
    ("istiny",        {"istina"}),               # genitive sg
    ("dlžníka",       {"dlžník"}),               # genitive sg
    ("dlžníkom",      {"dlžník"}),               # instrumental sg
    ("omeškania",     {"omeškanie"}),            # genitive sg
    ("omeškanie",     {"omeškanie"}),            # nominative sg
    ("omeškaním",     {"omeškanie"}),            # instrumental sg
    ("neplatnosti",   {"neplatnosť"}),           # genitive sg
    ("primeranosti",  {"primeranosť"}),          # genitive sg
    ("škody",         {"škoda"}),                # genitive sg
    ("mravmi",        {"mrav"}),                 # instrumental pl
    ("mravy",         {"mrav"}),                 # nominative pl

    # Adjectives
    ("zmluvná",       {"zmluvný", "zmluvná"}),
    ("zmluvnej",      {"zmluvný", "zmluvná"}),
    ("zmluvnú",       {"zmluvný", "zmluvná"}),
    ("neprimeraná",   {"neprimeraný", "neprimeraná"}),
    ("neprimeranú",   {"neprimeraný", "neprimeraná"}),
    ("neprimerané",   {"neprimeraný", "neprimerané"}),
    ("priznaná",      {"priznaný", "priznaná", "priznať"}),
    ("znížená",       {"znížený", "znížená", "znížiť"}),
    ("zamietnutá",    {"zamietnutý", "zamietnutá", "zamietnuť"}),
    ("zabezpečovacia", {"zabezpečovací", "zabezpečovacia"}),
    ("dobré",         {"dobrý", "dobré"}),

    # Verbs
    ("neuhradil",     {"neuhradiť", "uhradiť", "neuhradil"}),
    ("nesplnil",      {"nesplniť", "splniť", "nesplnil"}),
    ("porušil",       {"porušiť", "porušil"}),
    ("znížil",        {"znížiť", "znížil"}),
    ("posúdil",       {"posúdiť", "posúdil"}),
]

lemma_mismatches = []
for word, acceptable in LEMMA_TESTS:
    lemma = simplemma.lemmatize(word, lang='sk')
    ok = lemma in acceptable
    if not ok:
        lemma_mismatches.append((word, lemma, acceptable))
    check(f"'{word}' -> '{lemma}'", ok,
          f"expected one of {acceptable}")

if lemma_mismatches:
    print(f"\n  WARNING: {len(lemma_mismatches)} lemmatization mismatches found!")
    print("  This means BM25 query tokens and corpus tokens may NOT match for these words.")
    print("  Consider adding the actual lemma forms to BM25_QUERIES as fallback synonyms.\n")


# ===== TEST 2: tokenize_slovak on real legal sentences =====
print("\n" + "=" * 70)
print("TEST 2: tokenize_slovak on real legal text")
print("=" * 70)

test_sentences = [
    (
        "Zmluvná pokuta vo výške 0,05% za každý deň omeškania zo sumy 15.234,60 EUR",
        # Should preserve: 0,05%, 15.234,60, eur; should lemmatize: zmluvná->zmluvný, pokuty->pokuta
        {"must_contain": ["0,05%", "15.234,60", "eur", "omeškanie", "deň"],
         "must_not_contain": ["za", "zo"]}  # stopwords
    ),
    (
        "Súd uplatnil moderačné oprávnenie podľa § 301 Obchodného zákonníka",
        {"must_contain": ["§301", "súd", "moderačný"],
         "must_not_contain": ["podľa"]}
    ),
    (
        "Žalovaný neuhradil faktúru č. 200100539 v celkovej sume 350.743,12 Eur",
        {"must_contain": ["350.743,12", "200100539", "eur", "faktúra"],
         "must_not_contain": []}
    ),
]

for sentence, expectations in test_sentences:
    tokens = tokenize_slovak(sentence)
    print(f"\n  Input:  '{sentence}'")
    print(f"  Tokens: {tokens}")

    for word in expectations["must_contain"]:
        # Check if the word OR its lemma is in tokens
        word_lower = word.lower()
        lemma_form = simplemma.lemmatize(word_lower, lang='sk')
        found = word_lower in tokens or lemma_form in tokens
        check(f"  contains '{word}' (or lemma '{lemma_form}')", found,
              f"tokens = {tokens}")

    for word in expectations["must_not_contain"]:
        word_lower = word.lower()
        lemma_form = simplemma.lemmatize(word_lower, lang='sk')
        found = word_lower in tokens or lemma_form in tokens
        check(f"  does NOT contain stopword '{word}'", not found,
              f"stopword '{word}' (lemma '{lemma_form}') found in tokens!")


# ===== TEST 3: BM25 query-corpus token overlap =====
print("\n" + "=" * 70)
print("TEST 3: BM25 query tokens vs typical corpus tokens overlap")
print("=" * 70)
print("  (Do the query keywords actually match what appears in real text?)\n")

# Simulate a few real corpus chunks (from golden dataset)
corpus_chunks = [
    "Zmluva o výhradnej spolupráci materskej agentúry s modelkou. "
    "Žalovaná porušila ustanovenie zmluvy tým, že použila fotografie.",

    "Zmluvnú pokutu vo výške 0,05% za každý deň omeškania z ceny diela. "
    "Zo sumy 4.323.809 Sk predstavuje sumu 2.870,40 eur.",

    "Istina pohľadávky žalobcu vo výške 350.743,12 Eur podľa faktúry č. 200100539. "
    "Žalovaný bol povinný uhradiť cenu diela.",

    "Súd uplatnil moderačné oprávnenie podľa § 301 Obchodného zákonníka "
    "a zmluvnú pokutu znížil na sumu 1.000 Eur.",

    "Zmluvná pokuta je zjavne neprimeraná. Rozpor s dobrými mravmi. "
    "Pomer pokuty k istine je neúnosný. Škoda žalobcovi nevznikla.",

    "Predmetom zmluvy bol finančný leasing motorového vozidla. "
    "V dôsledku neplatenia leasingových splátok žalovaným žalobca od zmluvy odstúpil.",
]

tokenized_corpus = [tokenize_slovak(c) for c in corpus_chunks]

for q_key, q_text in BM25_QUERIES.items():
    q_tokens = tokenize_slovak(q_text)
    print(f"  {q_key}:")
    print(f"    Query tokens: {q_tokens}")

    # Check overlap with each chunk
    best_overlap = 0
    best_chunk_idx = -1
    for i, c_tokens in enumerate(tokenized_corpus):
        overlap = set(q_tokens) & set(c_tokens)
        if len(overlap) > best_overlap:
            best_overlap = len(overlap)
            best_chunk_idx = i
        if overlap:
            print(f"    Chunk {i}: overlap={len(overlap)} tokens: {overlap}")

    check(f"  {q_key} has at least 1 corpus match",
          best_overlap > 0,
          "NO overlap with any corpus chunk - query tokens are completely disconnected!")
    print()


# ===== TEST 4: BM25 scoring on mini-corpus =====
print("=" * 70)
print("TEST 4: BM25 scoring - does the right chunk rank #1?")
print("=" * 70)

bm25 = BM25Okapi(tokenized_corpus, k1=1.2, b=0.4)

# q_rate should rank the penalty-rate chunk highest (chunk 1)
q_rate_tokens = tokenize_slovak(BM25_QUERIES["q_rate"])
scores = bm25.get_scores(q_rate_tokens)
ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
print(f"\n  q_rate query: {q_rate_tokens}")
for rank, idx in enumerate(ranked):
    print(f"    Rank {rank+1}: chunk {idx} (score={scores[idx]:.4f}) - {corpus_chunks[idx][:60]}...")
check("q_rate: chunk 1 (penalty rate) is in top 2", ranked[0] == 1 or ranked[1] == 1,
      f"top ranked = chunk {ranked[0]}")

# q_principal should rank the principal/istina chunk highest (chunk 2)
q_principal_tokens = tokenize_slovak(BM25_QUERIES["q_principal"])
scores = bm25.get_scores(q_principal_tokens)
ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
print(f"\n  q_principal query: {q_principal_tokens}")
for rank, idx in enumerate(ranked):
    print(f"    Rank {rank+1}: chunk {idx} (score={scores[idx]:.4f}) - {corpus_chunks[idx][:60]}...")
check("q_principal: chunk 2 (istina/faktura) is in top 2", ranked[0] == 2 or ranked[1] == 2,
      f"top ranked = chunk {ranked[0]}")

# q_outcome should rank the §301 chunk highest (chunk 3)
q_outcome_tokens = tokenize_slovak(BM25_QUERIES["q_outcome"])
scores = bm25.get_scores(q_outcome_tokens)
ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
print(f"\n  q_outcome query: {q_outcome_tokens}")
for rank, idx in enumerate(ranked):
    print(f"    Rank {rank+1}: chunk {idx} (score={scores[idx]:.4f}) - {corpus_chunks[idx][:60]}...")
check("q_outcome: chunk 3 (§301 moderation) is in top 2", ranked[0] == 3 or ranked[1] == 3,
      f"top ranked = chunk {ranked[0]}")

# q_factors should rank the factors chunk highest (chunk 4)
q_factors_tokens = tokenize_slovak(BM25_QUERIES["q_factors"])
scores = bm25.get_scores(q_factors_tokens)
ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
print(f"\n  q_factors query: {q_factors_tokens}")
for rank, idx in enumerate(ranked):
    print(f"    Rank {rank+1}: chunk {idx} (score={scores[idx]:.4f}) - {corpus_chunks[idx][:60]}...")
check("q_factors: chunk 4 (dobré mravy, škoda, pomer) is in top 2", ranked[0] == 4 or ranked[1] == 4,
      f"top ranked = chunk {ranked[0]}")


# ===== TEST 5: Edge cases =====
print("\n" + "=" * 70)
print("TEST 5: Edge cases")
print("=" * 70)

check("Empty string returns empty list", tokenize_slovak("") == [], str(tokenize_slovak("")))
check("Single stopword returns empty", tokenize_slovak("je") == [], str(tokenize_slovak("je")))
check("Single char word filtered out", tokenize_slovak("a v o") == [], str(tokenize_slovak("a v o")))
check("Only numbers", tokenize_slovak("123 456,78 0,05%") == ["123", "456,78", "0,05%"],
      str(tokenize_slovak("123 456,78 0,05%")))
check("§ with space", "§301" in tokenize_slovak("§ 301"), str(tokenize_slovak("§ 301")))
check("§ without space", "§301" in tokenize_slovak("§301"), str(tokenize_slovak("§301")))


# ===== SUMMARY =====
print("\n" + "=" * 70)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
if failed == 0:
    print("All checks passed!")
else:
    print(f"WARNING: {failed} checks failed - review output above for details.")
    print("Failed checks may indicate BM25 query/corpus token mismatches.")
print("=" * 70)
