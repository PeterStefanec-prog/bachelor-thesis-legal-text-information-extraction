# #####################
# Shared query definitions for retrieval evaluation
# #####################
# i moved these here from evaluate_retrieval.py because both flat and hierarchical evaluation modes use the exact same queries.
# Having them in one place means i dont maintain two copies.


# Dense retrieval queries - full Slovak questions that capture semantic intent.
# Embedding model understands what we aer asking, so we use natural language
QUERIES = {
    # Call 1 - contract facts and penalty definition
    "q_breach":    "Čo bolo predmetom zmluvy a akú povinnosť dlžník porušil?",
    "q_contract":  "Aká zmluva bola uzatvorená medzi stranami, kto boli zmluvné strany a aké boli podmienky zmluvy?",
    # NOTE: i tried adding a 3rd sub-query "q_facts" to capture the factual background
    # (skutkový stav) for Call 1. the idea was that q_breach and q_contract miss some
    # q1 golden quotes that describe circumstances of the case. but in experiments it
    # only added +0.42% recall for +1.6% coverage increase - the hybrid retriever with
    # reranker already finds those chunks through q_breach and q_contract. not worth it.
    # "q_facts":     "Aký je skutkový stav veci? Aké boli okolnosti uzatvorenia zmluvy, priebeh plnenia a vznik nároku na zmluvnú pokutu?",
    "q_rate":      "Aká je sadzba a spôsob výpočtu zmluvnej pokuty?",
    "q_principal": "Aká je výška istiny, dlžnej sumy alebo hlavného záväzku?",
    # Call 2 - court decision and reasoning
    "q_outcome":   "Bola zmluvná pokuta priznaná, znížená podľa § 301, zamietnutá alebo vrátená?",
    "q_reasoning": "Aké argumenty súd použil pri posudzovaní primeranosti zmluvnej pokuty?",
    "q_factors":   "Posúdil súd rozpor s dobrými mravmi, pomer pokuty k istine, výšku škody alebo kumuláciu s úrokom?",
}



# BM25 queries - keyword bags instead of questions.
# BM25 matches exact tokens, not meaning.
# So queries must contain actual words that appear in the legal text, not questions about them.
# Includes synonyms and inflected forms that simplemma might not cover.
BM25_QUERIES = {
    # Call 1
    "q_breach":    "zmluva uzatvorená predmet porušenie povinnosť záväzok dlžník zhotoviteľ objednávateľ žalovaný neuhradil nesplnil skutkový stav sporové strany napadnutým rozsudkom",
    "q_contract":  "zmluva uzatvorená predmet zmluvy strany zmluvné podmienky článok zhotoviteľ objednávateľ nájomca prenajímateľ dodávateľ odberateľ žalobca žalovaný uzavreli dohoda",
    # "q_facts":     "skutkový stav okolnosti uzatvorenie zmluva plnenie záväzok dodanie dielo predmet omeškanie nesplnenie napadnutý rozsudok prvoinštančný súd žalobca tvrdil uviedol",
    "q_rate":      "zmluvná pokuta výška sadzba percentá ročne denne omeškanie eur suma článok bod zmluvy dohodnutá",
    "q_principal": "istina suma pohľadávka dlh záväzok eur faktúra cena dielo splatnosť uhradiť zaplatiť",
    # Call 2
    "q_outcome":   "zmluvná pokuta priznaná znížená zamietnutá moderácia moderačné oprávnenie §301 neplatnosť vrátená rozhodol uložil zaviazal",
    "q_reasoning": "primeranosť neprimeraná dôvod posúdenie zníženie argumenty súd záver odôvodnenie odvolací potvrdil zmenil",
    "q_factors":   "dobré mravy škoda pomer istina úrok omeškanie kumulácia zabezpečovacia funkcia hodnota význam zabezpečovanej povinnosti",
}


# Maps each query key to the golden dataset CSV column for evaluation.
# Some queries share a column - their retrieved chunks are merged before checking against golden quotes.
QUERY_TO_CSV_COLUMN = {
    "q_breach":    "q1_context_quotes",
    "q_contract":  "q1_context_quotes",    # evaluated together with q_breach
    # "q_facts":     "q1_context_quotes",  # removed - did not improve recall (see note above)
    "q_rate":      "q2_penalty_quotes",    # evaluated
    "q_principal": "q2_penalty_quotes",    # together
    "q_outcome":   "q3_moderation_quotes", #
    "q_reasoning": "q3_moderation_quotes", # evaluated
    "q_factors":   "q3_moderation_quotes", # together
}
