"""
Reranker for Slovak legal court decisions about contractual penalties (zmluvna pokuta).

This is basically a post-processing step that runs AFTER retrieval and BEFORE the LLM.
The problem is that retrieval sometimes returns chunks that are semantically similar
to the query but dont actually contain the useful information. For example, a chunk
about an appeal ("odvolanie") mentions "zmluvna pokuta" many times but doesnt have
the actual penalty amount or the court's reasoning.

So what this does:
  1. Retrieval returns more chunks than we need (e.g. 10 instead of 5)
  2. This reranker scores each chunk using regex patterns for legal keywords
  3. It combines the retrieval score with the keyword score
  4. Returns only the best top_k chunks

I figured out the keywords by reading through 15+ court decisions and noting which
phrases appear in the "useful" chunks vs the "noise" chunks. For example:
  - "zmluvnu pokutu vo vyske" almost always means the chunk has the actual penalty definition
  - "trovy konania" almost always means the chunk is just about case costs (useless)
  - "§ 301" means the chunk talks about penalty moderation (very useful for q_outcome)

The weights (how much each keyword is worth) come from how often these phrases
appeared in the golden dataset documents and how reliably they predict useful chunks.
"""

import re


# ==========================================
# KEYWORD PATTERNS AND THEIR WEIGHTS
# ==========================================
# Each pattern is a tuple: (compiled_regex, weight)
#
# How i decided the weights:
#   +4 = this keyword almost guarantees the chunk is useful for this query
#   +3 = strong signal, appears in most relevant chunks
#   +2 = moderate signal, helpful but not decisive
#   +1 = weak signal, slightly better than nothing
#   -1 = weak noise indicator
#   -2 = strong noise indicator (procedural stuff, case costs)
#
# I went through all 10 golden dataset documents + 5 extra ones and counted
# how often each phrase appears. The percentages in comments show how many
# of the 15 documents contained that phrase at least once.


# ==========================================
# BREACH / CONTRACT CONTEXT patterns
# ==========================================
# These help find chunks that describe what the contract was about and what
# the debtor did wrong. The main problem here is that chunk_0 (the first
# paragraph of the court decision) usually describes the contract and the
# breach, but dense retrieval misses it because the language is procedural:
# "Napadnutym rozhodnutim sud prvej instancie..." sounds like a court ruling,
# not like a contract description, so the embedding model ranks it low.

BREACH_PATTERNS = [
    # what type of contract was it? (appears in 40-53% of docs)
    (re.compile(r'(?i)zmluv[auúy]\s+o\s+(dielo|spolupráci|poskytovan|kúp|prenájm|obchodnom zastúpen|úver)', re.UNICODE), 3),
    (re.compile(r'(?i)(leasingov|rámcov|abonentn)[áaéeúu]\s+zmluv', re.UNICODE), 3),
    (re.compile(r'(?i)dohod[auúy]\s+o\s+užívan', re.UNICODE), 2),

    # what did the debtor do wrong? (73-93% of docs have these)
    (re.compile(r'(?i)porušil[aio]?\s+(zmluvnú\s+)?povinnosť', re.UNICODE), 3),
    (re.compile(r'(?i)porušeni[ae]\s+zmluvy', re.UNICODE), 3),  # "porusenie zmluvy" - direct breach language (3/10 docs)
    (re.compile(r'(?i)(nesplnil|neuhradil|nedodržal|nezaplatil|neposkytoval)[aio]?', re.UNICODE), 3),
    (re.compile(r'(?i)v\s+omeškan[ií]', re.UNICODE), 2),
    (re.compile(r'(?i)odstúpil[aio]?\s+(od\s+)?zmluvy', re.UNICODE), 2),
    (re.compile(r'(?i)(konkurenčn[áéú]\s+doložk|zákaz[ue]?\s+konkurencie)', re.UNICODE), 2),  # competition clause breach (BB_43CoPv)

    # who signed the contract? (very common in intro chunks)
    (re.compile(r'(?i)(žalobca|žalobkyňa|žalovan[ýá])\s+(a\s+)?(žalobca|žalovan[ýá])?\s*(uzavreli|uzatvorili|uzavrela)', re.UNICODE), 3),
    (re.compile(r'(?i)(zhotoviteľ|objednávateľ|prenajímateľ|nájomca|dodávateľ|odberateľ)', re.UNICODE), 1),
    # old-style party terminology (pre-2015 CPC decisions like KS_BA_2Co_380_2011)
    (re.compile(r'(?i)(navrhovateľ|odpor(?:ca|kyňa))', re.UNICODE), 1),

    # phrases that appear at the start of court decisions (chunk_0 markers)
    (re.compile(r'(?i)predmetom\s+zmluvy', re.UNICODE), 3),  # "predmetom zmluvy" = about the contract (strong)
    (re.compile(r'(?i)predmetom\s+(konania|sporu)', re.UNICODE), 1),  # "predmetom konania" = procedural meaning (weak)
    (re.compile(r'(?i)na\s+základe\s+(vykonaného\s+)?dokazovania', re.UNICODE), 2),
    (re.compile(r'(?i)(skutkov[ýé]\s+stav|skutkový\s+záver)', re.UNICODE), 1),  # reduced from 2: matches procedural review chunks too
    (re.compile(r'(?i)napadnutým\s+rozhodnutím', re.UNICODE), 2),
    (re.compile(r'(?i)súd\s+prvej\s+inštancie', re.UNICODE), 1),
    (re.compile(r'(?i)súd\s+prvého\s+stupňa', re.UNICODE), 1),  # old terminology (pre-2015 decisions)
]


# ==========================================
# PENALTY RATE / AMOUNT patterns
# ==========================================
# These find chunks with the actual numbers: how much is the penalty, what percentage,
# how is it calculated. The tricky part is that "zmluvna pokuta" appears in like
# 60-80 chunks per document (everyone keeps talking about it), but the actual
# DEFINITION with the numbers is only in 2-3 chunks.

RATE_PATTERNS = [
    # the actual penalty definition - this is the most valuable signal (80% of docs)
    # "zmluvnu pokutu vo vyske 10.000 eur" = jackpot, this IS the penalty clause
    (re.compile(r'(?i)zmluvn[úu]\s+pokut[uú]\s+vo\s+výške', re.UNICODE), 4),
    (re.compile(r'(?i)(povinný|povinná|zaviazal[ai]?\s+sa)\s+zaplatiť', re.UNICODE), 2),

    # percentage rates WITH context - "0,05% denne", "0,2% z dlznej sumy" (80% of docs)
    # NOTE: the bare percentage pattern (\d+[.,]\d+\s*%) was reduced from 3 to 2 because
    # it has too many false positives: catches interest rates (8,25%, 0,1%), trovy konania
    # percentages (99,9%), and fulfillment percentages (84,20%). The more specific pattern
    # with context words (denne/mesacne/rocne) stays at 4 and is the reliable one.
    (re.compile(r'\d+[.,]\d+\s*%', re.UNICODE), 2),  # bare percentage - reduced from 3 (too many false positives)
    (re.compile(r'(?i)\d+\s*%\s*(denne|mesačne|ročne|za\s+každý\s+deň)', re.UNICODE), 4),  # percentage WITH time unit = very reliable
    (re.compile(r'(?i)z\s+(dlžnej\s+)?sumy', re.UNICODE), 2),
    (re.compile(r'(?i)z\s+(fakturovanej\s+)?cen[yuy]', re.UNICODE), 2),

    # fixed amount penalty - "vo vyske 10.000 eur" (appears when penalty is a fixed sum)
    # reduced from 4 to 3: also matches non-penalty amounts like "nahradu trov vo vyske 623 eur"
    (re.compile(r'(?i)vo\s+výške\s+[\d\s.,]+\s*(eur|€|sk)', re.UNICODE), 3),

    # multiplier penalties - "3-nasobok mesacnej odplaty" (less common but very specific)
    (re.compile(r'(?i)\d+-?násobo?k', re.UNICODE), 3),

    # contract article references - "cl. IV. bod 4.8" (points to exact clause)
    (re.compile(r'(?i)(čl\.|článo?k)\s+[IVXLC]+\.?\s*(bod|ods\.?)\s*\d+', re.UNICODE), 2),
]


# ==========================================
# PRINCIPAL / SECURED AMOUNT patterns
# ==========================================
# These find chunks about the underlying debt - the amount that the penalty secures.
# Important because the LLM needs to extract "pokuta 0,05% z 350.743 EUR" and for
# that it needs BOTH the rate AND the principal amount, which are sometimes in
# different chunks.

PRINCIPAL_PATTERNS = [
    # specific EUR amounts (93% of docs have these)
    (re.compile(r'[\d\s]+[.,]\d{2}\s*(eur|€|sk)', re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'(?i)(istina|dlžná\s+suma|zostatok\s+dlhu|celkov[áé]\s+sum[auy])', re.UNICODE), 3),

    # invoice and price references (47-67% of docs)
    (re.compile(r'(?i)(faktúr[ouya]|cen[auúy]\s+diela|cen[auúy]\s+zákazky)', re.UNICODE), 2),
    (re.compile(r'(?i)(splatnosť|splatn[áé]|lehot[auúy]\s+splatnosti)', re.UNICODE), 2),
    (re.compile(r'(?i)(vyúčtovanie|vyúčtoval)', re.UNICODE), 2),

    # payment obligation phrases
    (re.compile(r'(?i)(uhradiť|zaplatiť|splatiť)\s+(sumu|čiastku|cenu)', re.UNICODE), 2),
]


# ==========================================
# OUTCOME patterns (what did the court decide?)
# ==========================================
# Was the penalty reduced, upheld, or dismissed?
# One thing i noticed: both the court AND the parties use the same words.
# "sud znizil" (court reduced) vs "zalovany navrhol znizit" (defendant proposed to reduce)
# look very similar in embedding space. The regex can at least partially tell them apart.

OUTCOME_PATTERNS = [
    # section 301 of the Commercial Code - THE legal basis for moderation (87% of docs)
    (re.compile(r'§\s*301', re.UNICODE), 4),
    (re.compile(r'(?i)moderačn[éeého]+\s+(oprávneni|práv)', re.UNICODE), 4),

    # court decision verbs (40-60% of docs)
    (re.compile(r'(?i)(súd\s+)?(znížil|priznal|zamietol|zmenil|potvrdil|uložil|zaviazal)', re.UNICODE), 3),
    (re.compile(r'(?i)(znížil|znížiť)\s+(zmluvnú\s+pokut|na\s+sumu)', re.UNICODE), 4),

    # judgment conclusion phrases
    (re.compile(r'(?i)(rozsudok|rozhodnutie)\s+(potvrdil|zmenil|zrušil)', re.UNICODE), 3),
    (re.compile(r'(?i)žalob[eu]\s+(zamietol|v\s+celom\s+rozsahu)', re.UNICODE), 2),
]


# ==========================================
# REASONING patterns (WHY did the court decide that way?)
# ==========================================
# These are the legal arguments the court used. Most important for thesis extraction
# because i need to know what factors the court considered.

REASONING_PATTERNS = [
    # judicial reasoning phrases (53-73% of docs)
    (re.compile(r'(?i)dospel\s+k\s+záveru', re.UNICODE), 3),
    (re.compile(r'(?i)(nestotožnil|nestotožňuje)\s+sa', re.UNICODE), 3),
    (re.compile(r'(?i)(prihliadol|s\s+prihliadnutím)\s+(na|k)', re.UNICODE), 3),
    (re.compile(r'(?i)(považoval|považuje)\s+za\s+(preukázan|primeran|neprimeran)', re.UNICODE), 3),
    (re.compile(r'(?i)s\s+ohľadom\s+na', re.UNICODE), 2),
    (re.compile(r'(?i)je\s+(toho\s+)?názoru', re.UNICODE), 2),

    # proportionality words
    (re.compile(r'(?i)(primeran[áéú]|neprimeran[áéú]|neprimerane\s+vysok)', re.UNICODE), 3),
    (re.compile(r'(?i)zjavne\s+neprimeran', re.UNICODE), 4),
]


# ==========================================
# FACTORS patterns (specific legal standards)
# ==========================================
# These are the exact legal phrases courts use when they evaluate whether a penalty
# is proportionate. I found these by reading through the moderation sections of all
# the court decisions. Some of them appear in almost every case.

FACTORS_PATTERNS = [
    # the big three legal standards (73-80% of docs, super reliable)
    (re.compile(r'(?i)dobr[ýéeých]+\s+mrav', re.UNICODE), 4),
    (re.compile(r'(?i)poctiv[ýéeých]+\s+obchodn[ýéeého]+\s+styk', re.UNICODE), 4),
    (re.compile(r'(?i)hodnot[auúy]\s+a\s+význam[ue]?\s+zabezpečovan', re.UNICODE), 4),

    # specific factors courts weigh (33-53% of docs)
    (re.compile(r'(?i)výšk[auúy]\s+škod', re.UNICODE), 3),
    (re.compile(r'(?i)(vzniknut[áé]\s+)?škod[auúy]', re.UNICODE), 2),
    (re.compile(r'(?i)zabezpečovaci[aeu]?\s+funkci', re.UNICODE), 3),
    (re.compile(r'(?i)(sankčn|reparačn|preventívn)[áéeú]\s+funkci', re.UNICODE), 3),  # sankcna/reparacna/preventivna funkcia pokuty
    (re.compile(r'(?i)kumuláci[auúe]\s+(s\s+)?úrok', re.UNICODE), 3),
    (re.compile(r'(?i)úrok\w*\s+z\s+omeškania', re.UNICODE), 2),
    (re.compile(r'(?i)hospodársk[auúe]\s+pozíci', re.UNICODE), 3),
    (re.compile(r'(?i)pomer\w*\s+(pokut|k\s+istin)', re.UNICODE), 3),
    (re.compile(r'(?i)reciprocit', re.UNICODE), 2),
    (re.compile(r'(?i)test\s+primeranosti', re.UNICODE), 3),  # "test primeranosti" - proportionality test (KS_TT_32Cob_3)
]


# ==========================================
# NOISE patterns (stuff we DONT want)
# ==========================================
# Procedural text that just adds noise. These chunks mention legal terms
# but dont actually contain any extractable information.
# I give them negative weights so they get pushed down in the ranking.

NOISE_PATTERNS = [
    # case costs - almost never useful (87-93% of docs mention it somewhere)
    (re.compile(r'(?i)trov[yoaú]\s+konania', re.UNICODE), -2),
    (re.compile(r'(?i)náhrad[auúy]\s+trov', re.UNICODE), -2),
    (re.compile(r'(?i)bez\s+nariadenia\s+pojednávania', re.UNICODE), -1),
    (re.compile(r'(?i)v\s+zákonnej\s+lehote\s+(podal|podala)', re.UNICODE), -1),

    # dovolanie procedural text - appears in 4/10 docs, always pure procedural boilerplate
    # about admissibility of appeal to the supreme court, never useful for extraction
    (re.compile(r'(?i)prípustnosť\s+dovolania', re.UNICODE), -1),
    (re.compile(r'(?i)dovolac[ýíieé]+\s+(súd|konani)', re.UNICODE), -1),

    # closing boilerplate - "toto rozhodnutie prijal senat najvyssieho sudu"
    (re.compile(r'(?i)toto\s+(rozhodnutie|uznesenie)\s+prijal\s+senát', re.UNICODE), -2),

    # party arguments (not the court's own reasoning)
    # only -1 because sometimes the court adopts what the party said
    (re.compile(r'(?i)(žalovan[ýá]|žalobca|žalobkyňa)\s+(v\s+odvolaní\s+)?(uviedol|namietal|namiet[ao]l|tvrdil)', re.UNICODE), -1),
]


# ==========================================
# WHICH PATTERNS TO USE FOR EACH QUERY TYPE
# ==========================================
# Each query gets its own combination of pattern groups.
# For example q_breach uses BREACH_PATTERNS (positive) + NOISE_PATTERNS (negative).
# q_principal also looks at RATE_PATTERNS because the principal amount is often
# mentioned right next to the penalty rate in the same chunk.

PATTERNS_FOR_QUERY = {
    "q_breach":    [BREACH_PATTERNS, NOISE_PATTERNS],
    "q_rate":      [RATE_PATTERNS, NOISE_PATTERNS],
    "q_principal": [PRINCIPAL_PATTERNS, RATE_PATTERNS, NOISE_PATTERNS],
    "q_outcome":   [OUTCOME_PATTERNS, REASONING_PATTERNS, NOISE_PATTERNS],
    "q_reasoning": [REASONING_PATTERNS, OUTCOME_PATTERNS, NOISE_PATTERNS],
    "q_factors":   [FACTORS_PATTERNS, REASONING_PATTERNS, NOISE_PATTERNS],
}


# ==========================================
# CHUNK_0 BONUS
# ==========================================
# Chunk_0 is the first paragraph of the court decision. It almost always
# contains the basic facts: what contract, who the parties are, what was
# breached. But dense retrieval consistently undervalues it because the
# language sounds procedural ("Napadnutym rozhodnutim sud prvej instancie...")
# not descriptive.
#
# I discovered this problem when i analyzed the MISS cases in experiment_results_details.csv.
# The biggest miss pattern was chunk_0 not making it into top-5 for q_breach queries.
# So i give chunk_0 a bonus for q_breach to fix that.
#
# UPDATE after reviewing all 10 golden docs: the bonus is only justified in 6/10 docs.
# In 4/10 docs chunk_0 is just a verdict summary (amounts, dates) with no contract/breach
# info - the real breach description is in chunk_1. So now the bonus is CONDITIONAL:
# only applied if chunk_0 already matched at least one BREACH_PATTERN (keyword_score > 0).
# This prevents giving a free bonus to verdict-summary chunk_0s that have no breach content.
# Also reduced from 5 to 3 to be more conservative.
#
# For other queries the bonus is 0 or very small because chunk_0 usually doesnt
# contain penalty rates or court reasoning - those are deeper in the document.

FIRST_CHUNK_BONUS = {
    "q_breach":    3,   # chunk_0 is critical for contract context (conditional - only if patterns matched)
    "q_rate":      0,   # penalty rate is never in the first chunk
    "q_principal": 1,   # sometimes the overview mentions the debt amount
    "q_outcome":   1,   # sometimes the intro says what the court decided
    "q_reasoning": 0,   # court reasoning is always deeper in the document
    "q_factors":   0,   # legal factors are always deeper in the document
}


# ==========================================
# SCORING FUNCTION
# ==========================================
# Goes through all patterns for the given query type and adds up the weights
# for each pattern that matches in the chunk text.

def get_keyword_score(text, query_key):
    """
    Score a chunk based on how many legal keywords it contains.

    Returns (score, matched_patterns) where matched_patterns is a dict
    showing which patterns matched and how many times. I use matched_patterns
    for debugging - it helps me understand why a chunk got a high or low score.
    """
    pattern_groups = PATTERNS_FOR_QUERY.get(query_key, [])
    score = 0.0
    matched_patterns = {}

    for group in pattern_groups:
        for pattern, weight in group:
            matches = pattern.findall(text)
            if matches:
                if weight < 0:
                    # FIX: noise penalties should only count once!
                    # I originally multiplied by match count, but that was wrong.
                    # A chunk mentioning "trovy konania" 3 times is the same procedural
                    # section - it shouldnt get 3x the penalty. This was causing chunk_0
                    # (which mentions "nahradu trov" twice in the verdict summary) to get
                    # an unfairly low score and drop out of the top-5 for q_breach.
                    points = weight  # just apply the penalty once
                else:
                    # positive signals: more matches = more relevant, but cap at 3
                    # so a chunk with 20x "zmluvna pokuta" doesnt dominate everything
                    how_many = min(len(matches), 3)
                    points = weight * how_many

                score += points
                matched_patterns[pattern.pattern[:50]] = len(matches)

    return score, matched_patterns


def get_chunk_number(chunk_id):
    """
    Get the chunk number from a chunk ID like 'document.pdf_chunk_5'.
    Returns None if the ID doesnt match the expected format.
    """
    found = re.search(r'_chunk_(\d+)$', chunk_id)
    if found:
        return int(found.group(1))
    return None


# ==========================================
# MAIN RERANKING FUNCTION
# ==========================================

def rerank_chunks(query_key, chunk_texts, chunk_scores, chunk_ids,
                  top_k=5, alpha=0.6, debug=False):
    """
    Rerank retrieved chunks using keyword-based scoring.

    This takes the output of do_retrieve() (which returned more chunks than needed)
    and re-scores each chunk by combining the original retrieval score with a
    keyword-based score. Then it picks the best top_k chunks.

    The formula is:
        final_score = alpha * retrieval_score_normalized + (1 - alpha) * keyword_score_normalized

    alpha controls the balance:
        alpha = 1.0  means only trust retrieval (reranker does nothing)
        alpha = 0.5  means equal weight for both
        alpha = 0.6  means retrieval is more important but keywords can still fix mistakes (default)
        alpha = 0.0  means only trust keywords (ignore retrieval completely)

    I picked alpha=0.6 because the retrieval is already pretty good (95% recall with hybrid),
    and i just want the keywords to fix the last few edge cases where retrieval puts
    procedural chunks above informative ones.

    Returns (reranked_texts, reranked_scores, reranked_ids) with only top_k items.
    """
    if not chunk_texts:
        return [], [], []

    total_chunks = len(chunk_texts)

    # Step 1: score every chunk with keyword patterns
    keyword_scores = []
    all_debug_info = []

    for i in range(total_chunks):
        kw_score, debug_info = get_keyword_score(chunk_texts[i], query_key)

        # add bonus for chunk_0 (first chunk of the document)
        # CONDITIONAL: only apply if chunk_0 already matched at least one positive pattern.
        # This prevents giving a free bonus to verdict-summary chunk_0s that contain
        # no breach/contract info (happened in 4/10 golden dataset docs).
        chunk_number = get_chunk_number(chunk_ids[i]) if i < len(chunk_ids) else None
        if chunk_number == 0 and kw_score > 0:
            bonus = FIRST_CHUNK_BONUS.get(query_key, 0)
            kw_score += bonus
            if bonus > 0:
                debug_info["first_chunk_bonus"] = bonus

        keyword_scores.append(kw_score)
        all_debug_info.append(debug_info)

    # Step 2: normalize both scores to [0, 1] so they are comparable
    # without normalization, retrieval scores are like 0.7-0.9 and keyword
    # scores are like 3-15, so they cant be combined fairly
    def min_max_normalize(scores):
        if not scores:
            return scores
        lowest = min(scores)
        highest = max(scores)
        if highest == lowest:
            # all scores are the same, just give everyone 0.5
            return [0.5] * len(scores)
        return [(s - lowest) / (highest - lowest) for s in scores]

    retrieval_normalized = min_max_normalize(chunk_scores)
    keyword_normalized = min_max_normalize(keyword_scores)

    # Step 3: combine both scores using the alpha weight
    combined_scores = []
    for i in range(total_chunks):
        final = alpha * retrieval_normalized[i] + (1 - alpha) * keyword_normalized[i]
        combined_scores.append(round(final, 6))

    # Step 4: sort by combined score (highest first) and take top_k
    sorted_indices = sorted(range(total_chunks), key=lambda i: combined_scores[i], reverse=True)
    sorted_indices = sorted_indices[:top_k]

    # debug output - helps me see if the reranker is doing what i expect
    if debug:
        print(f"\n  [RERANKER] query={query_key}, alpha={alpha}, pool={total_chunks}, returning top {top_k}")
        for rank, i in enumerate(sorted_indices):
            chunk_num = get_chunk_number(chunk_ids[i]) if i < len(chunk_ids) else "?"
            print(f"    #{rank+1}: chunk_{chunk_num}  "
                  f"retrieval={chunk_scores[i]:.4f} (norm={retrieval_normalized[i]:.3f})  "
                  f"keyword={keyword_scores[i]:.1f} (norm={keyword_normalized[i]:.3f})  "
                  f"combined={combined_scores[i]:.4f}  "
                  f"matched={all_debug_info[i]}")

    # build the output lists
    out_texts = [chunk_texts[i] for i in sorted_indices]
    out_scores = [combined_scores[i] for i in sorted_indices]
    out_ids = [chunk_ids[i] for i in sorted_indices]

    return out_texts, out_scores, out_ids
