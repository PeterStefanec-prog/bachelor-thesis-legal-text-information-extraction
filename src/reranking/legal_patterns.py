import re


# ########################################
# LEGAL REGEX PATTERNS for slovak court decisions about contractual penalties and moderation paragraph
# ########################################
# These patterns were originally inside reranker.py where i built them by reading
# through 15+ court decisions and noting which phrases appear in "useful" chunks vs "noise" chunks.
# I extracted them here to makes maintenance easier.
#
# Two types of exports:
# 1. pattern GROUPS (BREACH_PATTERNS, RATE_PATTERNS, etc.) - lists of (regex, weight)
#    tuples. Used by reranker.py for query-aware keyword scoring.
# 2. MAPPINGS (PATTERNS_FOR_QUERY, CALL_TO_QUERIES, FIRST_CHUNK_BONUS) - configs
#    used by reranker.py and evaluate_retrieval.py for query-specific behavior.
#
# Also has simple REGEXES (MONEY_REGEX, LEGAL_REF_REGEX, etc.) - not currently used
# in the pipeline but kept for potential use in extraction or future analysis.


# i impolemented 7 pattern groups
# I found these by reading through the moderation sections of all the court decisions and with help of lawyer

# #######################################
# 1. BREACH / CONTRACT CONTEXT patterns
# #######################################
# These help find chunks that describe what the contract was about and what
#  debtor did wrong. See reranker.py for detailed comments about each pattern.

BREACH_PATTERNS = [
    # contract type identification (40-53% of docs)
    (re.compile(r'(?i)zmluv[auúy]\s+o\s+(dielo|spolupráci|poskytovan|kúp|prenájm|obchodnom zastúpen|úver)', re.UNICODE), 3),
    (re.compile(r'(?i)(leasingov|rámcov|abonentn)[áaéeúu]\s+zmluv', re.UNICODE), 3),
    (re.compile(r'(?i)dohod[auúy]\s+o\s+užívan', re.UNICODE), 2),
    # more contract sub-types  (nájom, servis, sprostredkovanie)
    (re.compile(r'(?i)zmluv[auúy]\s+o\s+(nájme|servisn|sprostredkovan)', re.UNICODE), 3),
    (re.compile(r'(?i)kúpn[áaéeú]\s+zmluv', re.UNICODE), 3),

    # breach language (73-93% of docs)
    (re.compile(r'(?i)porušil[aio]?\s+(zmluvnú\s+)?povinnosť', re.UNICODE), 3),
    (re.compile(r'(?i)porušeni[ae]\s+zmluvy', re.UNICODE), 3),
    (re.compile(r'(?i)(nesplnil|neuhradil|nedodržal|nezaplatil|neposkytoval)[aio]?', re.UNICODE), 3),
    (re.compile(r'(?i)v\s+omeškan[ií]', re.UNICODE), 2),
    (re.compile(r'(?i)odstúpil[aio]?\s+(od\s+)?zmluvy', re.UNICODE), 2),
    (re.compile(r'(?i)(konkurenčn[áéú]\s+doložk|zákaz[ue]?\s+konkurencie)', re.UNICODE), 2),
    # new: payment failure patterns from nájom/úver docs
    (re.compile(r'(?i)neplaten[ií][ae]?\s+(nájomného|splátok|leasingov)', re.UNICODE), 3),
    (re.compile(r'(?i)faktúr[yuoa]\s+(neuhradil|nezaplatil)', re.UNICODE), 2),

    # party identification (very common in intro chunks)
    (re.compile(r'(?i)(žalobca|žalobkyňa|žalovan[ýá])\s+(a\s+)?(žalobca|žalovan[ýá])?\s*(uzavreli|uzatvorili|uzavrela)', re.UNICODE), 3),
    (re.compile(r'(?i)(zhotoviteľ|objednávateľ|prenajímateľ|nájomca|dodávateľ|odberateľ)', re.UNICODE), 1),
    (re.compile(r'(?i)(navrhovateľ|odpor(?:ca|kyňa))', re.UNICODE), 1),

    # intro paragraph markers (chunk_0 signals)
    (re.compile(r'(?i)predmetom\s+zmluvy', re.UNICODE), 3),
    # FIX: boosted from 1 to 2. "predmetom konania" appears in summary paragraphs
    # that state what the whole case is about — these are important for context.
    # Before this fix, chunk_33 in eval_09 (43Cob/75/2024) had keyword score=1
    # and was invisible to the reranker after min-max normalization.
    (re.compile(r'(?i)predmetom\s+(konania|sporu)', re.UNICODE), 2),
    (re.compile(r'(?i)na\s+základe\s+(vykonaného\s+)?dokazovania', re.UNICODE), 2),
    (re.compile(r'(?i)(skutkov[ýé]\s+stav|skutkový\s+záver)', re.UNICODE), 1),
    (re.compile(r'(?i)napadnutým\s+rozhodnutím', re.UNICODE), 2),
    (re.compile(r'(?i)súd\s+prvej\s+inštancie', re.UNICODE), 1),
    (re.compile(r'(?i)súd\s+prvého\s+stupňa', re.UNICODE), 1),
]


# #######################################
# 2. PENALTY RATE / AMOUNT patterns
# #######################################
# Find chunks with the actual penalty numbers. "zmluvna pokuta" appears in
# 60-80 chunks per document, but the actual DEFINITION is only in 2-3 chunks.

RATE_PATTERNS = [
    (re.compile(r'(?i)zmluvn[úu]\s+pokut[uú]\s+vo\s+výške', re.UNICODE), 4),
    (re.compile(r'(?i)(povinný|povinná|zaviazal[ai]?\s+sa)\s+zaplatiť', re.UNICODE), 2),
    (re.compile(r'\d+[.,]\d+\s*%', re.UNICODE), 2),
    (re.compile(r'(?i)\d+\s*%\s*(denne|mesačne|ročne|za\s+každý\s+deň)', re.UNICODE), 4),
    (re.compile(r'(?i)z\s+(dlžnej\s+)?sumy', re.UNICODE), 2),
    (re.compile(r'(?i)z\s+(fakturovanej\s+)?cen[yuy]', re.UNICODE), 2),
    (re.compile(r'(?i)vo\s+výške\s+[\d\s.,]+\s*(eur|€|sk)', re.UNICODE), 3),
    (re.compile(r'(?i)\d+-?násobo?k', re.UNICODE), 3),
    (re.compile(r'(?i)(čl\.|článo?k)\s+[IVXLC]+\.?\s*(bod|ods\.?)\s*\d+', re.UNICODE), 2),
    # new: penalty defined as fraction of contract value (nájom, paušál, mesačná odmena)
    (re.compile(r'(?i)(\d+\s*%\s*(zo\s+)?súčtu|mesačn[ýéého]+\s+paušál|mesačn[áéú]\s+(odmena|nájomné))', re.UNICODE), 3),
    # new: penalty linked to contract duration (common in telco/abonentnú contracts)
    (re.compile(r'(?i)do\s+skončenia\s+účinnosti\s+zmluvy', re.UNICODE), 2),
    # new: VZP / general contract conditions references
    (re.compile(r'(?i)(všeobecn[ýéých]+\s+zmluvn[ýéých]+\s+podmienok|VZP)', re.UNICODE), 2),

    # FIX: genitive form of penalty + amount. Slovak courts often write "zmluvnej
    # pokuty dvakrát po 300,- Eur" (genitive) instead of "zmluvnú pokutu vo výške"
    # (accusative). The old patterns only caught the accusative form. This caused
    # chunk_33 in eval_09 (43Cob/75/2024) to be missed entirely — it had keyword
    # score=1 when top chunks had 8+. After min-max normalization it became invisible.
    # FIX: using \w+ instead of character class to match all Slovak suffixes
    # (zmluvnej, zmluvnú, zmluvnou, zmluvných etc. — "j" was missing from [éeúu])
    (re.compile(r'(?i)zmluvn\w+\s+pokut\w+\s+(vo\s+výške\s+)?(\d|dvakrát|trikrát)', re.UNICODE), 3),
    # FIX: catch "300,- Eur" or "756,- €" amounts — very common in Slovak legal text
    # but not caught by the existing patterns (which require "vo výške" before the number)
    (re.compile(r'(?i)\d+,\-?\s*(eur|€)', re.UNICODE), 2),
]


# =#######################################
# 3. PRINCIPAL / SECURED AMOUNT patterns
# #######################################
# The underlying debt that the penalty secures. Sometimes in a different chunk
# than the penalty rate itself.

PRINCIPAL_PATTERNS = [
    (re.compile(r'[\d\s]+[.,]\d{2}\s*(eur|€|sk)', re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'(?i)(istina|dlžná\s+suma|zostatok\s+dlhu|celkov[áé]\s+sum[auy])', re.UNICODE), 3),
    (re.compile(r'(?i)(faktúr[ouya]|cen[auúy]\s+diela|cen[auúy]\s+zákazky)', re.UNICODE), 2),
    (re.compile(r'(?i)(splatnosť|splatn[áé]|lehot[auúy]\s+splatnosti)', re.UNICODE), 2),
    (re.compile(r'(?i)(vyúčtovanie|vyúčtoval)', re.UNICODE), 2),
    (re.compile(r'(?i)(uhradiť|zaplatiť|splatiť)\s+(sumu|čiastku|cenu)', re.UNICODE), 2),
    # new: loan-specific terms (úver, istina+úrok combo)
    (re.compile(r'(?i)úver[u]?\s+vo\s+výške', re.UNICODE), 3),
    (re.compile(r'(?i)(úrok[uy]?\s+(z\s+)?úveru|úrok\w*\s+vo\s+výške)', re.UNICODE), 2),
    # new: nájomné / rent amount — signals the secured obligation value
    (re.compile(r'(?i)nájomné\s+(vo\s+výške\s+)?\d', re.UNICODE), 2),
    # new: contract duration = key for calculating total contract value
    (re.compile(r'(?i)dob[auúy]\s+trvania\s+zmluvy', re.UNICODE), 2),
    (re.compile(r'(?i)(obdobie|po\s+dobu)\s+\d+\s+mesiacov', re.UNICODE), 2),
]


# #######################################
# 4. OUTCOME patterns
# #######################################
# What did the court decide about the penalty?

OUTCOME_PATTERNS = [
    (re.compile(r'§\s*301', re.UNICODE), 4),
    (re.compile(r'(?i)moderačn[éeého]+\s+(oprávneni|práv)', re.UNICODE), 4),
    (re.compile(r'(?i)(súd\s+)?(znížil|priznal|zamietol|zmenil|potvrdil|uložil|zaviazal)', re.UNICODE), 3),
    (re.compile(r'(?i)(znížil|znížiť)\s+(zmluvnú\s+pokut|na\s+sumu)', re.UNICODE), 4),
    (re.compile(r'(?i)(rozsudok|rozhodnutie)\s+(potvrdil|zmenil|zrušil)', re.UNICODE), 3),
    (re.compile(r'(?i)žalob[eu]\s+(zamietol|v\s+celom\s+rozsahu)', re.UNICODE), 2),
]


# #######################################
# 5.  REASONING patterns
# #######################################
#  legal arguments the court used. Most important for thesis extraction.

REASONING_PATTERNS = [
    (re.compile(r'(?i)dospel\s+k\s+záveru', re.UNICODE), 3),
    (re.compile(r'(?i)(nestotožnil|nestotožňuje)\s+sa', re.UNICODE), 3),
    (re.compile(r'(?i)(prihliadol|s\s+prihliadnutím)\s+(na|k)', re.UNICODE), 3),
    (re.compile(r'(?i)(považoval|považuje)\s+za\s+(preukázan|primeran|neprimeran)', re.UNICODE), 3),
    (re.compile(r'(?i)s\s+ohľadom\s+na', re.UNICODE), 2),
    (re.compile(r'(?i)je\s+(toho\s+)?názoru', re.UNICODE), 2),
    (re.compile(r'(?i)(primeran[áéú]|neprimeran[áéú]|neprimerane\s+vysok)', re.UNICODE), 3),
    (re.compile(r'(?i)zjavne\s+neprimeran', re.UNICODE), 4),
]


# #######################################
# 6. FACTORS patterns
# #######################################
# Specific legal standards that courts use to evaluate penalty proportionality (primeranost)


FACTORS_PATTERNS = [
    # the big three legal standards (73-80% of docs, super reliable)
    (re.compile(r'(?i)dobr[ýéeých]+\s+mrav', re.UNICODE), 4),
    (re.compile(r'(?i)poctiv[ýéeých]+\s+obchodn[ýéeého]+\s+styk', re.UNICODE), 4),
    (re.compile(r'(?i)hodnot[auúy]\s+a\s+význam[ue]?\s+zabezpečovan', re.UNICODE), 4),

    # specific factors courts weigh (33-53% of docs)
    (re.compile(r'(?i)výšk[auúy]\s+škod', re.UNICODE), 3),
    (re.compile(r'(?i)(vzniknut[áé]\s+)?škod[auúy]', re.UNICODE), 2),
    (re.compile(r'(?i)zabezpečovaci[aeu]?\s+funkci', re.UNICODE), 3),
    (re.compile(r'(?i)(sankčn|reparačn|preventívn)[áéeú]\s+funkci', re.UNICODE), 3),
    (re.compile(r'(?i)kumuláci[auúe]\s+(s\s+)?úrok', re.UNICODE), 3),
    (re.compile(r'(?i)úrok\w*\s+z\s+omeškania', re.UNICODE), 2),
    (re.compile(r'(?i)hospodársk[auúe]\s+pozíci', re.UNICODE), 3),
    (re.compile(r'(?i)pomer\w*\s+(pokut|k\s+istin)', re.UNICODE), 3),
    (re.compile(r'(?i)reciprocit', re.UNICODE), 2),
    (re.compile(r'(?i)test\s+primeranosti', re.UNICODE), 3),
]


# #######################################
# 7.  NOISE patterns
# #######################################
# Procedural text that just adds noise. Negative weights push these down.

NOISE_PATTERNS = [
    (re.compile(r'(?i)trov[yoaú]\s+konania', re.UNICODE), -2),
    (re.compile(r'(?i)náhrad[auúy]\s+trov', re.UNICODE), -2),
    (re.compile(r'(?i)bez\s+nariadenia\s+pojednávania', re.UNICODE), -1),
    (re.compile(r'(?i)v\s+zákonnej\s+lehote\s+(podal|podala)', re.UNICODE), -1),
    # FIX: reduced from -1 to -0.5. These patterns were penalizing chunks that
    # contain CORE content in NS SR (dovolací) decisions. For example in eval_04
    # (5Obdo/14/2023), chunks 28 and 32 contained both "dovolací súd" AND
    # moderation terms ("primeranosť") but got -1 penalty and were pushed out
    # of the top-k. With -0.5 they still get a mild push-down (valid for KS
    # decisions where dovolaci references are just noise) but the penalty is small
    # enough that strong positive signals can overcome it.
    (re.compile(r'(?i)prípustnosť\s+dovolania', re.UNICODE), -0.5),
    (re.compile(r'(?i)dovolac[ýíieé]+\s+(súd|konani)', re.UNICODE), -0.5),
    (re.compile(r'(?i)toto\s+(rozhodnutie|uznesenie)\s+prijal\s+senát', re.UNICODE), -2),
    (re.compile(r'(?i)(žalovan[ýá]|žalobca|žalobkyňa)\s+(v\s+odvolaní\s+)?(uviedol|namietal|namiet[ao]l|tvrdil)', re.UNICODE), -1),
]




# #######################################
# QUERY -> PATTERN GROUP MAPPING
# #######################################
# Which pattern groups to use for each query type. Used by reranker.py
# for the old flat-chunk keyword scoring.

PATTERNS_FOR_QUERY = {
    "q_breach":    [BREACH_PATTERNS, NOISE_PATTERNS],
    "q_contract":  [BREACH_PATTERNS, PRINCIPAL_PATTERNS, NOISE_PATTERNS],
    # "q_facts":     [BREACH_PATTERNS, PRINCIPAL_PATTERNS, NOISE_PATTERNS],  # removed - did not improve recall
    "q_rate":      [RATE_PATTERNS, NOISE_PATTERNS],
    "q_principal": [PRINCIPAL_PATTERNS, RATE_PATTERNS, NOISE_PATTERNS],
    "q_outcome":   [OUTCOME_PATTERNS, REASONING_PATTERNS, NOISE_PATTERNS],
    "q_reasoning": [REASONING_PATTERNS, OUTCOME_PATTERNS, NOISE_PATTERNS],
    "q_factors":   [FACTORS_PATTERNS, REASONING_PATTERNS, NOISE_PATTERNS],
}


# #######################################
# CHUNK_0 BONUS
# #######################################
# Chunk_0 is the first paragraph of the court decision. It almost always
# contains the basic facts but dense retrieval undervalues it because the
# language sounds procedural. So i give chunk_0 a conditional bonus for
# q_breach (only if it already matched at least one pattern).
# UPDATE: only justified in 6/10 docs. In 4/10 chunk_0 is just a verdict summary.
# So the bonus is CONDITIONAL and reduced from 5 to 3.

FIRST_CHUNK_BONUS = {
    "q_breach":    3,   # chunk_0 is critical for contract context (conditional)
    "q_contract":  3,   # chunk_0 almost always describes the contract (conditional)
    # "q_facts":     3,   # removed - did not improve recall
    "q_rate":      0,   # penalty rate is never in the first chunk
    "q_principal": 1,   # sometimes the overview mentions the debt amount
    "q_outcome":   1,   # sometimes the intro says what the court decided
    "q_reasoning": 0,   # court reasoning is always deeper in the document
    "q_factors":   0,   # legal factors are always deeper in the document
}


# #######################################
# CALL -> QUERY MAPPING
# #######################################
# Which queries belong to which LLM call. Used by evaluate_retrieval.py
# in hierarchical mode to know which queries to run together.

CALL_TO_QUERIES = {
    "call1": ["q_breach", "q_contract", "q_rate", "q_principal"],  # q_facts removed - did not help
    "call2": ["q_outcome", "q_reasoning", "q_factors"],
}





# #######################################
# SIMPLE HELPER REGEXES (not currently used in pipeline)
# #######################################
# General-purpose entity regexes for legal text. Originally used by anchor_detector.py
# for feature extraction. Kept here because MONEY_REGEX, LEGAL_REF_REGEX and
# INVOICE_REGEX are more general than anything in the pattern groups above
# and could be useful for extraction or future analysis.

MONEY_REGEX = re.compile(r'(?i)\b\d[\d\s.]*[.,]\d{2}\s*(eur|€|skk?|czk)?\b|\b\d[\d\s.]*\s*(eur|€|skk?|czk)\b')
PERCENT_REGEX = re.compile(r'\d+(?:[.,]\d+)?\s*%')
LEGAL_REF_REGEX = re.compile(r'§\s*\d+[a-zA-Z]*')
INVOICE_REGEX = re.compile(r'(?i)(faktúr[ayou]?\s*č\.?\s*)?\d{6,}')
ARTICLE_REGEX = re.compile(r'(?i)(čl\.|článo?k)\s*[IVXLC\d]+(?:\.\d+)?')
PARTY_ARGUMENT_REGEX = re.compile(r'(?i)(žalovan[ýá]|žalobca|žalobkyňa).{0,40}(namietal|tvrdil|uviedol|poukázal)')
PROCEDURAL_REGEX = re.compile(r'(?i)(trov[yau]|odvolac[íie]|dovolan|procesn|civilný sporový poriadok|csp|pojednávan)')
