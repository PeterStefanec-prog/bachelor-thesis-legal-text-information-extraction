"""
Stage 2: Penalty Index - load penalty index and filter it.

This loads penalty_index.jsonl + embeddings.npy at startup and provides filter function that returns only penalties matching  query.

Key design decisions:
- I filter at PENALTY level, not document level (prevents false cross-penalty matches)
- I use SIMILAR_CONTRACT_TYPES map for near-miss filtering - "dielo" also matches
  "dodavka_sluzieb" because IT projects can be classified as either in practice
- Only contract_type and breach_type are HARD filters
- decision_interest and factor_interest are NOT hard filters - they go to ranker as soft bonus signals.
        Reason: if i hard-filter on decision=awarded_full, analytics shows "100% upheld" which is trivially true and useless.
        I need BOTH awarded and moderated cases for meaningful factor lift computation.
        This was a bug in v1 - 3/5 demo queries produced tautological results.

Index lives in memory. Even at 1000+ docs (~1500 penalties) the JSONL is maybe 3-4 MB and Python filtering takes ~2ms.
No database needed (alsono FAISS - just having embeddings in numpy array in variable and filtering through python loop)
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# #########################################
# SIMILAR CONTRACT TYPES
# #########################################
# when  query LLM says "dodavka_sluzieb" but extraction says "dielo", exact match would miss it
# This map defines which types are close enough.
# i know this from domain knowledge - IT contract can be "dielo" or "dodavka_sluzieb" depending on how the court described it.
# Same for "uver" and "pozicka".

SIMILAR_CONTRACT_TYPES = {
    "dielo": ["dodavka_sluzieb"],
    "dodavka_sluzieb": ["dielo"],
    "uver": ["pozicka"],
    "pozicka": ["uver"],
    # other types are distinct enough (najom, kupna, telekom dont overlap)
}


# #########################################
# PENALTY INDEX CLASS
##########################################

class PenaltyIndex:
    """This class is In-memory index of penalty records with filtering.
    Loads once at pipeline startup, then used for every query.
    """

    def __init__(self, index_dir="data/10_penalty_index"):
        """Load penalty records and embeddings from disk."""
        jsonl_path = os.path.join(index_dir, "penalty_index.jsonl")
        npy_path = os.path.join(index_dir, "embeddings.npy")

        # load all penalty records (one JSON per line)
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # load embeddings - shape [N, 1536], same order as records
        self.embeddings = np.load(npy_path)

        # sanity check - records and embeddings must have same count
        assert len(self.records) == self.embeddings.shape[0], (
            f"Mismatch: {len(self.records)} records but {self.embeddings.shape[0]} embeddings"
        )

        print(f"PenaltyIndex loaded: {len(self.records)} penalties, "
              f"embeddings shape {self.embeddings.shape}")

    def filter(self, intent):
        """Filter penalties based on query intent.

        Only hard-filters on contract_type and breach_type.
        decision_interest and factor_interest are NOT used here - they go to ranker as soft scoring signals instead.

        Returns (filtered_records, filtered_embeddings, filter_metadata).
        """
        ct = intent.get("contract_type")
        bt = intent.get("breach_type")

        # track which filters are active (for transparency in output)
        active_filters = []

        # build list of acceptable contract types (exact + similar types)
        acceptable_types = None
        if ct:                          # for example - dielo
            acceptable_types = set()
            acceptable_types.add(ct)
            # add similar types so we dont miss near-matches
            similar_list = SIMILAR_CONTRACT_TYPES.get(ct, [])       # for example {"dielo", "dodavka_sluzieb"}
            for similar_type in similar_list:
                acceptable_types.add(similar_type)
            active_filters.append(f"contract_type IN {acceptable_types}")

        # go through all records and decide which to keep - single pass
        # mask is True/False per record (used for slicing numpy embeddings)
        # filtered_records collects matching records as we go
        mask = []
        filtered_records = []
        for rec in self.records:
            keep = True

            # contract type filter (with near-miss map)
            if acceptable_types is not None and rec["contract_type"] not in acceptable_types:
                keep = False

            # breach type filter (exact match)
            if keep and bt is not None and rec["breach_type"] != bt:
                keep = False

            # NOTE: decision and factors are NOT filtered here (soft ranking only)
            mask.append(keep)
            if keep:
                filtered_records.append(rec)

        if bt:
            active_filters.append(f"breach_type = {bt}")

        # slice numpy embeddings using boolean mask
        filtered_embeddings = self.embeddings[np.array(mask)]

        # metadata about what we did (for transparency)
        meta = {
            "total_penalties": len(self.records),
            "filtered_count": len(filtered_records),
            "active_filters": active_filters,
            "filter_type": "strict" if active_filters else "none",
        }

        return filtered_records, filtered_embeddings, meta

    def get_all(self):
        """Return all records and embeddings (no filtering).
        Used when query has no structured filters at all.
        """
        meta = {
            "total_penalties": len(self.records),
            "filtered_count": len(self.records),
            "active_filters": [],
            "filter_type": "none",
        }
        return self.records, self.embeddings, meta
