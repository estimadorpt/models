"""Shared constants and helpers for municipal election processing."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Mapping, Tuple

# Parties used throughout municipal modelling
TARGET_PARTIES: Tuple[str, ...] = (
    "PS",
    "PSD",
    "CDU",
    "CDS-PP",
    "BE",
    "CH",
    "IL",
    "OTHER",
    "LOCAL_INC",
)

# Coalition keywords that imply a major national party is leading the slate
COALITION_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "PSD": (
        "PSD",
        "PPD/PSD",
        "PPD\\PSD",
        "PPD/PSD.CDS-PP",
        "PSD/CDS",
        "PSD-CDS",
        "PSD.CDS",
        "PSD/CDS-PP",
        "NOVOS TEMPOS",
        "NOVOS TEMPOS LISBOA",
    ),
    "CDS-PP": (
        "CDS-PP",
        "CDS",
        "PPD/PSD.CDS-PP",
        "PSD/CDS",
        "PSD-CDS",
        "PSD.CDS",
        "PSD/CDS-PP",
    ),
    "PS": ("PS", "PS/MPT", r"PS/.+"),
    "IL": ("IL", "INICIATIVA LIBERAL"),
    "BE": ("BE", "BLOCO", "BLOCO DE ESQUERDA"),
    "CDU": ("CDU", "PCP-PEV", "PCP", "PEV"),
    "CH": ("CH", "CHEGA"),
}

# Municipal election dates (source: Comissão Nacional de Eleições)
MUNICIPAL_ELECTION_DATES: Mapping[int, str] = {
    2009: "2009-10-11",
    2013: "2013-09-29",
    2017: "2017-10-01",
    2021: "2021-09-26",
    2025: "2025-10-12",
}

# Heuristic coalition weights to disaggregate AD into PSD/CDS components
AD_DISAGGREGATION_WEIGHTS: Mapping[str, float] = {
    "PSD": 0.85,
    "CDS-PP": 0.15,
}

# Mapping from canonical tokens to target parties
TOKEN_TO_PARTY: Mapping[str, str] = {
    "PS": "PS",
    "PSD": "PSD",
    "CDS": "CDS-PP",
    "CDU": "CDU",
    "BE": "BE",
    "CH": "CH",
    "IL": "IL",
    "PAN": "OTHER",
    "L": "OTHER",
    "LOCAL_INC": "LOCAL_INC",
    "OTHER": "OTHER",
}

# Optional manual weighting for common coalition combinations.
# Keys use sorted tuples of target-party identifiers.
COALITION_SPLIT_WEIGHTS: Mapping[Tuple[str, ...], Mapping[str, float]] = {
    ("PSD", "CDS-PP"): {"PSD": 0.8, "CDS-PP": 0.2},
    ("PSD", "IL"): {"PSD": 0.65, "IL": 0.35},
    ("PSD", "CDS-PP", "IL"): {"PSD": 0.55, "IL": 0.35, "CDS-PP": 0.10},
    ("PSD", "CDS-PP", "IL", "OTHER"): {"PSD": 0.5, "IL": 0.3, "CDS-PP": 0.1, "OTHER": 0.1},
    ("PS", "BE"): {"PS": 0.85, "BE": 0.15},
    ("PS", "BE", "L"): {"PS": 0.75, "BE": 0.15, "OTHER": 0.10},
    ("PS", "L"): {"PS": 0.9, "OTHER": 0.1},
}

# Small constant to avoid log(0) operations
SMALL_CONSTANT = 1e-3


def canonicalize_party_tokens(raw_name: str) -> List[str]:
    """Extract canonical party tokens from an input coalition label."""

    if not raw_name:
        return []

    normalized = raw_name.upper()
    normalized = normalized.replace(" – ", "-").replace("–", "-").replace(" - ", "-")

    replacements = {
        "PPD/PSD": "PSD",
        "PPD\\PSD": "PSD",
        "PPD/PSD.CDS-PP": "PSD;CDS",
        "PPD\\PSD.CDS-PP": "PSD;CDS",
        "PSD/CDS": "PSD;CDS",
        "PSD-CDS": "PSD;CDS",
        "PSD.CDS": "PSD;CDS",
        "PSD/CDS-PP": "PSD;CDS",
        "CDS-PP": "CDS",
        "B.E.": "BE",
        "B.E": "BE",
        "BLOCO DE ESQUERDA": "BE",
        "PCP-PEV": "CDU",
        "PCP": "PCP",
        "PEV": "PEV",
        "CHEGA": "CH",
        "INICIATIVA LIBERAL": "IL",
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    if ";" in normalized:
        raw_tokens = [tok for tok in normalized.split(";") if tok]
    else:
        raw_tokens = [tok for tok in re.split(r"[./_\-\s+]+", normalized) if tok]

    canonical: List[str] = []
    for tok in raw_tokens:
        token = tok.strip()
        if token in {"PS", "PSD", "CDS", "CDU", "BE", "CH", "IL", "OTHER", "LOCAL_INC"}:
            canonical.append(token)
        elif token in {"PCP", "PEV"}:
            canonical.append("CDU")

    return sorted(set(canonical))


def normalize_municipality_name(name: object) -> str:
    """Create a lenient matching key for municipality names."""

    if not isinstance(name, str):
        return ""

    name = re.sub(r"\(.*?\)", "", name)
    normalized = unicodedata.normalize("NFKD", name)
    without_diacritics = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = without_diacritics.lower()
    for token in ("municipio", "municipal", "municipality", "vila", "da", "de", "do"):
        lowered = lowered.replace(token, "")
    return re.sub(r"[^a-z0-9]+", "", lowered)
