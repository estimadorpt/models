"""Utilities for loading municipal polling data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.municipal_common import (
    AD_DISAGGREGATION_WEIGHTS,
    COALITION_KEYWORDS,
    COALITION_SPLIT_WEIGHTS,
    MUNICIPAL_ELECTION_DATES,
    TARGET_PARTIES,
    TOKEN_TO_PARTY,
    canonicalize_party_tokens,
)


POLL_IGNORE_LABELS = {
    "NÃO VOTAVA",
    "NÃO SABE",
    "NÃO RESPONDE",
    "NS/NR",
    "VOTO BRANCO",
    "VOTARIA EM BRANCO",
    "BRANCOS/NULOS",
    "VOTO NULO",
    "INDECISO",
}


def _ensure_iterable(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NULL", "NAN"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        stripped = text.strip("[]")
        return [item.strip().strip('"\'') for item in stripped.split(',') if item.strip()]
    return [text]


def _normalize_label(label: str) -> str:
    return "".join(ch for ch in str(label).upper() if ch.isalnum())


def _infer_election_year(date: pd.Timestamp, election_dates: Mapping[int, str]) -> int:
    target_date = pd.to_datetime(date)
    sorted_dates: List[Tuple[pd.Timestamp, int]] = sorted(
        (pd.to_datetime(day), year) for year, day in election_dates.items()
    )
    for election_dt, year in sorted_dates:
        if target_date <= election_dt:
            return year
    return sorted_dates[-1][1]


def _map_poll_entry(
    label: str,
    coalition_members: Iterable[str],
    share: float,
    alias_tokens: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    contributions = {party: 0.0 for party in TARGET_PARTIES}

    # Prefer coalition_members or known aliases when available (more reliable than label parsing)
    tokens: List[str] = []
    if alias_tokens:
        tokens = [tok for tok in alias_tokens if tok]
    else:
        members_list = _ensure_iterable(coalition_members)
        if members_list:
            for member in members_list:
                tokens.extend(canonicalize_party_tokens(member))
            tokens = sorted(set(tokens))

    # Fall back to label parsing if coalition_members was empty
    if not tokens:
        tokens = canonicalize_party_tokens(label)

    if not tokens:
        normalized = label.upper()
        for party, keywords in COALITION_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                party_key = TOKEN_TO_PARTY.get(party, party)
                contributions[party_key] += share
                return contributions, [party_key]
        contributions["OTHER"] += share
        return contributions, ["OTHER"]

    mapped_tokens = [TOKEN_TO_PARTY.get(token, token) for token in tokens]
    mapped_tokens = [tok if tok in contributions else "OTHER" for tok in mapped_tokens]

    coalition_key = tuple(sorted(mapped_tokens))
    if coalition_key in COALITION_SPLIT_WEIGHTS:
        weights = COALITION_SPLIT_WEIGHTS[coalition_key]
        total_weight = sum(weights.get(party, 0.0) for party in mapped_tokens)
        if total_weight <= 0:
            total_weight = len(mapped_tokens)
            normalized = {party: 1 / total_weight for party in mapped_tokens}
        else:
            normalized = {party: weights.get(party, 0.0) / total_weight for party in mapped_tokens}
        parties_seen = []
        for party in mapped_tokens:
            portion = share * normalized.get(party, 0.0)
            contributions[party] += portion
            if portion > 0:
                parties_seen.append(party)
        return contributions, parties_seen

    if set(mapped_tokens) == {"PSD", "CDS-PP"}:
        contributions["PSD"] += share * AD_DISAGGREGATION_WEIGHTS.get("PSD", 0.5)
        contributions["CDS-PP"] += share * AD_DISAGGREGATION_WEIGHTS.get("CDS-PP", 0.5)
        return contributions, ["PSD", "CDS-PP"]

    if len(mapped_tokens) == 1:
        party_key = mapped_tokens[0]
        contributions[party_key] += share
        return contributions, [party_key]

    split_value = share / len(mapped_tokens)
    parties_seen: List[str] = []
    for party_key in mapped_tokens:
        contributions[party_key] += split_value
        parties_seen.append(party_key)
    return contributions, parties_seen


def _shares_to_counts(shares: np.ndarray, sample_size: int) -> np.ndarray:
    expected = shares * sample_size
    base = np.floor(expected).astype(int)
    residual = sample_size - base.sum()
    if residual > 0:
        order = np.argsort(expected - base)[::-1]
        for idx in order[:residual]:
            base[idx] += 1
    elif residual < 0:
        order = np.argsort(expected - base)
        for idx in order[: abs(residual)]:
            if base[idx] > 0:
                base[idx] -= 1
    return base


def _extract_municipality_code(row: pd.Series) -> Optional[str]:
    direct = str(row.get("municipality_code_full", "")).strip()
    if direct and direct.upper() != "NONE" and direct != "nan":
        return direct
    codes = _ensure_iterable(row.get("municipality_codes"))
    if codes:
        return codes[0]
    return None


def _extract_municipality_name(row: pd.Series) -> Optional[str]:
    name = str(row.get("municipality_name_official", "")).strip()
    if name and name.upper() != "NONE" and name != "nan":
        return name
    names = _ensure_iterable(row.get("municipality_names"))
    if names:
        return names[0]
    return None


def load_erc_municipal_polls(
    parquet_paths: Sequence[str | Path],
    election_dates: Optional[Mapping[int, str]] = None,
    coalition_aliases: Optional[Mapping[int, Dict[str, Dict[str, List[str]]]]] = None,
) -> pd.DataFrame:
    """Load municipal polls from the ERC parquet exports."""

    if not parquet_paths:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for raw_path in parquet_paths:
        path = Path(raw_path)
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["geographic_scope"].str.lower() == "municipal"].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["fieldwork_end"] = pd.to_datetime(merged["fieldwork_end"], errors="coerce")
    merged = merged[merged["fieldwork_end"].notna()]

    election_dates = election_dates or MUNICIPAL_ELECTION_DATES

    records: List[Dict[str, object]] = []
    group_cols = ["deposit_number", "pollster", "fieldwork_end"]

    for (_, pollster, poll_date), poll_df in merged.groupby(group_cols):
        sample_size_raw = pd.to_numeric(poll_df["sample_size"], errors="coerce").dropna()
        if sample_size_raw.empty:
            continue
        sample_size = int(round(sample_size_raw.iloc[0]))
        if sample_size <= 0:
            continue

        election_year = _infer_election_year(poll_date, election_dates)
        election_day = pd.to_datetime(election_dates[election_year])
        countdown = (election_day - poll_date).days

        # Skip polls taken after the election
        if countdown < 0:
            continue

        poll_df = poll_df.copy()
        poll_df["municipality_code"] = poll_df.apply(_extract_municipality_code, axis=1)
        poll_df["municipality_name"] = poll_df.apply(_extract_municipality_name, axis=1)

        for municipality_code, muni_df in poll_df.groupby("municipality_code"):
            if not municipality_code:
                continue

            municipality_name = muni_df["municipality_name"].dropna().iloc[0] if not muni_df["municipality_name"].dropna().empty else None
            district_series = muni_df["district_name"].dropna()
            district_name = district_series.iloc[0] if not district_series.empty else None

            alias_lookup: Dict[str, List[str]] = {}
            if coalition_aliases:
                aliases_for_year = coalition_aliases.get(election_year, {})
                if municipality_code in aliases_for_year:
                    alias_lookup = {
                        _normalize_label(alias): tokens
                        for alias, tokens in aliases_for_year[municipality_code].items()
                    }

            party_support = {party: 0.0 for party in TARGET_PARTIES}
            availability = {party: 0 for party in TARGET_PARTIES}

            raw_total_share = 0.0
            for _, entry in muni_df.iterrows():
                party_label = str(entry.get("party", "")).strip()
                if not party_label:
                    continue
                normalized_label = party_label.upper()
                if normalized_label in POLL_IGNORE_LABELS:
                    continue
                pct = pd.to_numeric(entry.get("vote_intention_pct"), errors="coerce")
                if pd.isna(pct) or pct < 0:
                    continue
                share = float(pct) / 100.0
                if share <= 0:
                    continue

                raw_total_share += share

                coalition_members = entry.get("coalition_members")
                alias_tokens = None
                if alias_lookup:
                    alias_tokens = alias_lookup.get(_normalize_label(party_label))
                mapped, parties_seen = _map_poll_entry(
                    party_label,
                    coalition_members,
                    share,
                    alias_tokens=alias_tokens,
                )
                for party, portion in mapped.items():
                    party_support[party] += portion
                for party in parties_seen:
                    availability[party] = 1

            total_share = sum(party_support.values())
            if total_share <= 0:
                continue

            # Calculate effective sample size based on response rate
            # If raw_total_share < 1.0, many respondents were undecided/non-response
            response_rate = min(raw_total_share, 1.0)
            effective_sample = int(round(sample_size * response_rate))
            if effective_sample < 50:  # Minimum threshold for reliable estimates
                continue

            # Normalize to valid probabilities (but track that we're doing this)
            for party in party_support:
                party_support[party] = party_support[party] / total_share

            shares_array = np.array([party_support[party] for party in TARGET_PARTIES], dtype=float)
            counts = _shares_to_counts(shares_array, effective_sample)

            record: Dict[str, object] = {
                "pollster": pollster,
                "poll_date": poll_date,
                "sample_size": sample_size,
                "effective_sample": effective_sample,
                "response_rate": response_rate,
                "municipality_code": municipality_code,
                "municipality_name": municipality_name,
                "district_name": district_name,
                "election_year": election_year,
                "countdown": countdown,
                "total_votes": effective_sample,
                "is_poll": 1,
            }

            for party, share_value, count_value in zip(TARGET_PARTIES, shares_array, counts):
                record[f"share_{party}"] = share_value
                record[f"votes_{party}"] = int(count_value)
                record[f"available_{party}"] = int(availability.get(party, 0))
                record[f"new_{party}"] = 0

            record.update(
                {
                    "local_list_flag": 0,
                    "incumbent_local_flag": 0,
                    "incumbent_prev_share": 0.0,
                }
            )

            for party in TARGET_PARTIES:
                record[f"incumbent_{party}"] = 0

            records.append(record)

    if not records:
        return pd.DataFrame()

    polls = pd.DataFrame.from_records(records)
    polls = polls.sort_values(["election_year", "municipality_code", "poll_date"])
    return polls.reset_index(drop=True)
