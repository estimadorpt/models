"""Municipal coupling dataset preparation utilities.

Implements the data requirements for the national-municipal coupling model
specified in issue #38.

Key responsibilities:
- Load municipal election results from the autárquicas parquet files
- Map coalition columns onto the target party structure
- Compute municipality-level historical baselines in centered log-ratio space
- Extract national-level PyMC latent signals for each election date
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Set

import arviz as az
import numpy as np
import pandas as pd
import unicodedata

from src.config import DATA_DIR
from src.data.municipal_polls import load_erc_municipal_polls
from src.data.municipal_common import COALITION_SPLIT_WEIGHTS


# Parties used throughout the coupling model
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
    "PSD": ("PSD", "PPD/PSD", "PPD\\PSD", "PPD/PSD.CDS-PP", "PSD/CDS", "PSD-CDS", "PSD.CDS", "PSD/CDS-PP"),
    "CDS-PP": ("CDS-PP", "CDS", "PPD/PSD.CDS-PP", "PSD/CDS", "PSD-CDS", "PSD.CDS", "PSD/CDS-PP"),
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

# Columns that should be treated as metadata (not vote columns)
METADATA_COLUMNS: Tuple[str, ...] = (
    "territory_code",
    "territory_name",
    "number_voters",
    "percentage_voters",
    "subscribed_voters",
    "null_votes",
    "blank_votes",
    "election_type",
)

# Small constant to avoid log(0) operations
SMALL_CONSTANT = 1e-3


@dataclass
class MunicipalCouplingDataset:
    """Container with all arrays required by the coupling model."""

    results: pd.DataFrame
    metadata: pd.DataFrame
    baseline_clr: pd.DataFrame
    national_clr: pd.DataFrame
    district_offsets: pd.DataFrame
    donor_weights: pd.DataFrame

    def get_training_splits(
        self,
        train_years: Sequence[int],
        test_year: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split results into training/test partitions by election year."""
        train_mask = self.results["election_year"].isin(train_years)
        test_mask = self.results["election_year"] == test_year
        return self.results.loc[train_mask], self.results.loc[test_mask]


def _canonical_tokens(raw_name: str) -> List[str]:
    """Extract canonical party tokens from a coalition column name."""
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

    if ';' in normalized:
        raw_tokens = [tok for tok in normalized.split(';') if tok]
    else:
        raw_tokens = [tok for tok in re.split(r"[./_\-\s+]+", normalized) if tok]

    canonical: List[str] = []
    for tok in raw_tokens:
        tok = tok.strip()
        if tok in {"PS", "PSD", "CDS", "CDU", "BE", "CH", "IL"}:
            canonical.append(tok)
        elif tok in {"PCP", "PEV"}:
            canonical.append("CDU")
    return sorted(set(canonical))


def _map_votes_to_target(
    row: pd.Series,
    vote_columns: Sequence[str],
    column_mapping: Dict[str, List[str]] | None = None,
) -> Tuple[Dict[str, float], Set[str]]:
    """Map coalition vote columns to the target party structure for a municipality."""
    party_totals = {party: 0.0 for party in TARGET_PARTIES}
    parties_seen: Set[str] = set()

    for column in vote_columns:
        value = row[column]
        if pd.isna(value) or value <= 0:
            continue

        tokens = None
        column_key = str(column).strip()

        if column_mapping:
            tokens = column_mapping.get(column_key)

        if tokens is None:
            if column_key not in COLUMN_CACHE:
                COLUMN_CACHE[column_key] = _canonical_tokens(column_key)
            tokens = COLUMN_CACHE[column_key]

        party_tokens = tokens
        if not party_tokens:
            normalized_label = column_key.upper()
            assigned = False
            for party, keywords in COALITION_KEYWORDS.items():
                if any(keyword in normalized_label for keyword in keywords):
                    party_key = TOKEN_TO_PARTY.get(party, party)
                    party_totals[party_key] += float(value)
                    parties_seen.add(party_key)
                    assigned = True
                    break
            if assigned:
                continue
            party_totals["OTHER"] += float(value)
            parties_seen.add("OTHER")
            continue

        mapped_tokens = [TOKEN_TO_PARTY.get(token, token) for token in party_tokens]
        mapped_tokens = [tok if tok in party_totals else "OTHER" for tok in mapped_tokens]

        coalition_key = tuple(sorted(mapped_tokens))
        if coalition_key in COALITION_SPLIT_WEIGHTS:
            weights = COALITION_SPLIT_WEIGHTS[coalition_key]
            total_weight = sum(weights.get(party, 0.0) for party in mapped_tokens)
            if total_weight <= 0:
                total_weight = len(mapped_tokens)
                normalized = {party: 1 / total_weight for party in mapped_tokens}
            else:
                normalized = {party: weights.get(party, 0.0) / total_weight for party in mapped_tokens}
            for party in mapped_tokens:
                portion = float(value) * normalized.get(party, 0.0)
                party_totals[party] += portion
                if portion > 0:
                    parties_seen.add(party)
            continue

        if set(mapped_tokens) == {"PSD", "CDS-PP"}:
            party_totals["PSD"] += float(value) * AD_DISAGGREGATION_WEIGHTS.get("PSD", 0.5)
            party_totals["CDS-PP"] += float(value) * AD_DISAGGREGATION_WEIGHTS.get("CDS-PP", 0.5)
            parties_seen.update({"PSD", "CDS-PP"})
            continue

        if len(mapped_tokens) == 1:
            party_key = mapped_tokens[0]
            party_totals[party_key] += float(value)
            parties_seen.add(party_key)
        else:
            split_value = float(value) / len(mapped_tokens)
            for party_key in mapped_tokens:
                party_totals[party_key] += split_value
                parties_seen.add(party_key)

    return party_totals, parties_seen


def _load_coalition_mappings(
    election_years: Sequence[int],
    data_dir: str,
) -> Tuple[
    Dict[int, Dict[str, Dict[str, List[str]]]],
    Dict[int, Dict[str, bool]],
    Dict[int, Dict[str, List[str]]],
]:
    """Load coalition definitions, local-list flags, and incumbent mappings."""
    component_mappings: Dict[int, Dict[str, Dict[str, List[str]]]] = {}
    local_flags: Dict[int, Dict[str, bool]] = {}
    incumbent_mappings: Dict[int, Dict[str, List[str]]] = {}

    for year in election_years:
        try:
            year_int = int(year)
        except ValueError:
            continue
        parquet_path = Path(data_dir) / f"municipal_coalitions_{year_int}.parquet"
        mapping_for_year: Dict[str, Dict[str, List[str]]] = {}
        local_flags_for_year: Dict[str, bool] = {}
        incumbent_for_year: Dict[str, List[str]] = {}

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            for muni_code, muni_df in df.groupby("municipality_code"):
                muni_code = str(muni_code).strip()
                if not muni_code:
                    continue

                incumbent_tokens: Optional[Set[str]] = None
                for _, row in muni_df.iterrows():
                    list_name = str(row.get("party", "")).strip()
                    if not list_name:
                        continue

                    raw_components = str(row.get("coalition_parties", "") or "").strip()
                    components = [part.strip() for part in raw_components.split(';') if part.strip()]

                    tokens: List[str] = []
                    if components:
                        for component in components:
                            tokens.extend(_canonical_tokens(component))
                    else:
                        tokens.extend(_canonical_tokens(list_name))

                    tokens = [tok for tok in tokens if tok != ""]

                    is_incumbent = bool(row.get("is_incumbent", False))
                    incumbent_party_raw = str(row.get("incumbent_party", "")).strip()

                    if not tokens:
                        tokens = ["LOCAL_INC"] if is_incumbent else ["OTHER"]

                    if is_incumbent and all(tok not in TARGET_PARTIES for tok in tokens):
                        tokens = ["LOCAL_INC"]

                    if tokens == ["OTHER"] and is_incumbent:
                        tokens = ["LOCAL_INC"]

                    mapping_for_year.setdefault(muni_code, {})[list_name] = sorted(set(tokens))

                    if any(tok == "LOCAL_INC" for tok in tokens):
                        local_flags_for_year[muni_code] = True
                    elif tokens == ["OTHER"]:
                        local_flags_for_year.setdefault(muni_code, False)

                    inc_tokens = set()
                    if is_incumbent:
                        inc_tokens.update(tokens)
                    elif incumbent_party_raw:
                        inc_tokens.update(_canonical_tokens(incumbent_party_raw))

                    if not inc_tokens and is_incumbent:
                        inc_tokens.add("LOCAL_INC")

                    if inc_tokens:
                        if "LOCAL_INC" in inc_tokens:
                            local_flags_for_year[muni_code] = True
                        if incumbent_tokens is None:
                            incumbent_tokens = set()
                        incumbent_tokens.update(inc_tokens)

                if incumbent_tokens:
                    incumbent_for_year[muni_code] = sorted(set(incumbent_tokens))

        component_mappings[year_int] = mapping_for_year
        local_flags[year_int] = local_flags_for_year
        incumbent_mappings[year_int] = incumbent_for_year

    return component_mappings, local_flags, incumbent_mappings


def _build_future_results_placeholder(
    year: int,
    parties: Sequence[str],
    data_dir: str,
    mapping_for_year: Mapping[str, Dict[str, List[str]]],
    local_flags_for_year: Mapping[str, bool],
    incumbent_for_year: Mapping[str, Sequence[str]],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    """Construct synthetic rows for a future election without official results."""

    coalition_path = Path(data_dir) / f"municipal_coalitions_{year}.parquet"
    if not coalition_path.exists():
        raise FileNotFoundError(
            f"Missing municipal coalition data for {year}; required for forecasting ({coalition_path})."
        )

    coalition_df = pd.read_parquet(coalition_path)
    if coalition_df.empty:
        raise ValueError(
            f"Municipal coalition data for {year} is empty; cannot build forecast placeholders."
        )

    placeholder_records: List[Dict[str, object]] = []
    metadata_entries: Dict[str, Dict[str, object]] = {}

    for muni_code, muni_df in coalition_df.groupby("municipality_code"):
        muni_code = str(muni_code).strip()
        if not muni_code:
            continue

        municipality_name = str(muni_df["municipality"].iloc[0]).strip()
        metadata_entries.setdefault(
            muni_code,
            {
                "municipality_code": muni_code,
                "municipality_name": municipality_name,
            },
        )

        tokens_seen: Set[str] = set()
        for tokens in mapping_for_year.get(muni_code, {}).values():
            tokens_seen.update(tokens)

        incumbents = set(incumbent_for_year.get(muni_code, []))

        row: Dict[str, object] = {
            "election_year": year,
            "municipality_code": muni_code,
            "municipality_name": municipality_name,
            "total_votes": 0.0,
        }

        for party in parties:
            row[f"votes_{party}"] = 0.0
            row[f"available_{party}"] = 1 if party in tokens_seen else 0
            row[f"incumbent_{party}"] = 1 if party in incumbents else 0

        row["local_list_flag"] = 1 if local_flags_for_year.get(muni_code, False) else 0
        row["incumbent_local_flag"] = 1 if "LOCAL_INC" in incumbents else 0

        placeholder_records.append(row)

    return placeholder_records, metadata_entries


def load_municipal_results(
    election_years: Sequence[int],
    parties: Sequence[str] = TARGET_PARTIES,
    data_dir: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and harmonise municipal election results for the requested years."""
    data_dir = data_dir or DATA_DIR

    coalition_mappings, local_flags_map, incumbent_map = _load_coalition_mappings(election_years, data_dir)

    records: List[Dict[str, object]] = []
    metadata_rows: Dict[str, Dict[str, object]] = {}

    for year in election_years:
        parquet_path = Path(data_dir) / f"autarquicas_{year}.parquet"
        mapping_for_year = coalition_mappings.get(int(year), {})
        local_flags_for_year = local_flags_map.get(int(year), {})
        incumbent_for_year = incumbent_map.get(int(year), {})

        try:
            df = pd.read_parquet(parquet_path)
        except FileNotFoundError:
            placeholder_records, placeholder_metadata = _build_future_results_placeholder(
                int(year),
                parties,
                data_dir,
                mapping_for_year,
                local_flags_for_year,
                incumbent_for_year,
            )
            records.extend(placeholder_records)
            for muni_code, meta in placeholder_metadata.items():
                metadata_rows.setdefault(muni_code, meta)
            continue

        vote_columns = [col for col in df.columns if col not in METADATA_COLUMNS]

        for column in vote_columns:
            if column not in COLUMN_CACHE:
                COLUMN_CACHE[column] = _canonical_tokens(column)

        for _, row in df.iterrows():
            municipality_code = str(row["territory_code"])
            column_mapping = mapping_for_year.get(municipality_code, {})
            party_totals, parties_seen = _map_votes_to_target(row, vote_columns, column_mapping)
            result_row = {
                "election_year": year,
                "municipality_code": municipality_code,
                "municipality_name": row["territory_name"],
                "total_votes": sum(party_totals.values()),
            }
            for party in parties:
                result_row[f"votes_{party}"] = party_totals.get(party, 0.0)
                result_row[f"available_{party}"] = 1 if party in parties_seen else 0
            result_row["local_list_flag"] = 1 if local_flags_for_year.get(municipality_code, False) else 0

            incumbent_tokens = incumbent_for_year.get(municipality_code, [])
            for party in parties:
                result_row[f"incumbent_{party}"] = 1 if party in incumbent_tokens else 0
            result_row["incumbent_local_flag"] = 1 if "LOCAL_INC" in incumbent_tokens else 0
            records.append(result_row)

            metadata_rows.setdefault(
                row["territory_code"],
                {
                    "municipality_code": row["territory_code"],
                    "municipality_name": row["territory_name"],
                },
            )

    results_df = pd.DataFrame.from_records(records)
    results_df = _augment_with_previous_results(results_df, parties)

    metadata_df = pd.DataFrame.from_records(list(metadata_rows.values()))
    metadata_df = _attach_district_metadata(metadata_df, data_dir=data_dir)

    return results_df, metadata_df


def _attach_district_metadata(metadata: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Augment metadata with district names using the coalition parquet file."""
    mapping_path = Path(data_dir) / "municipality_district_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        district_lookup = mapping.set_index("municipality_code")
        metadata = metadata.merge(
            district_lookup[["district_name", "region"]],
            how="left",
            left_on="municipality_code",
            right_index=True,
        )
        metadata["district_name"] = metadata["district_name"].fillna("UNKNOWN")
        metadata["region"] = metadata["region"].fillna("UNKNOWN")
        return metadata

    metadata["district_name"] = "UNKNOWN"
    metadata["region"] = "UNKNOWN"
    return metadata


def _normalize_name(name: object) -> str:
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


def _augment_with_previous_results(
    results: pd.DataFrame,
    parties: Sequence[str],
) -> pd.DataFrame:
    """Add previous-election share features and incumbency context."""

    if results.empty:
        return results

    results = results.sort_values(["municipality_code", "election_year"]).reset_index(drop=True)

    vote_columns = [f"votes_{party}" for party in parties]
    share_columns = [f"share_{party}" for party in parties]

    shares = results[vote_columns].div(results["total_votes"].replace(0, np.nan), axis=0).fillna(0.0)
    shares.columns = share_columns
    for column in share_columns:
        results[column] = shares[column]

    for party, share_col in zip(parties, share_columns):
        results[f"prev_share_{party}"] = (
            results.groupby("municipality_code")[share_col].shift(1).fillna(0.0)
        )

    for party in parties:
        available_col = f"available_{party}"
        prev_col = f"prev_share_{party}"
        new_col = f"new_{party}"
        if available_col in results.columns and prev_col in results.columns:
            results[new_col] = (
                (results[available_col] == 1) & (results[prev_col] <= 1e-6)
            ).astype(int)
        else:
            results[new_col] = 0

    results["winner_party"] = shares.idxmax(axis=1)
    results["winner_share"] = shares.max(axis=1)
    results["prev_winner_party"] = (
        results.groupby("municipality_code")["winner_party"].shift(1)
    )
    results["prev_winner_share"] = (
        results.groupby("municipality_code")["winner_share"].shift(1).fillna(0.0)
    )

    prev_share_columns = {party: f"prev_share_{party}" for party in parties}

    def _compute_incumbent_prev_share(row: pd.Series) -> float:
        incumbent_share = 0.0
        for party in parties:
            if row.get(f"incumbent_{party}", 0):
                incumbent_share = max(incumbent_share, float(row[prev_share_columns[party]]))
        if incumbent_share == 0.0 and row.get("incumbent_local_flag", 0):
            incumbent_share = float(row.get("prev_winner_share", 0.0))
        return float(incumbent_share)

    results["incumbent_prev_share"] = results.apply(_compute_incumbent_prev_share, axis=1)

    return results


def _estimate_local_donor_weights(
    results: pd.DataFrame,
    metadata: pd.DataFrame,
    baseline_clr: pd.DataFrame,
    parties: Sequence[str],
) -> pd.DataFrame:
    """Infer how local incumbents redistribute votes when they disappear."""

    if results.empty:
        return pd.DataFrame(columns=parties)

    metadata_indexed = metadata.set_index("municipality_code")
    local_index = parties.index("LOCAL_INC") if "LOCAL_INC" in parties else -1

    baseline_probs: Dict[str, np.ndarray] = {}
    if not baseline_clr.empty:
        for municipality_code, row in baseline_clr.iterrows():
            clr_vector = row[[f"clr_{party}" for party in parties]].to_numpy(dtype=float)
            probs = _softmax(clr_vector)
            baseline_probs[municipality_code] = probs

    donor_sums: Dict[str, np.ndarray] = {}
    donor_counts: Dict[str, int] = {}
    district_sums: Mapping[str, np.ndarray] = defaultdict(lambda: np.zeros(len(parties), dtype=float))
    district_counts: Mapping[str, int] = defaultdict(int)
    global_sum = np.zeros(len(parties), dtype=float)
    global_count = 0

    sorted_results = results.sort_values(["municipality_code", "election_year"])
    for municipality_code, muni_df in sorted_results.groupby("municipality_code"):
        accumulator = np.zeros(len(parties), dtype=float)
        count = 0
        if municipality_code in metadata_indexed.index:
            district_name = metadata_indexed.at[municipality_code, "district_name"]
        else:
            district_name = "UNKNOWN"
        for _, row in muni_df.iterrows():
            if row.get("available_LOCAL_INC", 0) != 1:
                continue
            prev = np.array([row.get(f"prev_share_{party}", 0.0) for party in parties], dtype=float)
            curr = np.array([row.get(f"share_{party}", 0.0) for party in parties], dtype=float)
            diff = prev - curr
            diff[diff < 0] = 0.0
            if local_index >= 0:
                diff[local_index] = 0.0
            total = diff.sum()
            if total <= 1e-6:
                continue
            weight_vec = diff / total
            accumulator += weight_vec
            count += 1
            district_sums[district_name] += weight_vec
            district_counts[district_name] += 1
            global_sum += weight_vec
            global_count += 1
        donor_sums[municipality_code] = accumulator
        donor_counts[municipality_code] = count

    donor_rows: List[List[object]] = []
    for municipality_code in metadata["municipality_code"].unique():
        accumulator = donor_sums.get(municipality_code, np.zeros(len(parties), dtype=float))
        count = donor_counts.get(municipality_code, 0)
        if count > 0:
            weight = accumulator / count
        else:
            if municipality_code in metadata_indexed.index:
                district_name = metadata_indexed.at[municipality_code, "district_name"]
            else:
                district_name = "UNKNOWN"
            district_count = district_counts.get(district_name, 0)
            if district_count > 0:
                weight = district_sums[district_name] / district_count
            elif global_count > 0:
                weight = global_sum / global_count
            else:
                weight = np.zeros(len(parties), dtype=float)
        if local_index >= 0:
            weight[local_index] = 0.0
        weight = np.clip(weight, 0.0, None)
        total = weight.sum()
        if total <= 1e-8:
            baseline = baseline_probs.get(municipality_code)
            if baseline is not None:
                fallback = baseline.copy()
            else:
                fallback = np.ones(len(parties), dtype=float)
            if local_index >= 0:
                fallback[local_index] = 0.0
            fallback_total = fallback.sum()
            if fallback_total > 0:
                weight = fallback / fallback_total
            else:
                weight = np.zeros(len(parties), dtype=float)
        else:
            weight = weight / total
        donor_rows.append([municipality_code, *weight.tolist()])

    donor_df = pd.DataFrame(donor_rows, columns=["municipality_code", *parties])
    donor_df = donor_df.set_index("municipality_code")
    return donor_df


def compute_baseline_clr(
    results: pd.DataFrame,
    parties: Sequence[str] = TARGET_PARTIES,
    training_years: Sequence[int] | None = None,
    small_constant: float = SMALL_CONSTANT,
    decay: float = 0.5,
) -> pd.DataFrame:
    """Compute municipality-party historical baselines in centered log-ratio space."""
    if training_years is not None:
        baseline_source = results[results["election_year"].isin(training_years)].copy()
    else:
        baseline_source = results.copy()

    if baseline_source.empty:
        return pd.DataFrame(columns=[f"clr_{party}" for party in parties])

    if training_years is not None and training_years:
        max_year = max(training_years)
        weight_map = {year: decay ** (max_year - year) for year in training_years}
    else:
        unique_years = baseline_source["election_year"].unique().tolist()
        max_year = max(unique_years)
        weight_map = {year: decay ** (max_year - year) for year in unique_years}

    baseline_source = baseline_source.copy()
    baseline_source["_weight"] = baseline_source["election_year"].map(weight_map).fillna(1.0)

    vote_columns = [f"votes_{party}" for party in parties]
    share_df = baseline_source[vote_columns].div(
        baseline_source["total_votes"].replace(0, np.nan), axis=0
    ).fillna(0.0)

    weighted_shares = share_df.multiply(baseline_source["_weight"], axis=0)
    grouped = weighted_shares.groupby(baseline_source["municipality_code"]).sum()
    weight_sums = baseline_source.groupby("municipality_code")["_weight"].sum()
    shares = grouped.div(weight_sums, axis=0)
    shares = shares.fillna(0) + small_constant

    log_shares = np.log(shares)
    centered = log_shares.sub(log_shares.mean(axis=1), axis=0)

    centered.index.name = "municipality_code"
    rename_map = {col: col.replace("votes_", "clr_") for col in centered.columns}
    return centered.rename(columns=rename_map)


def load_national_signal_clr(
    trace_path: str,
    election_years: Sequence[int],
    parties: Sequence[str] = TARGET_PARTIES,
    ad_weights: Mapping[str, float] | None = None,
    small_constant: float = SMALL_CONSTANT,
) -> pd.DataFrame:
    """Extract national-level latent logits for each election in CLR space."""
    ad_weights = ad_weights or AD_DISAGGREGATION_WEIGHTS

    idata = az.from_zarr(trace_path)
    national_trend = idata.posterior["national_trend_pt"].mean(dim=["chain", "draw"])

    calendar_index = pd.to_datetime(national_trend.coords["calendar_time"].values)
    parties_complete = list(national_trend.coords["parties_complete"].values)

    trend_array = national_trend.to_numpy()

    national_df = pd.DataFrame(trend_array, index=calendar_index, columns=parties_complete)

    records: List[Dict[str, object]] = []
    for year in election_years:
        target_date = pd.to_datetime(MUNICIPAL_ELECTION_DATES[year])
        closest_idx = national_df.index.get_indexer([target_date], method="nearest")[0]
        logits = national_df.iloc[closest_idx].to_numpy()
        shares = _softmax(logits)
        shares_map = dict(zip(parties_complete, shares))

        mapped_shares = {
            "PS": shares_map.get("PS", 0.0),
            "CDU": shares_map.get("CDU", 0.0),
            "BE": shares_map.get("BE", 0.0),
            "CH": shares_map.get("CH", 0.0),
            "IL": shares_map.get("IL", 0.0),
        }

        ad_share = shares_map.get("AD", 0.0)
        mapped_shares["PSD"] = ad_share * ad_weights.get("PSD", 0.0)
        mapped_shares["CDS-PP"] = ad_share * ad_weights.get("CDS-PP", 0.0)
        mapped_shares["OTHER"] = shares_map.get("PAN", 0.0) + shares_map.get("L", 0.0)
        mapped_shares["LOCAL_INC"] = 0.0

        ordered_shares = np.array([mapped_shares[party] for party in parties]) + small_constant
        log_shares = np.log(ordered_shares)
        clr_vector = log_shares - log_shares.mean()

        record = {"election_year": year}
        for i, party in enumerate(parties):
            record[f"clr_{party}"] = clr_vector[i]
        records.append(record)

    return pd.DataFrame.from_records(records)


def load_district_offsets(
    trace_path: str,
    parties: Sequence[str] = TARGET_PARTIES,
    ad_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    ad_weights = ad_weights or AD_DISAGGREGATION_WEIGHTS
    try:
        idata = az.from_zarr(trace_path)
    except Exception:
        return pd.DataFrame(columns=parties)

    if "static_district_offset" not in idata.posterior:
        return pd.DataFrame(columns=parties)

    offsets = idata.posterior["static_district_offset"].mean(dim=["chain", "draw"])
    offset_df = offsets.to_pandas()
    offset_df.index.name = "district_name"

    result = pd.DataFrame(index=offset_df.index, columns=parties, dtype=float)
    result.loc[:, :] = 0.0

    if "PS" in offset_df.columns:
        result["PS"] = offset_df["PS"].values
    if "CH" in offset_df.columns:
        result["CH"] = offset_df["CH"].values
    if "IL" in offset_df.columns:
        result["IL"] = offset_df["IL"].values
    if "BE" in offset_df.columns:
        result["BE"] = offset_df["BE"].values
    if "CDU" in offset_df.columns:
        result["CDU"] = offset_df["CDU"].values
    if "PAN" in offset_df.columns and "L" in offset_df.columns:
        result["OTHER"] = offset_df["PAN"].values + offset_df["L"].values
    elif "PAN" in offset_df.columns:
        result["OTHER"] = offset_df["PAN"].values
    if "AD" in offset_df.columns:
        ad_vals = offset_df["AD"].values
        result["PSD"] = ad_vals * ad_weights.get("PSD", 0.0)
        result["CDS-PP"] = ad_vals * ad_weights.get("CDS-PP", 0.0)
    result["LOCAL_INC"] = 0.0

    result = result.fillna(0.0)
    return result


def build_municipal_coupling_dataset(
    election_years: Sequence[int],
    trace_path: str,
    train_years: Sequence[int] | None = None,
    poll_paths: Sequence[str] | None = None,
    data_dir: str | None = None,
) -> MunicipalCouplingDataset:
    """High-level helper that assembles all inputs required by the PyMC model."""
    coalition_aliases, _, _ = _load_coalition_mappings(election_years, data_dir or DATA_DIR)
    results_only, metadata = load_municipal_results(election_years, data_dir=data_dir)
    results_only = results_only.copy()
    results_only["is_poll"] = 0

    polls_df = pd.DataFrame()
    if poll_paths:
        polls_df = load_erc_municipal_polls(
            poll_paths,
            election_dates=MUNICIPAL_ELECTION_DATES,
            coalition_aliases=coalition_aliases,
        )

    combined_results = results_only
    if not polls_df.empty:
        polls_df = polls_df.copy()
        polls_df["is_poll"] = 1
        polls_df["municipality_code"] = polls_df["municipality_code"].astype(str)
        polls_df["municipality_name"] = polls_df["municipality_name"].fillna("")
        if "district_name" in polls_df.columns:
            polls_df["district_name"] = polls_df["district_name"].fillna("UNKNOWN")
        else:
            polls_df["district_name"] = "UNKNOWN"
        if "region" in polls_df.columns:
            polls_df["region"] = polls_df["region"].fillna("UNKNOWN")
        else:
            polls_df["region"] = "UNKNOWN"
        polls_df["total_votes"] = pd.to_numeric(polls_df["total_votes"], errors="coerce").fillna(0.0)
        polls_df["election_year"] = pd.to_numeric(polls_df["election_year"], errors="coerce").astype(int)

        required_columns = set(results_only.columns)
        poll_only_columns = set(polls_df.columns)
        for column in required_columns - poll_only_columns:
            if column.startswith("votes_") or column.startswith("available_") or column.startswith("new_"):
                polls_df[column] = 0
            elif column.startswith("incumbent_"):
                polls_df[column] = 0
            elif column in {
                "total_votes",
                "winner_share",
                "prev_winner_share",
                "incumbent_prev_share",
            }:
                polls_df[column] = polls_df.get(column, 0.0)
            elif column in {"winner_party", "prev_winner_party"}:
                polls_df[column] = None
            else:
                polls_df[column] = None

        extra_poll_columns = poll_only_columns - required_columns
        for column in extra_poll_columns:
            combined_results[column] = None

        combined_results = pd.concat(
            [combined_results, polls_df[combined_results.columns]],
            ignore_index=True,
            sort=False,
        )
        combined_results = combined_results.sort_values(
            ["municipality_code", "election_year", "is_poll"],
            kind="mergesort",
        ).reset_index(drop=True)
        combined_results = _augment_with_previous_results(combined_results, TARGET_PARTIES)
    else:
        combined_results = results_only

    if train_years is None:
        train_years = [year for year in election_years if year != max(election_years)]

    baseline_source = combined_results[combined_results["is_poll"] == 0]
    baseline_clr = compute_baseline_clr(baseline_source, training_years=train_years)
    national_clr = load_national_signal_clr(trace_path, election_years)
    district_offsets = load_district_offsets(trace_path)
    donor_weights = _estimate_local_donor_weights(
        baseline_source,
        metadata,
        baseline_clr,
        TARGET_PARTIES,
    )

    availability_cols = [f"available_{party}" for party in TARGET_PARTIES]
    combined_results[availability_cols] = combined_results.groupby(
        ["municipality_code", "election_year"]
    )[availability_cols].transform("max")

    combined_results = combined_results.sort_values(
        ["election_year", "municipality_code", "is_poll"],
        kind="mergesort",
    ).reset_index(drop=True)

    return MunicipalCouplingDataset(
        results=combined_results,
        metadata=metadata,
        baseline_clr=baseline_clr,
        national_clr=national_clr,
        district_offsets=district_offsets,
        donor_weights=donor_weights,
    )


def _softmax(values: Iterable[float]) -> np.ndarray:
    """Numerically stable softmax helper."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr - np.max(arr)
    exp_values = np.exp(arr)
    return exp_values / exp_values.sum()


# Lazy initialisation caches for column parsing
COLUMN_CACHE: Dict[str, List[str]] = {}
TOKEN_TO_PARTY: Mapping[str, str] = {
    "PS": "PS",
    "PSD": "PSD",
    "CDS": "CDS-PP",
    "CDU": "CDU",
    "BE": "BE",
    "CH": "CH",
    "IL": "IL",
    "LOCAL_INC": "LOCAL_INC",
}


# Required imports placed at end to prevent circular dependencies during module import
from pathlib import Path
import re
