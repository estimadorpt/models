"""
Data loaders for Portuguese Presidential Elections.

This module provides functions to load and process presidential election polling data.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from src.config import DATA_DIR


# Default list of candidates for the 2026 presidential election
DEFAULT_CANDIDATES_2026 = [
    'Gouveia e Melo',
    'Marques Mendes',
    'António José Seguro',
    'André Ventura',
    'Cotrim Figueiredo',
    'Catarina Martins',
    'António Filipe',
    'Others'
]


def load_presidential_polls(
    file_name: str = 'presidenciais_polls_2026.parquet',
    candidates: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load presidential election polls from a parquet file.

    Args:
        file_name: Name of the parquet file in the data directory.
        candidates: List of candidate column names to include.
                   If None, uses DEFAULT_CANDIDATES_2026.

    Returns:
        pd.DataFrame with columns:
            - date: Poll date (datetime)
            - pollster: Polling organization name
            - sample_size: Number of respondents
            - fieldwork_start: Start of fieldwork period (optional)
            - fieldwork_end: End of fieldwork period (optional)
            - [candidate columns]: Vote share for each candidate (0-1 scale)
            - undecided: Proportion of undecided voters (0-1 scale)

    Expected parquet structure:
        The input file should have columns for date, pollster, sample_size,
        and vote shares for each candidate (as percentages 0-100 or proportions 0-1).
    """
    if candidates is None:
        candidates = DEFAULT_CANDIDATES_2026.copy()

    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Presidential polls file not found at {file_path}. "
            f"Please provide the file with polling data."
        )

    df = pd.read_parquet(file_path)

    # Standardize column names (case-insensitive matching)
    column_mapping = _create_column_mapping(df.columns, candidates)
    df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required_cols = ['date', 'pollster', 'sample_size']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Handle fieldwork dates if present
    for date_col in ['fieldwork_start', 'fieldwork_end']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Ensure sample_size is numeric
    df['sample_size'] = pd.to_numeric(df['sample_size'], errors='coerce')
    mean_sample = df['sample_size'].mean()
    if pd.isna(mean_sample) or mean_sample == 0:
        mean_sample = 800  # Default fallback for presidential polls
    df['sample_size'] = df['sample_size'].fillna(mean_sample).astype(int)

    # Process candidate columns - convert to 0-1 scale if needed
    present_candidates = [c for c in candidates if c in df.columns]

    for candidate in present_candidates:
        df[candidate] = pd.to_numeric(df[candidate], errors='coerce').fillna(0)
        # Convert from percentage to proportion if values are > 1
        if df[candidate].max() > 1:
            df[candidate] = df[candidate] / 100

    # Handle undecided column
    if 'undecided' in df.columns:
        df['undecided'] = pd.to_numeric(df['undecided'], errors='coerce').fillna(0)
        if df['undecided'].max() > 1:
            df['undecided'] = df['undecided'] / 100
    else:
        # Calculate undecided as remainder if not explicitly provided
        total_declared = df[present_candidates].sum(axis=1)
        df['undecided'] = np.clip(1 - total_declared, 0, 1)

    # Add any missing candidate columns with 0
    for candidate in candidates:
        if candidate not in df.columns:
            df[candidate] = 0.0

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Loaded {len(df)} presidential polls from {file_name}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Pollsters: {df['pollster'].nunique()} unique")
    print(f"Candidates found: {present_candidates}")

    return df


def _create_column_mapping(columns: pd.Index, candidates: List[str]) -> Dict[str, str]:
    """
    Create a mapping from actual column names to standardized names.

    Handles case-insensitive matching and common variations.
    """
    mapping = {}
    columns_lower = {col.lower().strip(): col for col in columns}

    # Standard column mappings
    standard_mappings = {
        'date': ['date', 'data', 'poll_date', 'survey_date'],
        'pollster': ['pollster', 'instituto', 'polling_org', 'organization', 'source'],
        'sample_size': ['sample_size', 'n', 'amostra', 'sample', 'respondents'],
        'fieldwork_start': ['fieldwork_start', 'fieldwork start', 'start_date', 'inicio'],
        'fieldwork_end': ['fieldwork_end', 'fieldwork end', 'end_date', 'fim'],
        'undecided': ['undecided', 'indecisos', 'undec', 'ns/nr', 'dk/na'],
    }

    for standard_name, variations in standard_mappings.items():
        for var in variations:
            if var.lower() in columns_lower:
                actual_col = columns_lower[var.lower()]
                if actual_col != standard_name:
                    mapping[actual_col] = standard_name
                break

    # Candidate column mappings (fuzzy matching)
    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        for col_lower, col_actual in columns_lower.items():
            # Exact match or partial match
            if candidate_lower == col_lower or candidate_lower in col_lower:
                if col_actual != candidate:
                    mapping[col_actual] = candidate
                break

    return mapping


def cast_presidential_as_multinomial(
    df: pd.DataFrame,
    candidates: List[str],
    include_undecided: bool = True
) -> pd.DataFrame:
    """
    Convert percentages to counts for multinomial modeling.

    Args:
        df: DataFrame with candidate vote shares (0-1 scale)
        candidates: List of candidate column names
        include_undecided: Whether to include undecided as a category

    Returns:
        DataFrame with vote counts instead of proportions
    """
    df = df.copy()

    # Determine columns to convert
    cols_to_convert = [c for c in candidates if c in df.columns]
    if include_undecided and 'undecided' in df.columns:
        cols_to_convert.append('undecided')

    # Convert proportions to counts
    for col in cols_to_convert:
        df[col] = (df[col] * df['sample_size']).round().fillna(0).astype(int)

    # Recalculate sample_size as sum of all candidate counts
    df['sample_size'] = df[cols_to_convert].sum(axis=1)

    return df


def load_presidential_results(
    election_date: str,
    candidates: Optional[List[str]] = None,
    file_pattern: str = 'presidenciais_{year}.parquet'
) -> pd.DataFrame:
    """
    Load historical presidential election results.

    Args:
        election_date: Election date string (YYYY-MM-DD)
        candidates: List of candidate names to include
        file_pattern: Pattern for result files with {year} placeholder

    Returns:
        DataFrame with election results (vote counts per candidate)

    Note:
        For presidential elections, each election has different candidates,
        so historical results are primarily useful for calibrating pollster accuracy.
    """
    if candidates is None:
        candidates = DEFAULT_CANDIDATES_2026.copy()

    year = pd.to_datetime(election_date).year
    file_name = file_pattern.format(year=year)
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(file_path):
        print(f"Warning: No results file found for {election_date} at {file_path}")
        # Return empty placeholder
        return pd.DataFrame({
            'election_date': [pd.to_datetime(election_date)],
            'date': [pd.to_datetime(election_date)],
            'pollster': ['result'],
            'sample_size': [0],
            **{c: [0] for c in candidates}
        })

    df = pd.read_parquet(file_path)

    # Process results similar to polls
    df['election_date'] = pd.to_datetime(election_date)
    df['date'] = pd.to_datetime(election_date)
    df['pollster'] = 'result'

    return df


# Candidate prior configuration based on party affiliations
CANDIDATE_PARTY_PRIORS = {
    'Marques Mendes': {'party': 'AD', 'prior_mean': 0.25, 'prior_sd': 0.10},
    'António José Seguro': {'party': 'PS', 'prior_mean': 0.22, 'prior_sd': 0.10},
    'André Ventura': {'party': 'CH', 'prior_mean': 0.18, 'prior_sd': 0.10},
    'Cotrim Figueiredo': {'party': 'IL', 'prior_mean': 0.08, 'prior_sd': 0.05},
    'Catarina Martins': {'party': 'BE', 'prior_mean': 0.04, 'prior_sd': 0.03},
    'António Filipe': {'party': 'CDU', 'prior_mean': 0.03, 'prior_sd': 0.02},
    'Gouveia e Melo': {'party': None, 'prior_mean': 0.15, 'prior_sd': 0.10},  # Independent
    'Others': {'party': None, 'prior_mean': 0.05, 'prior_sd': 0.03},
}


def get_candidate_priors(candidates: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Get prior configurations for each candidate.

    Args:
        candidates: List of candidate names

    Returns:
        Dictionary mapping candidate names to their prior configurations
    """
    priors = {}
    for candidate in candidates:
        if candidate in CANDIDATE_PARTY_PRIORS:
            priors[candidate] = CANDIDATE_PARTY_PRIORS[candidate]
        else:
            # Default prior for unknown candidates
            priors[candidate] = {
                'party': None,
                'prior_mean': 0.05,
                'prior_sd': 0.05
            }
    return priors
