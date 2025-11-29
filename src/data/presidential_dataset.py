"""
Dataset class for Portuguese Presidential Elections.

This module provides the PresidentialElectionDataset class for loading and
preparing presidential election polling data for Bayesian modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.data.presidential_loaders import (
    load_presidential_polls,
    cast_presidential_as_multinomial,
    get_candidate_priors,
    DEFAULT_CANDIDATES_2026,
)


class PresidentialElectionDataset:
    """
    Dataset class for presidential election forecasting.

    Unlike parliamentary elections where parties persist across elections,
    presidential elections have unique candidates each cycle. This class
    handles the specific requirements of presidential election modeling.

    Attributes:
        election_date: Target election date (e.g., '2026-01-18')
        candidates: List of candidate names being modeled
        polls_train: Training poll data (multinomial counts)
        unique_pollsters: Unique polling organizations
        candidate_priors: Prior configuration for each candidate
    """

    # Default candidates for 2026 election
    default_candidates = DEFAULT_CANDIDATES_2026

    def __init__(
        self,
        election_date: str,
        candidates: Optional[List[str]] = None,
        polls_file: str = 'presidenciais_polls_2026.parquet',
        include_undecided_as_candidate: bool = False,
        cutoff_date: Optional[str] = None,
    ):
        """
        Initialize the presidential election dataset.

        Args:
            election_date: Target election date (YYYY-MM-DD format).
                          For 2026: '2026-01-18'
            candidates: List of candidate names. If None, uses default list.
            polls_file: Name of the parquet file containing polling data.
            include_undecided_as_candidate: If True, treat undecided as a
                                            separate category in the model.
            cutoff_date: If provided, exclude polls after this date
                        (for retrodictive testing).
        """
        self.election_date = election_date
        self.election_date_dt = pd.to_datetime(election_date)
        self.candidates = candidates if candidates is not None else self.default_candidates.copy()
        self.include_undecided = include_undecided_as_candidate
        self.cutoff_date = pd.to_datetime(cutoff_date) if cutoff_date else None

        print(f"\n=== Initializing Presidential Election Dataset ===")
        print(f"Election date: {self.election_date}")
        print(f"Candidates: {self.candidates}")

        # Load polling data
        self.polls_raw = self._load_polls(polls_file)

        # Convert to multinomial format
        self.polls_train = self._prepare_multinomial_data()

        # Extract unique entities
        _, self.unique_pollsters = self.polls_train['pollster'].factorize()

        # Get candidate priors based on party affiliations
        self.candidate_priors = get_candidate_priors(self.candidates)

        # Store undecided data for probabilistic allocation
        self._prepare_undecided_data()

        self._print_diagnostics()

    def _load_polls(self, polls_file: str) -> pd.DataFrame:
        """Load and preprocess polling data."""
        df = load_presidential_polls(file_name=polls_file, candidates=self.candidates)

        # Add countdown to election
        df['countdown'] = (self.election_date_dt - df['date']).dt.days

        # Filter polls to those before the election
        df = df[df['countdown'] >= 0]

        # Apply cutoff date if specified
        if self.cutoff_date is not None:
            print(f"Applying cutoff date: {self.cutoff_date.date()}")
            df = df[df['date'] <= self.cutoff_date]

        # Filter out polls that are too old (e.g., more than 2 years before election)
        max_countdown = 730  # ~2 years
        df = df[df['countdown'] <= max_countdown]

        print(f"Loaded {len(df)} polls after filtering")

        return df

    def _prepare_multinomial_data(self) -> pd.DataFrame:
        """Convert poll percentages to multinomial counts."""
        # Determine which columns to include
        model_candidates = self.candidates.copy()

        df = cast_presidential_as_multinomial(
            self.polls_raw,
            candidates=model_candidates,
            include_undecided=self.include_undecided
        )

        return df

    def _prepare_undecided_data(self):
        """
        Prepare undecided voter data for probabilistic allocation.

        This stores the original undecided proportions before multinomial
        conversion, which can be used to model how undecided voters
        might break toward candidates.
        """
        if 'undecided' in self.polls_raw.columns:
            self.undecided_proportions = self.polls_raw['undecided'].values
            self.mean_undecided = self.polls_raw['undecided'].mean()
        else:
            self.undecided_proportions = np.zeros(len(self.polls_raw))
            self.mean_undecided = 0.0

        print(f"Mean undecided rate: {self.mean_undecided:.1%}")

    def _print_diagnostics(self):
        """Print diagnostic information about the dataset."""
        print(f"\n=== Dataset Diagnostics ===")
        print(f"Number of polls: {len(self.polls_train)}")
        print(f"Number of pollsters: {len(self.unique_pollsters)}")
        print(f"Pollsters: {list(self.unique_pollsters)}")
        print(f"Date range: {self.polls_train['date'].min().date()} to "
              f"{self.polls_train['date'].max().date()}")
        print(f"Countdown range: {self.polls_train['countdown'].min()} to "
              f"{self.polls_train['countdown'].max()} days")

        # Print average support per candidate
        print(f"\n=== Average Support (from raw polls) ===")
        for candidate in self.candidates:
            if candidate in self.polls_raw.columns:
                mean_support = self.polls_raw[candidate].mean()
                print(f"  {candidate}: {mean_support:.1%}")

        # Check for NaN values
        nan_counts = self.polls_train.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\nWARNING: NaN values found:")
            print(nan_counts[nan_counts > 0])

    def get_candidate_columns(self) -> List[str]:
        """Get list of candidate columns in the dataset."""
        cols = [c for c in self.candidates if c in self.polls_train.columns]
        if self.include_undecided and 'undecided' in self.polls_train.columns:
            cols.append('undecided')
        return cols

    def get_prior_means(self) -> np.ndarray:
        """
        Get prior mean values for each candidate.

        Returns:
            Array of prior means in the same order as self.candidates
        """
        means = []
        for candidate in self.candidates:
            if candidate in self.candidate_priors:
                means.append(self.candidate_priors[candidate]['prior_mean'])
            else:
                means.append(0.05)  # Default prior
        return np.array(means)

    def get_prior_sds(self) -> np.ndarray:
        """
        Get prior standard deviations for each candidate.

        Returns:
            Array of prior SDs in the same order as self.candidates
        """
        sds = []
        for candidate in self.candidates:
            if candidate in self.candidate_priors:
                sds.append(self.candidate_priors[candidate]['prior_sd'])
            else:
                sds.append(0.05)  # Default prior
        return np.array(sds)

    def prepare_model_data(self) -> Dict:
        """
        Prepare data dictionary for model building.

        Returns:
            Dictionary containing all data needed for model construction.
        """
        candidate_cols = self.get_candidate_columns()

        return {
            'polls': self.polls_train,
            'candidates': self.candidates,
            'candidate_columns': candidate_cols,
            'n_candidates': len(candidate_cols),
            'n_polls': len(self.polls_train),
            'n_pollsters': len(self.unique_pollsters),
            'pollster_names': list(self.unique_pollsters),
            'election_date': self.election_date_dt,
            'prior_means': self.get_prior_means(),
            'prior_sds': self.get_prior_sds(),
            'undecided_proportions': self.undecided_proportions,
            'mean_undecided': self.mean_undecided,
        }

    def generate_forecast_dates(
        self,
        start_days_before: int = 365,
        end_days_before: int = 0
    ) -> pd.DatetimeIndex:
        """
        Generate date range for forecasting.

        Args:
            start_days_before: Number of days before election to start
            end_days_before: Number of days before election to end (0 = election day)

        Returns:
            DatetimeIndex of forecast dates
        """
        start_date = self.election_date_dt - pd.Timedelta(days=start_days_before)
        end_date = self.election_date_dt - pd.Timedelta(days=end_days_before)

        return pd.date_range(start=start_date, end=end_date, freq='D')
