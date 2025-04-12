import os
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from scipy.special import softmax

from src.data.dataset import ElectionDataset # Assuming dataset structure
from src.models.base_model import BaseElectionModel


class DynamicGPElectionModel(BaseElectionModel):
    """
    A Bayesian election model combining a long-term baseline GP over calendar time
    with shorter-term, cycle-specific GPs over the days-to-election countdown.
    """

    def __init__(self, dataset: ElectionDataset, **kwargs):
        """
        Initializes the DynamicGPElectionModel.

        Args:
            dataset: An ElectionDataset object containing polls, results, etc.
            **kwargs: Additional keyword arguments for model configuration.
                      Expected keys:
                        'baseline_gp_lengthscale', 'baseline_gp_kernel',
                        'cycle_gp_lengthscale', 'cycle_gp_kernel', 'cycle_gp_max_days',
                        'hsgp_m', 'hsgp_c'.
        """
        super().__init__(dataset, **kwargs)

        # Max days before election for the cycle GP
        self.cycle_gp_max_days = kwargs.get("cycle_gp_max_days", 180)

        # Configuration for the GPs
        self.baseline_gp_config = {
            "lengthscale": kwargs.get("baseline_gp_lengthscale", 365.0), # Long lengthscale
            "kernel": kwargs.get("baseline_gp_kernel", "Matern52"),
            "hsgp_m": kwargs.get("hsgp_m", [100]), # Basis functions for HSGP (can be shared or separate)
            "hsgp_c": kwargs.get("hsgp_c", 2.0), # Expansion factor for HSGP
        }
        self.cycle_gp_config = {
            "lengthscale": kwargs.get("cycle_gp_lengthscale", 45.0), # Shorter lengthscale
            "kernel": kwargs.get("cycle_gp_kernel", "Matern52"),
            "hsgp_m": kwargs.get("hsgp_m_cycle", [50]), # Potentially fewer basis functions
            "hsgp_c": kwargs.get("hsgp_c_cycle", 1.5),
        }

        # Attributes to be set
        self.pollster_id = None
        self.calendar_time_poll_id = None
        self.calendar_time_result_id = None
        self.observed_election_indices = None # Indices relative to all_election_dates
        self.calendar_time_numeric = None # Numerical representation of calendar time coord
        self.poll_days_numeric = None     # Numeric days to election for each poll
        self.poll_cycle_idx = None        # Index mapping poll to election cycle
        self.result_cycle_idx = None       # Index mapping result to election cycle
        self.gp_calendar_time = None
        self.gp_cycle_time = None
        self.calendar_time_cycle_idx = None
        self.calendar_time_days_numeric = None


    def _build_coords(self, polls: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray, np.ndarray]:
        """
        Build coordinates for the PyMC model, including calendar time, election cycles,
        and days_to_election.
        Calculates mappings for polls, results, AND the full calendar time axis.

        Args:
            polls: Optional DataFrame of polls to use instead of self.polls_train.

        Returns:
            Tuple containing:
            - pollster_id
            - calendar_time_poll_id
            - calendar_time_result_id (unfiltered)
            - poll_days_numeric
            - poll_cycle_idx
            - result_cycle_idx (unfiltered)
            - calendar_time_cycle_idx (Full mapping for calendar_time)
            - calendar_time_days_numeric (Full mapping for calendar_time)
            - COORDS dictionary
            - observed_election_indices
        """
        data_polls = polls if polls is not None else self.polls_train
        historical_results = self.results_oos # From base class

        # --- Create Unified Calendar Time Coordinate ---
        poll_dates_dt = pd.to_datetime(data_polls['date']).unique()
        all_election_dates_dt = pd.to_datetime(self.all_election_dates).unique()
        unique_dates = pd.to_datetime(
             np.union1d(poll_dates_dt, all_election_dates_dt)
        ).unique().sort_values()

        min_date = unique_dates.min()
        calendar_time_numeric = (unique_dates - min_date).days.values
        self.calendar_time_numeric = calendar_time_numeric
        date_to_calendar_index = {date: i for i, date in enumerate(unique_dates)}

        # --- Define Election Cycles & Mappings ---
        cycle_boundaries_dt = pd.to_datetime(sorted(self.all_election_dates))
        cycle_names = [f"Cycle_{d.strftime('%Y-%m-%d')}" for d in cycle_boundaries_dt]
        date_str_to_cycle_idx = {date.strftime('%Y-%m-%d'): i for i, date in enumerate(cycle_boundaries_dt)}

        # Calculate mappings for ALL calendar time points
        calendar_time_cycle_idx_list = []
        calendar_time_days_list = []
        for date in unique_dates:
            # Find the cycle this date belongs to
            cycle_end_date = min([d for d in cycle_boundaries_dt if d >= date], default=cycle_boundaries_dt[-1])
            cycle_idx = date_str_to_cycle_idx[cycle_end_date.strftime('%Y-%m-%d')]
            days_numeric = (cycle_end_date - date).days
            days_numeric_clipped = np.clip(days_numeric, 0, self.cycle_gp_max_days)

            calendar_time_cycle_idx_list.append(cycle_idx)
            calendar_time_days_list.append(days_numeric_clipped)

        self.calendar_time_cycle_idx = np.array(calendar_time_cycle_idx_list, dtype=int)
        self.calendar_time_days_numeric = np.array(calendar_time_days_list, dtype=float)

        # Calculate days to election for polls (as before)
        if 'countdown' not in data_polls.columns:
             raise ValueError("Missing 'countdown' column in polls data, needed for cycle GP.")
        poll_days_numeric_raw = data_polls['countdown'].values
        poll_days_numeric = np.clip(poll_days_numeric_raw, 0, self.cycle_gp_max_days)
        self.poll_days_numeric = poll_days_numeric.astype(float)

        # Map polls to their cycle index (as before)
        poll_election_dates_str = pd.to_datetime(data_polls['election_date']).dt.strftime('%Y-%m-%d')
        poll_cycle_idx = poll_election_dates_str.map(date_str_to_cycle_idx).values
        if np.isnan(poll_cycle_idx).any():
             missing_poll_cycles = data_polls.loc[np.isnan(poll_cycle_idx), 'election_date'].unique()
             raise ValueError(f"Could not map some poll election dates to cycles: {missing_poll_cycles}")
        self.poll_cycle_idx = poll_cycle_idx.astype(int)

        # --- Map Polls and ALL Results to Calendar Time and Cycle (Initially Unfiltered) ---
        poll_dates_dt = pd.to_datetime(data_polls['date'])
        calendar_time_poll_id = poll_dates_dt.map(date_to_calendar_index).values.astype(int)

        historical_results_dates_dt = pd.to_datetime(historical_results['election_date'])
        self.calendar_time_result_id = historical_results_dates_dt.map(date_to_calendar_index).values.astype(int)

        result_election_dates_str = historical_results_dates_dt.dt.strftime('%Y-%m-%d')
        self.result_cycle_idx = result_election_dates_str.map(date_str_to_cycle_idx).values
        if np.isnan(self.result_cycle_idx).any():
            missing_result_cycles = historical_results.loc[np.isnan(self.result_cycle_idx), 'election_date'].unique()
            raise ValueError(f"Could not map some (unfiltered) result election dates to cycles: {missing_result_cycles}")
        self.result_cycle_idx = self.result_cycle_idx.astype(int)

        # Find indices of observed elections relative to *all* election dates
        all_election_dates_iso = [d.date().isoformat() for d in cycle_boundaries_dt]
        observed_election_dates_iso_set = set(historical_results_dates_dt.dt.strftime('%Y-%m-%d'))
        self.observed_election_indices = [i for i, election_iso in enumerate(all_election_dates_iso)
                                          if election_iso in observed_election_dates_iso_set]

        # --- Define Coordinates ---
        COORDS = {
            "observations": data_polls.index,
            "parties_complete": self.political_families,
            "calendar_time": unique_dates.strftime('%Y-%m-%d'),
            "elections_observed": [all_election_dates_iso[i] for i in self.observed_election_indices],
            "election_cycles": cycle_names,
        }

        pollster_id, COORDS["pollsters"] = data_polls["pollster"].factorize(sort=True)

        print("\n=== MODEL COORDINATES ===")
        for key, value in COORDS.items():
            print(f"{key}: length={len(value)}")
        print(f"Calendar time range: {COORDS['calendar_time'][0]} to {COORDS['calendar_time'][-1]}")
        print(f"Number of election cycles: {len(COORDS['election_cycles'])}")

        # Return only 10 values, excluding days_to_election_numeric_coord
        return (pollster_id, calendar_time_poll_id, self.calendar_time_result_id,
                self.poll_days_numeric, self.poll_cycle_idx, self.result_cycle_idx,
                self.calendar_time_cycle_idx, self.calendar_time_days_numeric,
                COORDS, self.observed_election_indices)


    def _build_data_containers(self,
                              polls: pd.DataFrame = None
                             ) -> Dict[str, pm.Data]:
        """
        Build the data containers for the PyMC model.

        Note on Penalties (Oct 2024):
        Previous versions used large negative additive penalties (-100)
        in the `non_competing_*_additive` masks to suppress probabilities
        for parties during periods they weren't polled or didn't exist.
        However, empirical runs (e.g., `pen0` vs `pen100`) showed that for
        this two-timescale calendar GP model, the likelihood pressure from
        zero observations was sufficient to correctly push latent scores low,
        without causing sampling issues.
        Therefore, the explicit penalty masks are no longer created or used.
        The multiplicative mask (`non_competing_polls_multiplicative`) was also
        found to be unused in this model's likelihood and has been removed.
        """
        current_polls = polls if polls is not None else self.polls_train

        # --- Filter results-related data based on observed_election_indices ---
        if self.observed_election_indices is None:
             raise ValueError("observed_election_indices not set. Ensure _build_coords runs first.")

        # Use observed_election_indices (indices relative to all_election_dates)
        # to filter the *original* results_oos dataframe and cycle indices
        if isinstance(self.observed_election_indices, (list, np.ndarray)):
             observed_election_dates_in_all = pd.to_datetime([self.all_election_dates[i] for i in self.observed_election_indices])
             results_filter_mask = self.results_oos['election_date'].isin(observed_election_dates_in_all)

             if results_filter_mask.sum() == 0 and len(self.observed_election_indices) > 0:
                  print(f"Warning: No rows in results_oos matched the observed_election_indices dates. Check alignment.")
                  results_N_historical = np.array([], dtype=float)
                  observed_results_historical = np.empty((0, len(self.political_families)), dtype=float)
                  calendar_time_result_id_filtered = np.array([], dtype=int)
                  result_cycle_idx_filtered = np.array([], dtype=int)
             elif results_filter_mask.sum() > 0:
                  results_N_historical = self.results_oos.loc[results_filter_mask, "sample_size"].to_numpy()
                  observed_results_historical = self.results_oos.loc[results_filter_mask, self.political_families].to_numpy()
                  # Filter indices based on the mask applied to results_oos
                  if self.calendar_time_result_id is not None and len(self.calendar_time_result_id) == len(self.results_oos):
                      calendar_time_result_id_filtered = self.calendar_time_result_id[results_filter_mask]
                  else: calendar_time_result_id_filtered = np.array([], dtype=int); print("Warn: cal_time_res_id filtering issue")
                  if self.result_cycle_idx is not None and len(self.result_cycle_idx) == len(self.results_oos):
                      result_cycle_idx_filtered = self.result_cycle_idx[results_filter_mask]
                  else: result_cycle_idx_filtered = np.array([], dtype=int); print("Warn: res_cycle_idx filtering issue")
             else: # observed_election_indices is empty
                 print(f"Warning: observed_election_indices is empty. Filtering results to empty arrays.")
                 results_N_historical = np.array([], dtype=float)
                 observed_results_historical = np.empty((0, len(self.political_families)), dtype=float)
                 calendar_time_result_id_filtered = np.array([], dtype=int)
                 result_cycle_idx_filtered = np.array([], dtype=int)
        else:
             raise ValueError(f"Invalid type or structure for observed_election_indices: {self.observed_election_indices}")
        # --- End Filtering ---

        # print(f"Shape of filtered observed_results: {observed_results_historical.shape}") # Keep useful prints
        # print(f"Shape of filtered results_N: {results_N_historical.shape}")
        # print(f"Shape of filtered calendar_time_result_id: {calendar_time_result_id_filtered.shape}")
        # print(f"Shape of filtered result_cycle_idx: {result_cycle_idx_filtered.shape}")

        # Ensure other necessary arrays are set
        if self.poll_days_numeric is None or self.poll_cycle_idx is None or \
           self.calendar_time_cycle_idx is None or self.calendar_time_days_numeric is None:
            raise ValueError("Indices/mappings not set. Ensure _build_coords runs first.")

        data_containers = dict(
            # --- Indices ---
            calendar_time_poll_idx=pm.Data("calendar_time_poll_idx", self.calendar_time_poll_id, dims="observations"),
            calendar_time_result_idx=pm.Data("calendar_time_result_idx", calendar_time_result_id_filtered, dims="elections_observed"),
            pollster_idx=pm.Data("pollster_idx", self.pollster_id, dims="observations"),
            poll_cycle_idx=pm.Data("poll_cycle_idx", self.poll_cycle_idx, dims="observations"),
            result_cycle_idx=pm.Data("result_cycle_idx", result_cycle_idx_filtered, dims="elections_observed"),
            calendar_time_cycle_idx=pm.Data("calendar_time_cycle_idx", self.calendar_time_cycle_idx, dims="calendar_time"),
            # --- GP Inputs ---
            poll_days_numeric=pm.Data("poll_days_numeric", self.poll_days_numeric, dims="observations"),
            calendar_time_days_numeric=pm.Data("calendar_time_days_numeric", self.calendar_time_days_numeric, dims="calendar_time"),
            # --- Observed Data ---
            observed_N_polls=pm.Data("observed_N_polls", current_polls["sample_size"].to_numpy(), dims="observations"),
            observed_polls=pm.Data("observed_polls", current_polls[self.political_families].to_numpy(), dims=("observations", "parties_complete")),
            observed_N_results=pm.Data("observed_N_results", results_N_historical, dims="elections_observed"),
            observed_results=pm.Data("observed_results", observed_results_historical, dims=("elections_observed", "parties_complete")),
            # --- Masks (Removed as penalty is no longer needed) ---
            # non_competing_parties_results=... (Removed)
            # non_competing_polls_additive=... (Removed)
            # non_competing_polls_multiplicative=... (Removed)
            # non_competing_calendar_additive=... (Removed)
        )
        return data_containers


    def build_model(self, polls: pd.DataFrame = None) -> pm.Model:
        """
        Build the PyMC model with two GPs over calendar time (long & short timescale) + house effects.
        """
        # Unpacking expects 10 values (no days_to_election_coord)
        (
            self.pollster_id,
            self.calendar_time_poll_id,
            self.calendar_time_result_id,
            self.poll_days_numeric,
            self.poll_cycle_idx,
            self.result_cycle_idx,
            self.calendar_time_cycle_idx,
            self.calendar_time_days_numeric,
            self.coords,
            self.observed_election_indices
        ) = self._build_coords(polls)

        with pm.Model(coords=self.coords) as model:
            data_containers = self._build_data_containers(polls)

            # --------------------------------------------------------
            #        1. BASELINE GP (Long Trend over Calendar Time)
            # --------------------------------------------------------
            baseline_gp_ls = pm.LogNormal("baseline_gp_ls", mu=np.log(365*2.5), sigma=0.3) # Mean ~2.5 years
            baseline_gp_amp_sd = pm.HalfNormal("baseline_gp_amp_sd", sigma=0.2) # Tight amplitude

            if self.baseline_gp_config["kernel"] == "Matern52":
                 cov_func_baseline = baseline_gp_amp_sd**2 * pm.gp.cov.Matern52(input_dim=1, ls=baseline_gp_ls)
            elif self.baseline_gp_config["kernel"] == "ExpQuad":
                 cov_func_baseline = baseline_gp_amp_sd**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=baseline_gp_ls)
            else: raise ValueError(f"Unsupported Baseline GP kernel: {self.baseline_gp_config['kernel']}")

            baseline_hsgp_m = [50] # Keep reduced complexity
            baseline_hsgp_c = 1.8
            self.gp_baseline_time = pm.gp.HSGP(
                cov_func=cov_func_baseline, m=baseline_hsgp_m, c=baseline_hsgp_c
            )
            phi_baseline, sqrt_psd_baseline = self.gp_baseline_time.prior_linearized(X=self.calendar_time_numeric[:, None])

            coord_name_gp_basis_base = "gp_basis_baseline"
            if coord_name_gp_basis_base not in model.coords:
                 model.add_coords({coord_name_gp_basis_base: np.arange(self.gp_baseline_time.n_basis_vectors)})

            baseline_gp_coef_raw = pm.Normal("baseline_gp_coef_raw", mu=0, sigma=1,
                                             dims=(coord_name_gp_basis_base, "parties_complete"))
            baseline_gp_coef = pm.Deterministic("baseline_gp_coef",
                                                baseline_gp_coef_raw - baseline_gp_coef_raw.mean(axis=1, keepdims=True),
                                                dims=(coord_name_gp_basis_base, "parties_complete"))
            baseline_effect_calendar = pm.Deterministic("baseline_effect_calendar",
                pt.einsum('cb,bp->cp', phi_baseline, baseline_gp_coef * sqrt_psd_baseline[:, None]),
                dims=("calendar_time", "parties_complete")
            )

            # --------------------------------------------------------
            #        2. SHORT-TERM GP (Over Calendar Time)
            # --------------------------------------------------------
            short_term_gp_ls = pm.LogNormal("short_term_gp_ls", mu=np.log(30), sigma=0.5) # Shorter Mean: ~1 month
            short_term_gp_amp_sd = pm.HalfNormal("short_term_gp_amp_sd", sigma=0.1) # Keep smaller amplitude than baseline

            # Assume same kernel type as baseline for now, can be configured later
            if self.baseline_gp_config["kernel"] == "Matern52":
                 cov_func_short_term = short_term_gp_amp_sd**2 * pm.gp.cov.Matern52(input_dim=1, ls=short_term_gp_ls)
            elif self.baseline_gp_config["kernel"] == "ExpQuad":
                 cov_func_short_term = short_term_gp_amp_sd**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=short_term_gp_ls)
            else: raise ValueError(f"Unsupported Short-Term GP kernel (using baseline type): {self.baseline_gp_config['kernel']}")

            # Potentially different complexity for short-term GP
            short_term_hsgp_m = [75] # Maybe allow more flexibility here?
            short_term_hsgp_c = 1.8
            self.gp_short_term_time = pm.gp.HSGP(
                cov_func=cov_func_short_term, m=short_term_hsgp_m, c=short_term_hsgp_c
            )
            # Uses the same calendar time input as baseline GP
            phi_short_term, sqrt_psd_short_term = self.gp_short_term_time.prior_linearized(X=self.calendar_time_numeric[:, None])

            coord_name_gp_basis_short = "gp_basis_short_term"
            if coord_name_gp_basis_short not in model.coords:
                 model.add_coords({coord_name_gp_basis_short: np.arange(self.gp_short_term_time.n_basis_vectors)})

            short_term_gp_coef_raw = pm.Normal("short_term_gp_coef_raw", mu=0, sigma=1,
                                              dims=(coord_name_gp_basis_short, "parties_complete"))
            short_term_gp_coef = pm.Deterministic("short_term_gp_coef",
                                                 short_term_gp_coef_raw - short_term_gp_coef_raw.mean(axis=1, keepdims=True),
                                                 dims=(coord_name_gp_basis_short, "parties_complete"))
            short_term_effect_calendar = pm.Deterministic("short_term_effect_calendar",
                pt.einsum('cb,bp->cp', phi_short_term, short_term_gp_coef * sqrt_psd_short_term[:, None]),
                dims=("calendar_time", "parties_complete")
            )

            # --------------------------------------------------------
            #          3. HOUSE EFFECTS (Pollster Bias)
            # --------------------------------------------------------
            house_effects_sd = pm.HalfNormal("house_effects_sd", sigma=0.1) # Keep relaxed prior
            house_effects = pm.ZeroSumNormal("house_effects", sigma=house_effects_sd,
                                             dims=("pollsters", "parties_complete"))

            # --------------------------------------------------------
            #                      4. POLL LIKELIHOOD
            # --------------------------------------------------------
            latent_mu_polls = pm.Deterministic("latent_mu_polls",
                (
                    baseline_effect_calendar[data_containers["calendar_time_poll_idx"]] + # Baseline effect
                    short_term_effect_calendar[data_containers["calendar_time_poll_idx"]]   # Short-term effect
                ),
                dims=("observations", "parties_complete")
            )

            noisy_mu_polls = pm.Deterministic("noisy_mu_polls",
                (
                    latent_mu_polls +
                    house_effects[data_containers["pollster_idx"]] # Add house effect directly
                ),
                dims=("observations", "parties_complete")
            )
            noisy_popularity_polls = pm.Deterministic("noisy_popularity_polls",
                pt.special.softmax(noisy_mu_polls, axis=1),
                dims=("observations", "parties_complete"),
            )
            concentration_polls = pm.Gamma("concentration_polls", alpha=100, beta=0.1)
            N_approve = pm.DirichletMultinomial("N_approve",
                a=(concentration_polls * noisy_popularity_polls),
                n=data_containers["observed_N_polls"],
                observed=data_containers["observed_polls"],
                dims=("observations", "parties_complete"),
            )

            # --------------------------------------------------------
            #                 5. ELECTION RESULT LIKELIHOOD
            # --------------------------------------------------------
            latent_mu_results = pm.Deterministic("latent_mu_results",
                (
                     baseline_effect_calendar[data_containers["calendar_time_result_idx"]] + # Baseline effect
                     short_term_effect_calendar[data_containers["calendar_time_result_idx"]] # Short-term effect
                 ),
                dims=("elections_observed", "parties_complete")
            )
            latent_pop_results = pm.Deterministic("latent_pop_results",
                pt.special.softmax(latent_mu_results, axis=1),
                dims=("elections_observed", "parties_complete"),
            )
            concentration_results = pm.Gamma("concentration_results", alpha=100, beta=0.05)
            R = pm.DirichletMultinomial("R",
                n=data_containers["observed_N_results"],
                a=(concentration_results * latent_pop_results),
                observed=data_containers["observed_results"],
                dims=("elections_observed", "parties_complete")
            )

            # --------------------------------------------------------
            #       (Optional) Deterministics for Analysis
            # --------------------------------------------------------
            # Combined latent trajectory sums both GP effects (unpenalized)
            latent_mu_calendar = pm.Deterministic("latent_mu_calendar",
                baseline_effect_calendar + short_term_effect_calendar,
                dims=("calendar_time", "parties_complete")
            )

            # --- Original Deterministics --- 
            # This is what the plot currently uses
            latent_popularity_calendar_trajectory = pm.Deterministic(
                "latent_popularity_calendar_trajectory",
                pt.special.softmax(latent_mu_calendar, axis=1),
                dims=("calendar_time", "parties_complete")
            )

            # Penalized versions removed as penalty mask is no longer used
            # penalized_latent_mu_calendar = ...
            # penalized_latent_popularity_calendar_trajectory = ...

        self.model = model
        print("\nModel building complete (Two-Timescale Calendar GP + House Effects).")
        return model

    # --- Placeholder methods ---
    def posterior_predictive_check(self, posterior):
        print("Warning: posterior_predictive_check not yet implemented.")
        return {}

    def predict_latent_trajectory(self, idata: az.InferenceData, start_date: pd.Timestamp, end_date: pd.Timestamp) -> az.InferenceData:
        print("Warning: predict_latent_trajectory not yet implemented.")
        return az.InferenceData()

    def predict(self, oos_data: pd.DataFrame) -> az.InferenceData:
        print("Warning: predict (for polls) not yet implemented.")
        return az.InferenceData()

    def predict_history(self, elections_to_predict: List[str]) -> az.InferenceData:
        print("Warning: predict_history not yet implemented.")
        return az.InferenceData() 