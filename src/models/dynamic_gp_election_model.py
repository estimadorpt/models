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
from src.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_log_score,
    calculate_rps,
    calculate_calibration_data
)


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
    def posterior_predictive_check(self, idata: az.InferenceData, extend_idata: bool = True) -> az.InferenceData | Dict:
        """
        Performs posterior predictive checks on the poll and result likelihoods.

        Generates samples from the posterior predictive distribution for observed
        variables (poll counts 'N_approve' and result counts 'R').

        Args:
            idata: The InferenceData object containing the posterior trace.
            extend_idata: If True (default), adds the posterior predictive samples
                          to the input InferenceData object and returns it.
                          If False, returns a dictionary containing the samples.

        Returns:
            The input InferenceData object extended with posterior predictive samples
            (if extend_idata=True), or a dictionary containing the samples.

        Raises:
            ValueError: If the model hasn't been built yet.
        """
        if self.model is None:
            raise ValueError("Model must be built before running PPC. Call build_model() first.")
        if idata is None or "posterior" not in idata:
             raise ValueError("Valid InferenceData object with posterior samples required.")

        with self.model: # Re-enter the model context
            # Sample from the posterior predictive distribution for polls and results
            # We want samples of the *observed* variables based on the posterior
            ppc_samples = pm.sample_posterior_predictive(
                idata,
                var_names=[
                    "N_approve", # Simulated poll counts
                    "R",         # Simulated election result counts
                    "noisy_popularity_polls", # Also sample the underlying probability for polls
                    "latent_pop_results"      # And the underlying probability for results
                    ],
                predictions=False, # Ensure we are sampling the observed nodes
                extend_inferencedata=extend_idata, # Add samples to idata
            )

        print("Posterior predictive samples generated.")
        print("Use ArviZ (e.g., az.plot_ppc) or custom functions to compare:")
        print("  - 'N_approve' samples vs observed_polls data")
        print("  - 'R' samples vs observed_results data")

        if extend_idata:
            return idata # idata is modified in-place if extend_idata=True in sample_posterior_predictive
        else:
            # If not extending, sample_posterior_predictive returns the dictionary directly
            return ppc_samples


    def calculate_fit_metrics(self, idata: az.InferenceData) -> Dict[str, float]:
        """
        Calculates various goodness-of-fit metrics using posterior predictive samples.

        Compares model predictions against observed poll and election result data.

        Args:
            idata: InferenceData object containing posterior and ideally
                   posterior_predictive and observed_data groups.

        Returns:
            A dictionary containing calculated metrics:
            - poll_mae: Mean Absolute Error on poll proportions.
            - poll_rmse: Root Mean Squared Error on poll proportions.
            - poll_log_score: Average Log Score for poll counts.
            - poll_rps: Average Rank Probability Score for poll proportions.
            - result_mae: Mean Absolute Error on result proportions.
            - result_rmse: Root Mean Squared Error on result proportions.
            - result_log_score: Average Log Score for result counts.
            - result_rps: Average Rank Probability Score for result proportions.

        Raises:
            ValueError: If required data groups (posterior_predictive, observed_data)
                      or variables are missing from idata.
        """
        # 1. Ensure Posterior Predictive samples exist
        if "posterior_predictive" not in idata:
            print("Posterior predictive samples not found in idata. Running posterior_predictive_check...")
            try:
                self.posterior_predictive_check(idata, extend_idata=True) # Modifies idata in-place
            except Exception as e:
                 raise ValueError(f"Failed to generate posterior predictive samples: {e}")

        if "observed_data" not in idata:
            raise ValueError("observed_data group is required in InferenceData to calculate metrics.")

        # 2. Extract Predictions (Mean Probabilities) and Observed Data
        print("\n--- Debugging Metric Calculation Inputs ---") # DEBUG
        try:
            # --- Polls --- 
            pp_poll_da = idata.posterior_predictive["noisy_popularity_polls"]
            pred_poll_probs = pp_poll_da.mean(dim=["chain", "draw"]).values
            obs_poll_counts = idata.observed_data["observed_polls"].values
            obs_poll_n = idata.observed_data["observed_N_polls"].values
            with np.errstate(divide='ignore', invalid='ignore'):
                obs_poll_probs = obs_poll_counts / obs_poll_n[:, np.newaxis]
            obs_poll_probs = np.nan_to_num(obs_poll_probs)
            
            print("\n[Poll Data]")
            print(f"  Shape pred_poll_probs: {pred_poll_probs.shape}")
            print(f"  Shape obs_poll_probs: {obs_poll_probs.shape}")
            print(f"  Shape obs_poll_counts: {obs_poll_counts.shape}")
            print(f"  Pred Poll Probs (mean): {pred_poll_probs.mean():.4f}, (min): {pred_poll_probs.min():.4f}, (max): {pred_poll_probs.max():.4f}")
            print(f"  Obs Poll Probs (mean): {obs_poll_probs.mean():.4f}, (min): {obs_poll_probs.min():.4f}, (max): {obs_poll_probs.max():.4f}")
            # print(f"  Pred Poll Probs (first 2 rows):\n{pred_poll_probs[:2]}") # Optional: uncomment for more detail
            # print(f"  Obs Poll Probs (first 2 rows):\n{obs_poll_probs[:2]}") # Optional: uncomment for more detail

            # --- Results --- 
            pp_result_da = idata.posterior_predictive["latent_pop_results"]
            pred_result_probs = pp_result_da.mean(dim=["chain", "draw"]).values
            obs_result_counts = idata.observed_data["observed_results"].values
            obs_result_n = idata.observed_data["observed_N_results"].values
            with np.errstate(divide='ignore', invalid='ignore'):
                obs_result_probs = obs_result_counts / obs_result_n[:, np.newaxis]
            obs_result_probs = np.nan_to_num(obs_result_probs)
            
            print("\n[Result Data]")
            print(f"  Shape pred_result_probs: {pred_result_probs.shape}")
            print(f"  Shape obs_result_probs: {obs_result_probs.shape}")
            print(f"  Shape obs_result_counts: {obs_result_counts.shape}")
            print(f"  Pred Result Probs (mean): {pred_result_probs.mean():.4f}, (min): {pred_result_probs.min():.4f}, (max): {pred_result_probs.max():.4f}")
            print(f"  Obs Result Probs (mean): {obs_result_probs.mean():.4f}, (min): {obs_result_probs.min():.4f}, (max): {obs_result_probs.max():.4f}")
            print(f"  Pred Result Probs:\n{pred_result_probs}") # Print full array for results (usually small)
            print(f"  Obs Result Probs:\n{obs_result_probs}") # Print full array for results

        except KeyError as e:
            raise ValueError(f"Missing required variable in InferenceData: {e}")
        except Exception as e:
             raise ValueError(f"Error extracting data from InferenceData: {e}")

        # 3. Calculate Metrics
        metrics = {}

        # --- Poll Metrics ---
        if obs_poll_counts.shape[0] > 0:
            metrics["poll_mae"] = calculate_mae(pred_poll_probs, obs_poll_probs)
            metrics["poll_rmse"] = calculate_rmse(pred_poll_probs, obs_poll_probs)
            individual_poll_log_scores = calculate_log_score(pred_poll_probs, obs_poll_counts)
            metrics["poll_log_score"] = np.mean(individual_poll_log_scores[np.isfinite(individual_poll_log_scores)]) # Average finite scores
            metrics["poll_rps"] = calculate_rps(pred_poll_probs, obs_poll_probs)
            # Calculate calibration data for polls
            metrics["poll_calibration"] = calculate_calibration_data(pred_poll_probs, obs_poll_probs)

            # Identify worst fitting polls
            num_worst_polls = 5
            if len(individual_poll_log_scores) >= num_worst_polls:
                worst_poll_indices = np.argsort(individual_poll_log_scores)[-num_worst_polls:][::-1] # Indices of highest scores
                print(f"\n  Top {num_worst_polls} Worst Poll Log Scores (Indices):")
                for idx in worst_poll_indices:
                     if np.isfinite(individual_poll_log_scores[idx]):
                          print(f"    Poll Index {idx}: {individual_poll_log_scores[idx]:.2f}")
                     else:
                          print(f"    Poll Index {idx}: {individual_poll_log_scores[idx]}") # Print inf/nan
        else:
            print("Warning: No poll observations found in idata. Skipping poll metrics.")
            metrics["poll_mae"] = np.nan
            metrics["poll_rmse"] = np.nan
            metrics["poll_log_score"] = np.nan
            metrics["poll_rps"] = np.nan

        # --- Result Metrics ---
        if obs_result_counts.shape[0] > 0:
            metrics["result_mae"] = calculate_mae(pred_result_probs, obs_result_probs)
            metrics["result_rmse"] = calculate_rmse(pred_result_probs, obs_result_probs)
            individual_result_log_scores = calculate_log_score(pred_result_probs, obs_result_counts)
            metrics["result_log_score"] = np.mean(individual_result_log_scores[np.isfinite(individual_result_log_scores)])
            metrics["result_rps"] = calculate_rps(pred_result_probs, obs_result_probs)
            # Calculate calibration data for results
            metrics["result_calibration"] = calculate_calibration_data(pred_result_probs, obs_result_probs)

            # Identify worst fitting results
            num_worst_results = min(5, len(individual_result_log_scores))
            if num_worst_results > 0:
                 worst_result_indices = np.argsort(individual_result_log_scores)[-num_worst_results:][::-1]
                 print(f"\n  Top {num_worst_results} Worst Result Log Scores (Indices):")
                 # Get corresponding election dates for context
                 election_coords = idata.observed_data["observed_results"].coords["elections_observed"].values
                 for idx in worst_result_indices:
                      if np.isfinite(individual_result_log_scores[idx]):
                           print(f"    Result Index {idx} (Election: {election_coords[idx]}): {individual_result_log_scores[idx]:.2f}")
                      else:
                           print(f"    Result Index {idx} (Election: {election_coords[idx]}): {individual_result_log_scores[idx]}")
        else:
            print("Warning: No result observations found in idata. Skipping result metrics.")
            metrics["result_mae"] = np.nan
            metrics["result_rmse"] = np.nan
            metrics["result_log_score"] = np.nan
            metrics["result_rps"] = np.nan

        print("--- End Debugging Metric Calculation Inputs ---") # DEBUG
        return metrics


    def predict_latent_trajectory(self, idata: az.InferenceData, start_date: pd.Timestamp, end_date: pd.Timestamp) -> az.InferenceData:
        print("Warning: predict_latent_trajectory not yet implemented.")
        return az.InferenceData()

    def predict(self, oos_data: pd.DataFrame) -> az.InferenceData:
        print("Warning: predict (for polls) not yet implemented.")
        return az.InferenceData()

    def predict_history(self, elections_to_predict: List[str]) -> az.InferenceData:
        print("Warning: predict_history not yet implemented.")
        return az.InferenceData() 