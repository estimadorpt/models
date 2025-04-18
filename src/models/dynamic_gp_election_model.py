import os
from typing import Dict, List, Tuple, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from scipy.special import softmax
import time

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
    with shorter-term, cycle-specific GPs over the days-to-election countdown,
    and incorporating district-level effects.
    """

    def __init__(
        self,
        dataset: ElectionDataset,
        baseline_gp_config: Dict = {"kernel": "ExpQuad", "hsgp_m": [25], "hsgp_c": 1.5}, # Wrap hsgp_m in list
        short_term_gp_config: Dict = {"kernel": "Matern52", "hsgp_m": [25], "hsgp_c": 1.5}, # Wrap hsgp_m in list
        house_effect_sd_prior_scale: float = 0.1, # Adjusted default
        district_offset_sd_prior_scale: float = 0.1, # Adjusted default
        beta_sd_prior_scale: float = 0.5, # NEW sensitivity parameter scale
        cycle_gp_max_days: int = 90,
        **kwargs
    ):
        """
        Initializes the DynamicGPElectionModel.

        Args:
            dataset: An ElectionDataset object containing polls, results, etc.
            baseline_gp_config: Configuration for the baseline GP
            short_term_gp_config: Configuration for the short-term GP
            house_effect_sd_prior_scale: Prior scale for house effect standard deviation
            district_offset_sd_prior_scale: Prior scale for district effect standard deviation
            beta_sd_prior_scale: Prior scale for beta (sensitivity parameter)
            cycle_gp_max_days: Maximum days before election for the cycle GP
            **kwargs: Additional keyword arguments for model configuration.
                      Expected keys:
                        'baseline_gp_lengthscale', 'baseline_gp_kernel',
                        'cycle_gp_lengthscale', 'cycle_gp_kernel', 'cycle_gp_max_days',
                        'hsgp_m', 'hsgp_c'.
        """
        super().__init__(dataset, **kwargs)

        # Max days before election for the cycle GP
        self.cycle_gp_max_days = cycle_gp_max_days

        # Add election_date attribute initialization
        self.election_date = dataset.election_date # Store election date from dataset

        # Configuration for the GPs
        self.baseline_gp_config = baseline_gp_config
        self.short_term_gp_config = short_term_gp_config
        self.house_effect_sd_prior_scale = house_effect_sd_prior_scale
        self.district_offset_sd_prior_scale = district_offset_sd_prior_scale
        self.beta_sd_prior_scale = beta_sd_prior_scale # Store the new parameter

        # Ensure m is a list in the configs (handle potential overrides from kwargs if needed, though defaults are usually enough)
        if not isinstance(self.baseline_gp_config.get("hsgp_m"), (list, tuple)):
            print("Warning: baseline_gp_config['hsgp_m'] was not a list. Wrapping.")
            self.baseline_gp_config["hsgp_m"] = [self.baseline_gp_config["hsgp_m"]]
        if not isinstance(self.short_term_gp_config.get("hsgp_m"), (list, tuple)):
             print("Warning: short_term_gp_config['hsgp_m'] was not a list. Wrapping.")
             self.short_term_gp_config["hsgp_m"] = [self.short_term_gp_config["hsgp_m"]]

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
        # --- District Specific Attributes ---
        self.district_id = None                 # Factorized ID for districts
        self.result_district_idx = None         # Index mapping district result obs to district_id
        self.calendar_time_result_district_id = None # Index mapping district result obs to calendar_time
        self.result_cycle_district_idx = None    # Index mapping district result obs to cycle
        self.observed_district_result_indices = None # Row index of district result observations


    def _build_coords(self, polls: pd.DataFrame = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, Dict, np.ndarray]:
        """
        Build coordinates for the PyMC model, including calendar time, election cycles,
        districts, and days_to_election.
        Calculates mappings for polls, district-level results, and the full calendar time axis.

        Args:
            polls: Optional DataFrame of polls to use instead of self.polls_train.

        Returns:
            Tuple containing:
            - pollster_id
            - calendar_time_poll_id
            - poll_days_numeric
            - poll_cycle_idx
            - calendar_time_cycle_idx (Full mapping for calendar_time)
            - calendar_time_days_numeric (Full mapping for calendar_time)
            - district_id (Factorized district IDs)
            - result_district_idx (Mapping district result obs to district ID)
            - calendar_time_result_district_id (Mapping district result obs to calendar time)
            - result_cycle_district_idx (Mapping district result obs to cycle)
            - COORDS dictionary
            - observed_district_result_indices (Indices of the district results used)
        """
        data_polls = polls if polls is not None else self.polls_train
        historical_results_district = self.dataset.results_mult_district # Use district results

        # --- Create Unified Calendar Time Coordinate ---
        poll_dates_dt = pd.to_datetime(data_polls['date']).unique()
        all_election_dates_dt = pd.to_datetime(self.all_election_dates).unique()
        # Include district result dates in calendar time coord calculation
        district_result_dates_dt = pd.to_datetime(historical_results_district['election_date']).unique()
        unique_dates = pd.to_datetime(
             np.union1d(np.union1d(poll_dates_dt, all_election_dates_dt), district_result_dates_dt)
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
            cycle_end_date = min([d for d in cycle_boundaries_dt if d >= date], default=cycle_boundaries_dt[-1])
            cycle_idx = date_str_to_cycle_idx[cycle_end_date.strftime('%Y-%m-%d')]
            days_numeric = (cycle_end_date - date).days
            days_numeric_clipped = np.clip(days_numeric, 0, self.cycle_gp_max_days) # Clip days for cycle GP input (if used)
            calendar_time_cycle_idx_list.append(cycle_idx)
            calendar_time_days_list.append(days_numeric_clipped)
        self.calendar_time_cycle_idx = np.array(calendar_time_cycle_idx_list, dtype=int)
        self.calendar_time_days_numeric = np.array(calendar_time_days_list, dtype=float) # Used by calendar time GP

        # Calculate days to election for polls (as before)
        if 'countdown' not in data_polls.columns:
             raise ValueError("Missing 'countdown' column in polls data, needed for indexing.")
        poll_days_numeric_raw = data_polls['countdown'].values
        # Clipping here is for cycle GP input (if used), keep original countdown if needed elsewhere
        poll_days_numeric = np.clip(poll_days_numeric_raw, 0, self.cycle_gp_max_days).astype(float)
        self.poll_days_numeric = poll_days_numeric # Still used by calendar time GP indexing

        # Map polls to their cycle index (as before)
        poll_election_dates_str = pd.to_datetime(data_polls['election_date']).dt.strftime('%Y-%m-%d')
        poll_cycle_idx = poll_election_dates_str.map(date_str_to_cycle_idx).values
        if np.isnan(poll_cycle_idx).any():
             missing_poll_cycles = data_polls.loc[np.isnan(poll_cycle_idx), 'election_date'].unique()
             raise ValueError(f"Could not map some poll election dates to cycles: {missing_poll_cycles}")
        self.poll_cycle_idx = poll_cycle_idx.astype(int)

        # --- Map Polls to Calendar Time ---
        poll_dates_dt = pd.to_datetime(data_polls['date'])
        self.calendar_time_poll_id = poll_dates_dt.map(date_to_calendar_index).values.astype(int)

        # --- Map District Results to Calendar Time, Cycle, and District ID ---
        if historical_results_district.empty:
            print("Warning: No historical district results found. District effects might not be well estimated.")
            self.calendar_time_result_district_id = np.array([], dtype=int)
            self.result_cycle_district_idx = np.array([], dtype=int)
            self.result_district_idx = np.array([], dtype=int)
            self.observed_district_result_indices = np.array([], dtype=int)
            district_coords = []
        else:
            historical_results_dates_dt = pd.to_datetime(historical_results_district['election_date'])
            self.calendar_time_result_district_id = historical_results_dates_dt.map(date_to_calendar_index).values.astype(int)

            result_election_dates_str = historical_results_dates_dt.dt.strftime('%Y-%m-%d')
            self.result_cycle_district_idx = result_election_dates_str.map(date_str_to_cycle_idx).values
            if np.isnan(self.result_cycle_district_idx).any():
                missing_result_cycles = historical_results_district.loc[np.isnan(self.result_cycle_district_idx), 'election_date'].unique()
                raise ValueError(f"Could not map some district result election dates to cycles: {missing_result_cycles}")
            self.result_cycle_district_idx = self.result_cycle_district_idx.astype(int)

            # Factorize districts from the historical results
            if 'Circulo' not in historical_results_district.columns:
                 raise ValueError("Missing 'Circulo' column in district results data.")
            self.district_id, district_coords = historical_results_district['Circulo'].factorize(sort=True)
            self.result_district_idx = self.district_id # Direct mapping from the factorized result
            self.observed_district_result_indices = historical_results_district.index.values # Use DF index

        # --- Define Coordinates ---
        COORDS = {
            "observations": data_polls.index, # Poll observations
            "parties_complete": self.political_families,
            "calendar_time": unique_dates.strftime('%Y-%m-%d'),
            "election_cycles": cycle_names,
            "districts": district_coords if 'district_coords' in locals() else [], # Use factorized districts
            # Dimension for observed district results (rows in results_mult_district)
            "elections_observed_district": self.observed_district_result_indices if self.observed_district_result_indices.size > 0 else [],
        }

        self.pollster_id, COORDS["pollsters"] = data_polls["pollster"].factorize(sort=True)

        print("\n=== MODEL COORDINATES (with Districts) ===")
        for key, value in COORDS.items():
            print(f"{key}: length={len(value)}")
        print(f"DEBUG: district_id shape: {self.district_id.shape if self.district_id is not None else 'None'}")
        print(f"DEBUG: result_district_idx shape: {self.result_district_idx.shape if self.result_district_idx is not None else 'None'}")
        print(f"DEBUG: calendar_time_result_district_id shape: {self.calendar_time_result_district_id.shape if self.calendar_time_result_district_id is not None else 'None'}")
        print(f"DEBUG: result_cycle_district_idx shape: {self.result_cycle_district_idx.shape if self.result_cycle_district_idx is not None else 'None'}")
        print(f"DEBUG: observed_district_result_indices shape: {self.observed_district_result_indices.shape if self.observed_district_result_indices is not None else 'None'}")


        return (self.pollster_id, self.calendar_time_poll_id,
                self.poll_days_numeric, self.poll_cycle_idx,
                self.calendar_time_cycle_idx, self.calendar_time_days_numeric,
                self.district_id, self.result_district_idx,
                self.calendar_time_result_district_id, self.result_cycle_district_idx,
                COORDS, self.observed_district_result_indices)


    def _build_data_containers(self,
                              polls: pd.DataFrame = None
                             ) -> Dict[str, pm.Data]:
        """
        Build the data containers for the PyMC model, using district-level results.
        """
        current_polls = polls if polls is not None else self.polls_train
        results_district = self.dataset.results_mult_district # Use district results

        # Ensure indices are set
        if self.pollster_id is None or self.calendar_time_poll_id is None or \
           self.poll_cycle_idx is None or self.calendar_time_cycle_idx is None or \
           self.calendar_time_days_numeric is None or self.result_district_idx is None or \
           self.calendar_time_result_district_id is None or self.result_cycle_district_idx is None:
            raise ValueError("Indices/mappings not set. Ensure _build_coords runs first.")

        # Prepare district result data
        if results_district.empty:
            print("Warning: results_mult_district is empty. Creating empty data containers for district results.")
            results_N_district = np.array([], dtype=float)
            observed_results_district = np.empty((0, len(self.political_families)), dtype=float)
            # Ensure index arrays are also empty but correctly typed
            calendar_time_result_district_idx_data = np.array([], dtype=int)
            result_cycle_district_idx_data = np.array([], dtype=int)
            result_district_idx_data = np.array([], dtype=int)
        else:
            # Check alignment of indices with the dataframe used in _build_coords
            if len(self.observed_district_result_indices) != len(results_district):
                 raise ValueError(f"Mismatch between observed_district_result_indices ({len(self.observed_district_result_indices)}) and results_mult_district ({len(results_district)}). Check _build_coords.")
            
            # Ensure results_mult_district is aligned with the indices before extracting data
            # Assuming results_mult_district index matches observed_district_result_indices
            results_N_district = results_district.loc[self.observed_district_result_indices, "sample_size"].to_numpy()
            observed_results_district = results_district.loc[self.observed_district_result_indices, self.political_families].to_numpy()
            # Use the index arrays directly calculated in _build_coords
            calendar_time_result_district_idx_data = self.calendar_time_result_district_id
            result_cycle_district_idx_data = self.result_cycle_district_idx
            result_district_idx_data = self.result_district_idx

        print("\n--- Debugging Data Containers (District Results) ---")
        print(f"  Shape observed_N_results_district: {results_N_district.shape}")
        print(f"  Shape observed_results_district: {observed_results_district.shape}")
        print(f"  Shape calendar_time_result_district_idx_data: {calendar_time_result_district_idx_data.shape}")
        print(f"  Shape result_cycle_district_idx_data: {result_cycle_district_idx_data.shape}")
        print(f"  Shape result_district_idx_data: {result_district_idx_data.shape}")
        print("--- End Debugging Data Containers ---")


        data_containers = dict(
            # --- Poll Indices ---
            calendar_time_poll_idx=pm.Data("calendar_time_poll_idx", self.calendar_time_poll_id, dims="observations"),
            pollster_idx=pm.Data("pollster_idx", self.pollster_id, dims="observations"),
            poll_cycle_idx=pm.Data("poll_cycle_idx", self.poll_cycle_idx, dims="observations"),
            # --- Result Indices (District) ---
            calendar_time_result_district_idx=pm.Data("calendar_time_result_district_idx", calendar_time_result_district_idx_data, dims="elections_observed_district"),
            result_cycle_district_idx=pm.Data("result_cycle_district_idx", result_cycle_district_idx_data, dims="elections_observed_district"),
            result_district_idx=pm.Data("result_district_idx", result_district_idx_data, dims="elections_observed_district"),
            # --- Calendar Time Indices ---
            calendar_time_cycle_idx=pm.Data("calendar_time_cycle_idx", self.calendar_time_cycle_idx, dims="calendar_time"),
            # --- GP Inputs ---
            poll_days_numeric=pm.Data("poll_days_numeric", self.poll_days_numeric, dims="observations"), # Used for indexing calendar time GPs
            calendar_time_days_numeric=pm.Data("calendar_time_days_numeric", self.calendar_time_days_numeric, dims="calendar_time"), # Input to calendar time GPs
            # --- Observed Poll Data ---
            observed_N_polls=pm.Data("observed_N_polls", current_polls["sample_size"].to_numpy(), dims="observations"),
            observed_polls=pm.Data("observed_polls", current_polls[self.political_families].to_numpy(), dims=("observations", "parties_complete")),
            # --- Observed Result Data (District) ---
            observed_N_results_district=pm.Data("observed_N_results_district", results_N_district, dims="elections_observed_district"),
            observed_results_district=pm.Data("observed_results_district", observed_results_district, dims=("elections_observed_district", "parties_complete")),
        )
        return data_containers


    def build_model(self, polls: pd.DataFrame = None) -> pm.Model:
        """
        Build the PyMC model with two GPs over calendar time, house effects,
        and dynamic district effects (base offset + beta sensitivity).
        """
        # Unpacking expects 12 values now
        (
            self.pollster_id, self.calendar_time_poll_id,
            self.poll_days_numeric, self.poll_cycle_idx,
            self.calendar_time_cycle_idx, self.calendar_time_days_numeric,
            self.district_id, self.result_district_idx,
            self.calendar_time_result_district_id, self.result_cycle_district_idx,
            self.coords, self.observed_district_result_indices # Renamed from observed_election_indices
        ) = self._build_coords(polls)

        with pm.Model(coords=self.coords) as model:
            data_containers = self._build_data_containers(polls)
            # Store the data containers dictionary as an instance attribute
            self.data_containers = data_containers

            # --------------------------------------------------------
            #        1. BASELINE GP (Long Trend over Calendar Time)
            # --------------------------------------------------------
            baseline_gp_ls = pm.LogNormal("baseline_gp_ls", mu=np.log(365*2.5), sigma=0.3)
            baseline_gp_amp_sd = pm.HalfNormal("baseline_gp_amp_sd", sigma=0.2)

            if self.baseline_gp_config["kernel"] == "Matern52":
                 cov_func_baseline = baseline_gp_amp_sd**2 * pm.gp.cov.Matern52(input_dim=1, ls=baseline_gp_ls)
            elif self.baseline_gp_config["kernel"] == "ExpQuad":
                 cov_func_baseline = baseline_gp_amp_sd**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=baseline_gp_ls)
            else: raise ValueError(f"Unsupported Baseline GP kernel: {self.baseline_gp_config['kernel']}")

            baseline_hsgp_m = self.baseline_gp_config["hsgp_m"]
            baseline_hsgp_c = self.baseline_gp_config["hsgp_c"]
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
            # Ensure short_term_gp_config has the right keys from __init__
            short_term_gp_ls_prior_mu = self.short_term_gp_config.get("lengthscale", 30) # Default 30 days
            short_term_gp_amp_sd_prior_scale = self.short_term_gp_config.get("amp_sd", 0.1) # Default 0.1

            short_term_gp_ls = pm.LogNormal("short_term_gp_ls", mu=np.log(short_term_gp_ls_prior_mu), sigma=0.5)
            short_term_gp_amp_sd = pm.HalfNormal("short_term_gp_amp_sd", sigma=short_term_gp_amp_sd_prior_scale)

            short_term_gp_kernel = self.short_term_gp_config.get("kernel", "Matern52") # Default Matern52

            if short_term_gp_kernel == "Matern52":
                 cov_func_short_term = short_term_gp_amp_sd**2 * pm.gp.cov.Matern52(input_dim=1, ls=short_term_gp_ls)
            elif short_term_gp_kernel == "ExpQuad":
                 cov_func_short_term = short_term_gp_amp_sd**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=short_term_gp_ls)
            else: raise ValueError(f"Unsupported Short-Term GP kernel: {short_term_gp_kernel}")

            # Use HSGP parameters from short_term_gp_config
            short_term_hsgp_m = self.short_term_gp_config.get("hsgp_m", [25]) # Default 25 basis vectors
            short_term_hsgp_c = self.short_term_gp_config.get("hsgp_c", 1.5) # Default expansion factor

            self.gp_short_term_time = pm.gp.HSGP(
                cov_func=cov_func_short_term, m=short_term_hsgp_m, c=short_term_hsgp_c
            )
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

            # --- Combine GPs for National Trend ---
            national_trend_pt = pm.Deterministic("national_trend_pt",
                                                  baseline_effect_calendar + short_term_effect_calendar,
                                                  dims=("calendar_time", "parties_complete"))

            # --- Calculate Softmax Probabilities (National) ---
            latent_popularity_national = pm.Deterministic(
                "latent_popularity_national",
                pm.math.softmax(national_trend_pt, axis=1), # Apply softmax along party axis (axis=1)
                dims=("calendar_time", "parties_complete")
            )
            # --- End Softmax Calculation ---

            # Calculate average national trend per party (across calendar time)
            national_avg_trend_p = national_trend_pt.mean(axis=0, keepdims=True) # Keep dims for broadcasting
            # national_avg_trend_p shape will be (1, n_parties)

            # --------------------------------------------------------
            #          3. HOUSE EFFECTS (Pollster Bias)
            # --------------------------------------------------------
            house_effects_sd = pm.HalfNormal("house_effects_sd", sigma=self.house_effect_sd_prior_scale, dims="parties_complete")
            # Non-centered parameterization
            house_effects_raw = pm.Normal("house_effects_raw", mu=0, sigma=1,
                                          dims=("pollsters", "parties_complete"))
            house_effects = pm.Deterministic("house_effects",
                                             house_effects_raw * house_effects_sd[None, :],
                                             dims=("pollsters", "parties_complete"))
            # Ensure house effects sum to zero across parties for each pollster
            house_effect_pollster_adj = pm.Deterministic("house_effect_pollster_adj",
                                                    house_effects - house_effects.mean(axis=1, keepdims=True),
                                             dims=("pollsters", "parties_complete"))

            # --------------------------------------------------------
            #          4. DISTRICT EFFECTS (Dynamic: Base Offset + Beta)
            # --------------------------------------------------------
            # Check if districts coordinate exists and has members
            has_districts = "districts" in self.coords and len(self.coords["districts"]) > 0
            if has_districts:
                 # --- Base Offset ---
                 sigma_base_offset_p = pm.HalfNormal("sigma_base_offset", 
                                                     sigma=self.district_offset_sd_prior_scale,
                                                     dims="parties_complete")
                 # Define with shape (parties, districts) for easier broadcasting later
                 base_offset_raw_p = pm.Normal("base_offset_raw", mu=0, sigma=1,
                                                dims=("parties_complete", "districts"))
                 # Apply scale and ensure sums to zero across parties for each district
                 base_offset_p_noncentered = base_offset_raw_p * sigma_base_offset_p[:, None]
                 base_offset_p = pm.Deterministic("base_offset_p", 
                                                   base_offset_p_noncentered - base_offset_p_noncentered.mean(axis=0, keepdims=True), 
                                                   dims=("parties_complete", "districts"))

                 # --- Beta Sensitivity ---
                 sigma_beta_p = pm.HalfNormal("sigma_beta", 
                                              sigma=self.beta_sd_prior_scale, # Use new prior scale
                                              dims="parties_complete")
                 # Define beta centered around 1, shape (parties, districts)
                 beta_raw_p = pm.Normal("beta_raw", mu=0, sigma=1, # Non-centered around zero
                                         dims=("parties_complete", "districts"))
                 # Center beta around 1 after scaling
                 beta_p = pm.Deterministic("beta_p", 
                                            1 + beta_raw_p * sigma_beta_p[:, None],
                                            dims=("parties_complete", "districts"))
                 # Note: We are NOT forcing beta to sum to zero or one across parties.
                 # Each party's sensitivity in a district is modeled relative to its own national trend.

            # --------------------------------------------------------
            #          5. LATENT VOTE INTENTIONS
            # --------------------------------------------------------

            # --- Latent intention for Polls ---
            # National trend indexed by poll observation time
            national_trend_polls = national_trend_pt[data_containers["calendar_time_poll_idx"], :]
            # House effect indexed by poll observation pollster
            house_effect_polls = house_effect_pollster_adj[data_containers["pollster_idx"], :]
            
            latent_polls = national_trend_polls + house_effect_polls
            p_polls = pm.Deterministic("p_polls", pm.math.softmax(latent_polls, axis=1), dims=("observations", "parties_complete"))

            # --- Latent intention for District Results ---
            if has_districts and data_containers["observed_results_district"].get_value(borrow=True).shape[0] > 0:
                 # National trend indexed by district result observation time
                 national_trend_results = national_trend_pt[data_containers["calendar_time_result_district_idx"], :]

                 # --- Calculate Dynamic District Adjustment ---
                 # Deviation of national trend from its average at result times
                 # national_avg_trend_p has shape (1, n_parties) due to keepdims=True
                 national_dev_results = national_trend_results - national_avg_trend_p # Broadcasting works

                 # Index base offset and beta for each result
                 # base_offset_p has shape (parties, districts)
                 # beta_p has shape (parties, districts)
                 # result_district_idx provides the district index for each observation
                 base_offset_term = base_offset_p[:, data_containers["result_district_idx"]] # Shape (parties, results_obs)
                 beta_term = beta_p[:, data_containers["result_district_idx"]]          # Shape (parties, results_obs)

                 # Calculate the adjustment term: beta * deviation
                 # Need to transpose national_dev_results to (parties, results_obs) for elementwise mult
                 # (beta_term - 1): Shape (parties, results_obs)
                 # national_dev_results.T: Shape (parties, results_obs)
                 dynamic_adjustment_term = (beta_term - 1) * national_dev_results.T

                 # Combine base offset and dynamic adjustment
                 # Both terms have shape (parties, results_obs)
                 district_adjustment_results = base_offset_term + dynamic_adjustment_term
                 
                 # Transpose back to (results_obs, parties) to add to national_trend_results
                 district_adjustment_results = district_adjustment_results.T

                 # --- Combine for final latent results ---
                 latent_results = national_trend_results + district_adjustment_results
                 p_results_district = pm.Deterministic("p_results_district",
                                                      pm.math.softmax(latent_results, axis=1),
                                                      dims=("elections_observed_district", "parties_complete"))
            else:
                 # Define placeholder if no districts or no results
                 p_results_district = None # Or handle differently if needed later

            # --------------------------------------------------------
            #          6. LIKELIHOODS
            # --------------------------------------------------------

            # --- Poll Likelihood (Dirichlet Multinomial) --- 
            concentration_polls = pm.Gamma("concentration_polls", alpha=100, beta=0.1) # Reintroduce concentration
            pm.DirichletMultinomial(
                "poll_likelihood", # Renamed variable from original N_approve
                n=data_containers["observed_N_polls"],
                a=concentration_polls * p_polls, # Use concentration * probability
                observed=data_containers["observed_polls"], 
                dims=("observations", "parties_complete"),
            )

            # --- District Result Likelihood (Dirichlet Multinomial) --- 
            # Only add if there are districts and observed district results and p_results_district is defined
            if has_districts and data_containers["observed_results_district"].get_value(borrow=True).shape[0] > 0 and p_results_district is not None:
                 concentration_results = pm.Gamma("concentration_results", alpha=100, beta=0.05) # Reintroduce concentration (can be separate or shared)
                 pm.DirichletMultinomial(
                     "observed_district", # Name matches data container, but it's the likelihood RV.
                                          # If this causes issues, rename to observed_district_lik
                     n=data_containers["observed_N_results_district"],
                     a=concentration_results * p_results_district, # Use concentration * probability
                     observed=data_containers["observed_results_district"],
                     dims=("elections_observed_district", "parties_complete"),
                 )

        return model

    def posterior_predictive_check(self, idata: az.InferenceData, extend_idata: bool = True) -> az.InferenceData | Dict:
        """
        Performs posterior predictive checks on the poll and district result likelihoods.

        Generates samples from the posterior predictive distribution for observed
        variables (poll counts 'N_approve' and district result counts 'R_district').

        Args:
            idata: The InferenceData object containing the posterior trace.
            extend_idata: If True (default), adds the posterior predictive samples
                          to the input InferenceData object and returns it.
                          If False, returns a dictionary containing the samples.

        Returns:
            The input InferenceData object extended with posterior predictive samples
            (if extend_idata=True), or a dictionary containing the samples.

        Raises:
            ValueError: If the model hasn't been built yet or idata is invalid.
            KeyError: If 'R_district' is not in the model (e.g., no district data).
        """
        if self.model is None:
            raise ValueError("Model must be built before running PPC. Call build_model() first.")
        if idata is None or "posterior" not in idata:
             raise ValueError("Valid InferenceData object with posterior samples required.")

        # --- Variables to Sample ---
        vars_to_sample_observed = []
        vars_to_sample_predictions = []

        if "poll_likelihood" in self.model.named_vars:
            vars_to_sample_observed.append("poll_likelihood")
        if "p_polls" in self.model.deterministics:
            vars_to_sample_predictions.append("p_polls")

        if "observed_district" in self.model.named_vars:
             vars_to_sample_observed.append("observed_district")
        if "p_results_district" in self.model.deterministics:
             vars_to_sample_predictions.append("p_results_district")
             print("DEBUG PPC: Added 'p_results_district' to prediction sampling list.") # Debug print

        print(f"DEBUG PPC: Observed RVs to sample: {vars_to_sample_observed}")
        print(f"DEBUG PPC: Prediction RVs to sample: {vars_to_sample_predictions}")

        # --- Sample Observed Nodes --- 
        # These represent the actual outcome variables (e.g., counts)
        if vars_to_sample_observed:
            print("Sampling observed nodes for PPC...")
            with self.model: # Re-enter the model context
                try:
                     idata = pm.sample_posterior_predictive(
                         idata,
                         var_names=vars_to_sample_observed,
                         predictions=False, # Sample the OBSERVED nodes
                         extend_inferencedata=True, # <<< Add directly to idata
                     )
                     print("Observed nodes sampled successfully.")
                except Exception as e:
                     print(f"Error sampling observed nodes for PPC: {e}")
                     # Decide whether to raise or just warn
                     # raise e # Or return idata as is?

        # --- Sample Prediction Nodes --- 
        # These often represent underlying probabilities or means
        if vars_to_sample_predictions:
             print("Sampling prediction nodes for PPC...")
             with self.model:
                 try:
                      idata = pm.sample_posterior_predictive(
                           idata,
                           var_names=vars_to_sample_predictions,
                           predictions=True, # Sample the DETERMINISTIC nodes
                           extend_inferencedata=True, # <<< Add directly to idata
                      )
                      print("Prediction nodes sampled successfully.")
                 except Exception as e:
                      print(f"Error sampling prediction nodes for PPC: {e}")
                      # Decide whether to raise or just warn

        print("\nPosterior predictive check complete. Samples added to InferenceData.")
        # Now `idata` should contain the samples in its `posterior_predictive` group.

        # The original `extend_idata` argument is less relevant now, as we always extend.
        return idata


    def calculate_fit_metrics(self, idata: az.InferenceData) -> Tuple[Dict[str, float], az.InferenceData]:
        """
        Calculates various goodness-of-fit metrics using posterior samples.
        Compares model predictions against observed poll and district election result data.
        Args:
            idata: InferenceData object containing posterior and observed_data groups.
        Returns:
            A tuple containing:
            - A dictionary with calculated metrics (poll_mae, poll_rmse, ..., result_district_mae, ...)
            - The input idata object (unchanged in this version).
        Raises:
            ValueError: If required data groups or variables are missing from idata.
        """
        # 1. Check for required groups
        # REMOVED Check/call for posterior_predictive_check
        # if "posterior_predictive" not in idata:
        #    print("Posterior predictive samples not found in idata. Running posterior_predictive_check...")
        #    try:
        #        idata = self.posterior_predictive_check(idata, extend_idata=True) # Modifies idata in-place
        #    except Exception as e:
        #         raise ValueError(f"Failed to generate posterior predictive samples: {e}")

        if "posterior" not in idata:
            raise ValueError("posterior group is required in InferenceData to calculate metrics.")
        if "observed_data" not in idata:
            raise ValueError("observed_data group is required in InferenceData to calculate metrics.")
        # if "posterior_predictive" not in idata: # Double check after PPC run
        #    raise ValueError("posterior_predictive group still missing after running check.")

        # --- START DEBUG --- 
        print("\nDEBUG METRICS: Checking idata contents...")
        # if hasattr(idata, 'posterior_predictive'):
        #    print(f"  idata.posterior_predictive keys: {list(idata.posterior_predictive.keys())}")
        # else:
        #    print("  idata.posterior_predictive group not found.")
        if hasattr(idata, 'posterior'):
            print(f"  idata.posterior keys: {list(idata.posterior.keys())}")
        else:
             print("  idata.posterior group not found.")
        if hasattr(idata, 'observed_data'):
            print(f"  idata.observed_data keys: {list(idata.observed_data.keys())}")
        else:
            print("  idata.observed_data group not found.")
        # --- END DEBUG ---

        # 2. Extract Predictions and Observed Data
        metrics = {}
        print("\n--- Debugging Metric Calculation Inputs (District Model) ---")

        # --- Polls --- 
        try:
            # Get predicted probabilities directly from posterior
            if "p_polls" not in idata.posterior:
                 raise KeyError("'p_polls' missing from posterior. Cannot calculate probability-based metrics.")
            posterior_poll_prob_da = idata.posterior["p_polls"] # Shape (chain, draw, obs, party)
            # Calculate mean probability for MAE, RMSE etc.
            pred_poll_probs = posterior_poll_prob_da.mean(dim=["chain", "draw"]).values

            # Use observed data name 'observed_polls' for counts
            if "poll_likelihood" not in idata.observed_data:
                 raise KeyError("'poll_likelihood' missing from observed_data.")
            obs_poll_counts = idata.observed_data["poll_likelihood"].values
            # Ensure counts are integer for log_score
            obs_poll_counts_int = obs_poll_counts.astype(int)

            # Get total poll counts directly from model's data containers
            if "observed_N_polls" not in self.data_containers:
                 raise KeyError("'observed_N_polls' not found in self.data_containers")
            # Ensure obs_poll_n is a numpy array and float for division
            obs_poll_n = self.data_containers["observed_N_polls"].get_value() # Keep as original dtype (likely int)
            # We don't need obs_poll_probs here anymore
            # with np.errstate(divide='ignore', invalid='ignore'):
            #    obs_poll_probs = obs_poll_counts / obs_poll_n[:, np.newaxis]
            # obs_poll_probs = np.nan_to_num(obs_poll_probs)
            
            print("\n[Poll Data]")
            print(f"  Shape pred_poll_probs (mean): {pred_poll_probs.shape}")
            print(f"  Shape posterior_poll_prob_da (samples): {posterior_poll_prob_da.shape}")
            print(f"  Shape obs_poll_counts: {obs_poll_counts.shape}")

            if obs_poll_counts.shape[0] > 0:
                # Pass counts and N to metric functions
                metrics["poll_mae"] = calculate_mae(pred_poll_probs, obs_poll_counts_int, obs_poll_n)
                metrics["poll_rmse"] = calculate_rmse(pred_poll_probs, obs_poll_counts_int, obs_poll_n)
                # Use integer counts for log_score
                individual_poll_log_scores = calculate_log_score(pred_poll_probs, obs_poll_counts_int) 
                metrics["poll_log_score"] = np.mean(individual_poll_log_scores[np.isfinite(individual_poll_log_scores)])
                metrics["poll_rps"] = calculate_rps(pred_poll_probs, obs_poll_counts_int, obs_poll_n)
                # Pass the full posterior distribution, counts, and N for calibration
                metrics["poll_calibration"] = calculate_calibration_data(posterior_poll_prob_da, obs_poll_counts_int, obs_poll_n)
            else:
                print("Warning: No poll observations found. Skipping poll metrics.")
                metrics["poll_mae"] = np.nan
                # ... (set other poll metrics to nan/default) ...
                metrics["poll_calibration"] = {"mean_predicted_prob": [], "mean_observed_prob": [], "bin_counts": [], "bin_edges": []} # Default empty

        except KeyError as e:
            print(f"Warning: Missing poll variable: {e}. Skipping poll metrics.")
            metrics["poll_mae"] = np.nan
            # ... (set other poll metrics to nan/default) ...
            metrics["poll_calibration"] = {"mean_predicted_prob": [], "mean_observed_prob": [], "bin_counts": [], "bin_edges": []} # Default empty
        except Exception as e:
            print(f"Error calculating poll metrics: {e}")
            metrics["poll_mae"] = np.nan
            # ... (set other poll metrics to nan/default) ...
            metrics["poll_calibration"] = {"mean_predicted_prob": [], "mean_observed_prob": [], "bin_counts": [], "bin_edges": []} # Default empty

        # --- District Results --- 
        # Check if district probabilities and observed counts are available
        district_probs_available = "p_results_district" in idata.posterior
        district_counts_available = "observed_district" in idata.observed_data

        if district_probs_available and district_counts_available:
            try:
                posterior_result_prob_da = idata.posterior["p_results_district"] # Shape (chain, draw, obs, party)
                # Calculate mean probability for MAE, RMSE etc.
                pred_result_probs = posterior_result_prob_da.mean(dim=["chain", "draw"]).values
                
                obs_result_counts = idata.observed_data["observed_district"].values
                # Ensure counts are integer for log_score
                obs_result_counts_int = obs_result_counts.astype(int)

                if "observed_N_results_district" not in self.data_containers:
                     raise KeyError("'observed_N_results_district' not found in self.data_containers")
                # Ensure obs_result_n is a numpy array
                obs_result_n = self.data_containers["observed_N_results_district"].get_value() # Keep as original dtype
                # We don't need obs_result_probs here anymore
                # with np.errstate(divide='ignore', invalid='ignore'):
                #     obs_result_probs = obs_result_counts / obs_result_n[:, np.newaxis]
                # obs_result_probs = np.nan_to_num(obs_result_probs)

                print("\n[District Result Data]")
                print(f"  Shape pred_result_probs (mean, district): {pred_result_probs.shape}")
                print(f"  Shape posterior_result_prob_da (samples): {posterior_result_prob_da.shape}")
                print(f"  Shape obs_result_counts (district): {obs_result_counts.shape}")

                if obs_result_counts.shape[0] > 0:
                    # Pass counts and N to metric functions
                    metrics["result_district_mae"] = calculate_mae(pred_result_probs, obs_result_counts_int, obs_result_n)
                    metrics["result_district_rmse"] = calculate_rmse(pred_result_probs, obs_result_counts_int, obs_result_n)
                    # Use integer counts for log_score
                    individual_result_log_scores = calculate_log_score(pred_result_probs, obs_result_counts_int)
                    metrics["result_district_log_score"] = np.mean(individual_result_log_scores[np.isfinite(individual_result_log_scores)])
                    metrics["result_district_rps"] = calculate_rps(pred_result_probs, obs_result_counts_int, obs_result_n)
                    # Pass the full posterior distribution, counts, and N for calibration
                    metrics["result_district_calibration"] = calculate_calibration_data(posterior_result_prob_da, obs_result_counts_int, obs_result_n)
                else:
                    print("Warning: No district result observations found. Skipping district result metrics.")
                    metrics["result_district_mae"] = np.nan
                    # ... (set other district metrics to nan/default) ...
                    metrics["result_district_calibration"] = {"mean_predicted_prob": [], "mean_observed_prob": [], "bin_counts": [], "bin_edges": []} # Default empty
            except KeyError as e:
                print(f"Warning: Missing district result variable: {e}. Skipping district result metrics.")
                metrics["result_district_mae"] = np.nan
                # ... (set other district metrics to nan/default) ...
                metrics["result_district_calibration"] = {"mean_predicted_prob": [], "mean_observed_prob": [], "bin_counts": [], "bin_edges": []} # Default empty
            except Exception as e:
                 print(f"Error calculating district result metrics: {e}")
                 metrics["result_district_mae"] = np.nan
                 # ... (set other district metrics to nan/default) ...
                 metrics["result_district_calibration"] = {"mean_predicted_prob": [], "mean_observed_prob": [], "bin_counts": [], "bin_edges": []} # Default empty
        else:
            missing_var = "'p_results_district' in posterior" if not district_probs_available else "'observed_district' in observed_data"
            print(f"Warning: {missing_var} not found. Skipping district result metrics.")
            metrics["result_district_mae"] = np.nan
            # ... (set other district metrics to nan/default) ...
            metrics["result_district_calibration"] = {"mean_predicted_prob": [], "mean_observed_prob": [], "bin_counts": [], "bin_edges": []} # Default empty

        print("--- End Debugging Metric Calculation Inputs ---")
        # Return the original idata, as we didn't modify it here
        return metrics, idata


    def predict_latent_trajectory(self, idata: az.InferenceData, start_date: pd.Timestamp, end_date: pd.Timestamp) -> az.InferenceData:
        # This method currently returns the *national* trajectory.
        # Needs adaptation if district-specific trajectories are required.
        print("Warning: predict_latent_trajectory currently returns NATIONAL trajectory.")
        # Placeholder for future implementation if needed
        return az.InferenceData()

    def predict(self, oos_data: pd.DataFrame) -> az.InferenceData:
        # This likely needs adaptation for district model if predicting polls is still desired.
        print("Warning: predict (for polls) not yet implemented/adapted for district model.")
        return az.InferenceData()

    def predict_history(self, elections_to_predict: List[str]) -> az.InferenceData:
        # Needs adaptation for district model.
        print("Warning: predict_history not yet implemented/adapted for district model.")
        return az.InferenceData()

    def get_latent_popularity(self, idata: az.InferenceData, target_date: pd.Timestamp = None, date_mode: str = 'election_day', district: Optional[str] = None) -> Optional[xr.DataArray]:
        """
        Extracts the posterior distribution of latent popularity at a specific time point,
        optionally adjusted for a specific district.

        Args:
            idata (az.InferenceData): InferenceData object with posterior samples.
            target_date (pd.Timestamp, optional): Specific date. Used if date_mode is None or 'specific_date'.
            date_mode (str): Defines the time point: 'election_day', 'last_poll', 'today', or 'specific_date'.
            district (str, optional): Name of the district to get adjusted popularity for.
                                      If None, returns the national latent popularity.

        Returns:
            xr.DataArray or None: Posterior samples of latent popularity.
                          Dimensions: (chain, draw, parties_complete).
                                  Returns None on error or if data/coords are missing.
        """
        if idata is None or 'posterior' not in idata:
            print("Error: Valid InferenceData with posterior samples required."); return None

        # --- Determine Target Date based on mode ---
        actual_target_date = None
        if date_mode == 'election_day':
            # Assumes self.election_date is set correctly in the model/facade instance
            if hasattr(self, 'election_date') and self.election_date:
                 actual_target_date = pd.Timestamp(self.election_date).normalize()
                 print(f"DEBUG get_latent: Using election_day: {actual_target_date.date()}")
            else:
                 print("Error: election_date not available in model instance for 'election_day' mode."); return None
        elif date_mode == 'last_poll':
             # Find last poll date from the dataset used for the model coords
             if hasattr(self.dataset, 'polls_train') and not self.dataset.polls_train.empty:
                  last_poll_date = pd.to_datetime(self.dataset.polls_train['date']).max()
                  actual_target_date = pd.Timestamp(last_poll_date).normalize()
                  print(f"DEBUG get_latent: Using last_poll date: {actual_target_date.date()}")
             else:
                  print("Error: Cannot determine last poll date from dataset."); return None
        elif date_mode == 'today':
             actual_target_date = pd.Timestamp.now().normalize()
             print(f"DEBUG get_latent: Using today's date: {actual_target_date.date()}")
        elif date_mode == 'specific_date':
             if target_date is None:
                  print("Error: target_date must be provided for 'specific_date' mode."); return None
             actual_target_date = pd.Timestamp(target_date).normalize()
             print(f"DEBUG get_latent: Using specific_date: {actual_target_date.date()}")
        else:
            print(f"Error: Invalid date_mode '{date_mode}'."); return None
        # --- End Determine Target Date ---

        # --- Get National Latent Popularity at Target Date ---
        national_pop_var = "latent_popularity_calendar_trajectory"
        if national_pop_var not in idata.posterior:
            print(f"Error: National trajectory variable '{national_pop_var}' not found."); return None
        if 'calendar_time' not in idata.posterior.coords:
             print("Error: Coordinate 'calendar_time' not found."); return None

        try:
            # Ensure coordinate is datetime
            calendar_coords = pd.to_datetime(idata.posterior['calendar_time'].values).normalize()
            min_cal_date = calendar_coords.min()
            max_cal_date = calendar_coords.max()

            if not (min_cal_date <= actual_target_date <= max_cal_date):
                 print(f"Warning: Target date {actual_target_date.date()} outside modeled range ({min_cal_date.date()} to {max_cal_date.date()}). Using nearest.")
                 # .sel with method='nearest' will handle this, but good to warn.

            national_pop_da = idata.posterior[national_pop_var].copy() # Work on a copy
            national_pop_da['calendar_time'] = calendar_coords # Assign converted coords

            national_pop_at_date = national_pop_da.sel(
                calendar_time=actual_target_date,
                method="nearest",
                tolerance=pd.Timedelta(days=1) # Allow slight tolerance
            )
            selected_date = pd.Timestamp(national_pop_at_date.calendar_time.item()).normalize()
            print(f"Selected national popularity for nearest date: {selected_date.date()}")

        except Exception as e:
            print(f"Error selecting national popularity at {actual_target_date.date()}: {e}"); return None
        # --- End Get National Latent Popularity ---


        # --- Adjust for District Effects (if requested) ---
        if district is not None:
            print(f"Adjusting popularity for district: {district}")
            if "district_effects" not in idata.posterior:
                 print(f"Error: 'district_effects' not found in posterior. Cannot adjust for district '{district}'."); return None
            if 'districts' not in idata.posterior.coords:
                 print("Error: 'districts' coordinate not found. Cannot map district name."); return None

            try:
                # Find the index for the requested district
                district_coord_list = idata.posterior['districts'].values.tolist()
                if district not in district_coord_list:
                    print(f"Error: District '{district}' not found in model coordinates: {district_coord_list}"); return None
                # No need for district_idx, .sel handles the name directly

                # Get the district effect posterior for the specific district
                district_effect_da = idata.posterior["district_effects"].sel(districts=district)

                # Need the *latent mean* at the target date to add the effect before softmax
                latent_mean_var = "latent_mu_calendar" # National latent mean over time
                if latent_mean_var not in idata.posterior:
                     # Try calculating it if missing (e.g., from older trace)
                     print(f"Warning: '{latent_mean_var}' not found. Attempting calculation...")
                     if "baseline_effect_calendar" in idata.posterior and "short_term_effect_calendar" in idata.posterior:
                          idata.posterior[latent_mean_var] = idata.posterior["baseline_effect_calendar"] + idata.posterior["short_term_effect_calendar"]
                     else:
                          print(f"Error: Cannot calculate '{latent_mean_var}'. Baseline or short-term effect missing."); return None


                latent_mean_da = idata.posterior[latent_mean_var].copy()
                latent_mean_da['calendar_time'] = calendar_coords # Use converted coords

                national_latent_mean_at_date = latent_mean_da.sel(
                    calendar_time=actual_target_date,
                    method="nearest", 
                    tolerance=pd.Timedelta(days=1)
                )
            
                # Add the district effect to the national latent mean
                # Dimensions should broadcast correctly: (chain, draw, party) + (chain, draw, party)
                district_latent_mean_at_date = national_latent_mean_at_date + district_effect_da

                # Apply softmax to get district-adjusted popularity
                # Need to ensure applying softmax along the correct axis ('parties_complete')
                adjusted_popularity = xr.apply_ufunc(
                    softmax,
                    district_latent_mean_at_date,
                    input_core_dims=[["parties_complete"]],
                    output_core_dims=[["parties_complete"]],
                    exclude_dims=set(("parties_complete",)),
                    output_dtypes=[district_latent_mean_at_date.dtype]
                ).rename("district_adjusted_popularity").assign_coords(district_latent_mean_at_date.coords)

                # Ensure coordinates are preserved and dimensions are in desired order (c, d, di, p)
                adjusted_popularity = adjusted_popularity.transpose("chain", "draw", "districts", "parties_complete")


                print(f"Successfully calculated popularity adjusted for district '{district}'.")
                return adjusted_popularity
            
            except KeyError as e:
                 print(f"Error accessing district data for '{district}': {e}"); return None
            except Exception as e:
                 print(f"Unexpected error adjusting for district '{district}': {e}"); return None

        else:
            # Return the national popularity if no district was specified
            print("Returning national latent popularity (no district specified).")
            return national_pop_at_date

    def get_district_vote_share_posterior(self, idata: az.InferenceData, date_mode: str = 'election_day', target_date: Optional[pd.Timestamp] = None) -> Optional[xr.DataArray]:
        """
        Calculates the posterior distribution of vote shares for ALL districts at a specific time point.

        This function performs post-processing using the posterior samples from the InferenceData object.

        Args:
            idata (az.InferenceData): InferenceData object with posterior samples. Must contain
                                      'national_trend_pt', 'base_offset_p', 'beta_p', and coordinates.
            date_mode (str): Defines the time point: 'election_day', 'last_poll', 'today', or 'specific_date'.
            target_date (pd.Timestamp, optional): Specific date. Used if date_mode is 'specific_date'.

        Returns:
            xr.DataArray or None: Posterior samples of district vote shares (probabilities).
                                  Dimensions: (chain, draw, districts, parties_complete).
                                  Returns None on error or if required data/coords are missing.
        """
        if idata is None or 'posterior' not in idata:
            print("Error: Valid InferenceData with posterior samples required."); return None

        # --- Determine Target Date based on mode ---
        actual_target_date = None
        # Use similar logic as get_latent_popularity to find the date
        if date_mode == 'election_day':
            if hasattr(self, 'election_date') and self.election_date:
                 actual_target_date = pd.Timestamp(self.election_date).normalize()
            else: print("Error: election_date not available for 'election_day' mode."); return None
        elif date_mode == 'last_poll':
             if hasattr(self.dataset, 'polls_train') and not self.dataset.polls_train.empty:
                  actual_target_date = pd.Timestamp(self.dataset.polls_train['date'].max()).normalize()
             else: print("Error: Cannot determine last poll date from dataset."); return None
        elif date_mode == 'today':
             actual_target_date = pd.Timestamp.now().normalize()
        elif date_mode == 'specific_date':
             if target_date is None: print("Error: target_date must be provided for 'specific_date' mode."); return None
             actual_target_date = pd.Timestamp(target_date).normalize()
        else:
            print(f"Error: Invalid date_mode '{date_mode}'."); return None
        print(f"DEBUG get_district_shares: Calculating for target date: {actual_target_date.date()} (mode: {date_mode})")
        # --- End Determine Target Date ---

        # --- Check for required variables and coordinates ---
        required_vars = ["national_trend_pt", "base_offset_p", "beta_p"]
        required_coords = ["calendar_time", "districts", "parties_complete"]

        missing_vars = [v for v in required_vars if v not in idata.posterior]
        if missing_vars: print(f"Error: Missing required posterior variables: {missing_vars}"); return None

        missing_coords = [c for c in required_coords if c not in idata.posterior.coords]
        if missing_coords: print(f"Error: Missing required coordinates: {missing_coords}"); return None

        # Also need national average trend (or calculate it)
        if "national_avg_trend_p" not in idata.posterior:
             print("Warning: 'national_avg_trend_p' not found. Calculating from 'national_trend_pt'.")
             try:
                  # Calculate and add to idata for potential future use (be cautious modifying idata)
                  # This calculation happens in memory here, doesn't modify the stored idata object persistently unless saved later
                  idata.posterior["national_avg_trend_p"] = idata.posterior["national_trend_pt"].mean(dim="calendar_time")
             except Exception as e:
                  print(f"Error calculating 'national_avg_trend_p': {e}"); return None
        # --- End Check ---

        try:
            # --- Prepare Coordinates and DataArrays ---
            calendar_coords = pd.to_datetime(idata.posterior['calendar_time'].values).normalize()
            district_coords = idata.posterior['districts'].values
            party_coords = idata.posterior['parties_complete'].values

            national_trend_da = idata.posterior["national_trend_pt"].copy()
            national_trend_da['calendar_time'] = calendar_coords # Assign converted coords

            national_avg_trend_da = idata.posterior["national_avg_trend_p"] # Should have dims (chain, draw, party)

            base_offset_da = idata.posterior["base_offset_p"] # Dims (chain, draw, party, district)
            beta_da = idata.posterior["beta_p"] # Dims (chain, draw, party, district)

            # --- Select National Trend at Target Date ---
            min_cal_date = calendar_coords.min()
            max_cal_date = calendar_coords.max()
            if not (min_cal_date <= actual_target_date <= max_cal_date):
                 print(f"Warning: Target date {actual_target_date.date()} outside modeled range ({min_cal_date.date()} to {max_cal_date.date()}). Using nearest.")

            # Select national trend at the target date
            # Result dims: (chain, draw, parties_complete)
            national_trend_at_date = national_trend_da.sel(
                calendar_time=actual_target_date,
                method="nearest",
                tolerance=pd.Timedelta(days=1)
            )
            selected_date = pd.Timestamp(national_trend_at_date.calendar_time.item()).normalize()
            print(f"Using national trend from nearest date: {selected_date.date()}")

            # --- Calculate District Latent Support ---
            # national_trend_at_date: (chain, draw, party)
            # national_avg_trend_da: (chain, draw, party)
            # base_offset_da:       (chain, draw, party, district)
            # beta_da:              (chain, draw, party, district)

            # Calculate deviation: (national_trend_at_date - national_avg_trend_da)
            # Need to add a 'district' dimension for broadcasting
            national_dev_at_date = national_trend_at_date - national_avg_trend_da
            # national_dev_at_date: (chain, draw, party)

            # Reshape/transpose for broadcasting:
            # base_offset_da already (chain, draw, party, district)
            # beta_da already (chain, draw, party, district)
            # Need national_trend_at_date as (chain, draw, party, district=1) ? No, easier to broadcast other way.
            # Need national_dev_at_date as (chain, draw, party, district=1)? Yes.

            # Let's align dimensions:
            # national_trend_at_date_b = national_trend_at_date.expand_dims(dim={"districts": district_coords}, axis=-1) # Add district dim
            # base_offset_da_t = base_offset_da # Already correct dims (c,d,p,di) -> (c,d,di,p)? Check dims again.
                # base_offset_p = pm.Deterministic("base_offset_p", ..., dims=("parties_complete", "districts"))
                # So base_offset_da is (chain, draw, parties_complete, districts)
            # beta_da also (chain, draw, parties_complete, districts)

            # Recalculate district adjustment (similar to model build but using posterior samples)
            # national_dev_at_date has dims (chain, draw, party)
            # base_offset_da has dims (chain, draw, party, district)
            # beta_da has dims (chain, draw, party, district)

            # We need to multiply (beta - 1) by national_dev_at_date.
            # Add district dim to national_dev_at_date for broadcasting
            national_dev_at_date_b = national_dev_at_date.expand_dims(dim={"districts": district_coords}, axis=-1)
            # Now national_dev_at_date_b has dims (chain, draw, party, district)

            # Calculate dynamic adjustment term: (beta - 1) * national_dev_at_date_b
            dynamic_adjustment = (beta_da - 1) * national_dev_at_date_b # Element-wise ok

            # Calculate total district adjustment: base_offset + dynamic_adjustment
            district_adjustment = base_offset_da + dynamic_adjustment # Element-wise ok
            # district_adjustment has dims (chain, draw, party, district)

            # Add national trend at date to district adjustment
            # Need to add district dim to national_trend_at_date for broadcasting
            national_trend_at_date_b = national_trend_at_date.expand_dims(dim={"districts": district_coords}, axis=-1)
            # national_trend_at_date_b has dims (chain, draw, party, district)

            latent_district_support = national_trend_at_date_b + district_adjustment
            # latent_district_support has dims (chain, draw, party, district)

            # --- Apply Softmax (MANUAL LOOP METHOD) ---
            # Apply softmax across the 'parties_complete' dimension manually
            # Get dimensions and values
            latent_vals = latent_district_support.values
            n_chains, n_draws, n_parties, n_districts = latent_vals.shape
            
            # Initialize output array
            share_vals = np.empty_like(latent_vals)
            
            print("Applying softmax manually (looping through samples/districts)...")
            loop_start = time.time() # Import time at top if not already present
            for c in range(n_chains):
                # Add progress print if needed
                # if (c + 1) % (n_chains // 4) == 0: print(f"  Softmax loop progress: Chain {c+1}/{n_chains}")
                for d in range(n_draws):
                    for k in range(n_districts):
                        # Apply softmax along the party dimension (axis=2)
                        share_vals[c, d, :, k] = softmax(latent_vals[c, d, :, k])
            loop_end = time.time()
            print(f"Manual softmax loop finished in {loop_end - loop_start:.2f} seconds.")

            # Reconstruct DataArray
            district_vote_shares = xr.DataArray(
                share_vals,
                coords=latent_district_support.coords, # Use original coords
                dims=latent_district_support.dims,     # Use original dims
                name="district_vote_shares"
            )
            
            # --- Old apply_ufunc method (commented out) ---
            # district_vote_shares = xr.apply_ufunc(
            #     softmax, # from scipy.special import softmax
            #     latent_district_support,
            #     input_core_dims=[["parties_complete"]],
            #     output_core_dims=[["parties_complete"]],
            #     exclude_dims=set(("parties_complete",)),
            #     # dask="parallelized", # <--- REMOVED THIS
            #     output_dtypes=[latent_district_support.dtype]
            # ).rename("district_vote_shares").assign_coords(latent_district_support.coords)

            # Ensure coordinates are preserved and dimensions are in desired order (c, d, di, p)
            # Check if transpose is still necessary with the manual method
            # The manual method should preserve the original (c, d, p, di) order, let's verify dims before transpose
            print(f"DEBUG: district_vote_shares dims before potential transpose: {district_vote_shares.dims}")
            # Transpose to ensure standard (c, d, di, p) order if needed
            target_dims = ("chain", "draw", "districts", "parties_complete")
            if district_vote_shares.dims != target_dims:
                 print(f"Transposing dimensions from {district_vote_shares.dims} to {target_dims}")
                 district_vote_shares = district_vote_shares.transpose(*target_dims)
            else:
                 print("Dimensions already in target order.")

            print(f"Successfully calculated district vote shares for date {selected_date.date()}.")
            return district_vote_shares

        except KeyError as e:
             print(f"Error: Key error during calculation (likely missing variable/coordinate): {e}"); return None
        except Exception as e:
             print(f"Unexpected error calculating district vote shares: {e}")
             import traceback
             traceback.print_exc()
             return None

# ... (rest of the file remains the same - placeholders, etc.) ...

        # ... (rest of the file remains the same - placeholders, etc.) ... 