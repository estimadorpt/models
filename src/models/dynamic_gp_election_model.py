import os
from typing import Dict, List, Tuple, Optional

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
        Calculates various goodness-of-fit metrics using posterior predictive samples.

        Compares model predictions against observed poll and district election result data.

        Args:
            idata: InferenceData object containing posterior and ideally
                   posterior_predictive and observed_data groups.

        Returns:
            A tuple containing:
            - A dictionary with calculated metrics (poll_mae, poll_rmse, ..., result_district_mae, ...)
            - The potentially updated idata object (if PPC was run).

        Raises:
            ValueError: If required data groups or variables are missing from idata.
        """
        # 1. Ensure Posterior Predictive samples exist
        if "posterior_predictive" not in idata:
            print("Posterior predictive samples not found in idata. Running posterior_predictive_check...")
            try:
                idata = self.posterior_predictive_check(idata, extend_idata=True) # Modifies idata in-place
            except Exception as e:
                 raise ValueError(f"Failed to generate posterior predictive samples: {e}")

        if "observed_data" not in idata:
            raise ValueError("observed_data group is required in InferenceData to calculate metrics.")
        if "posterior_predictive" not in idata: # Double check after PPC run
            raise ValueError("posterior_predictive group still missing after running check.")

        # --- START DEBUG ---
        print("\nDEBUG METRICS: Checking idata contents...")
        if hasattr(idata, 'posterior_predictive'):
            print(f"  idata.posterior_predictive keys: {list(idata.posterior_predictive.keys())}")
        else:
            print("  idata.posterior_predictive group not found.")
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
            # Use predicted probabilities (p_polls) if available
            if "p_polls" not in idata.posterior_predictive:
                 raise KeyError("'p_polls' missing from posterior_predictive. Cannot calculate probability-based metrics.")
            pp_poll_da = idata.posterior_predictive["p_polls"]
            pred_poll_probs = pp_poll_da.mean(dim=["chain", "draw"]).values

            # Use observed data name 'observed_polls' for counts
            if "observed_polls" not in idata.observed_data:
                 raise KeyError("'observed_polls' missing from observed_data.")
            obs_poll_counts = idata.observed_data["observed_polls"].values
            # Get total poll counts directly from model's data containers
            if "observed_N_polls" not in self.data_containers:
                 raise KeyError("'observed_N_polls' not found in self.data_containers")
            obs_poll_n = self.data_containers["observed_N_polls"].eval()
            with np.errstate(divide='ignore', invalid='ignore'):
                obs_poll_probs = obs_poll_counts / obs_poll_n[:, np.newaxis]
            obs_poll_probs = np.nan_to_num(obs_poll_probs)
            
            print("\n[Poll Data]")
            print(f"  Shape pred_poll_probs: {pred_poll_probs.shape}")
            print(f"  Shape obs_poll_probs: {obs_poll_probs.shape}")
            print(f"  Shape obs_poll_counts: {obs_poll_counts.shape}")

            if obs_poll_counts.shape[0] > 0:
                metrics["poll_mae"] = calculate_mae(pred_poll_probs, obs_poll_probs)
                metrics["poll_rmse"] = calculate_rmse(pred_poll_probs, obs_poll_probs)
                # Calculate log score using predicted probs and observed counts
                # pp_poll_counts_samples = idata.posterior_predictive["poll_likelihood"] # Predicted counts
                individual_poll_log_scores = calculate_log_score(pred_poll_probs, obs_poll_counts) # Use mean pred prob for simplicity here
                metrics["poll_log_score"] = np.mean(individual_poll_log_scores[np.isfinite(individual_poll_log_scores)])
                metrics["poll_rps"] = calculate_rps(pred_poll_probs, obs_poll_probs)
                # Need samples of probabilities for calibration, not just mean
                # Use pp_poll_da which has (chain, draw, obs, party) dimensions
                metrics["poll_calibration"] = calculate_calibration_data(pp_poll_da, obs_poll_probs)
            else:
                print("Warning: No poll observations found. Skipping poll metrics.")
                metrics["poll_mae"] = np.nan
                metrics["poll_rmse"] = np.nan
                metrics["poll_log_score"] = np.nan
                metrics["poll_rps"] = np.nan
                metrics["poll_calibration"] = {"bins": [], "observed_freq": [], "expected_freq": [], "bin_counts": []}

        except KeyError as e:
            print(f"Warning: Missing poll variable in InferenceData: {e}. Skipping poll metrics.")
            metrics["poll_mae"] = np.nan
            metrics["poll_rmse"] = np.nan
            metrics["poll_log_score"] = np.nan
            metrics["poll_rps"] = np.nan
            metrics["poll_calibration"] = {"bins": [], "observed_freq": [], "expected_freq": [], "bin_counts": []}
        except Exception as e:
            print(f"Error extracting poll data: {e}")
             # Handle error state for poll metrics - setting metrics to NaN
            metrics["poll_mae"] = np.nan
            metrics["poll_rmse"] = np.nan
            metrics["poll_log_score"] = np.nan
            metrics["poll_rps"] = np.nan
            metrics["poll_calibration"] = {"bins": [], "observed_freq": [], "expected_freq": [], "bin_counts": []}


        # --- District Results ---
        # Check if district probabilities and observed counts are available
        district_probs_available = "p_results_district" in idata.posterior_predictive
        district_counts_available = "observed_results_district" in idata.observed_data

        if district_probs_available and district_counts_available:
            try:
                pp_result_da = idata.posterior_predictive["p_results_district"]
                pred_result_probs = pp_result_da.mean(dim=["chain", "draw"]).values
                # Use observed data name 'observed_results_district' for counts
                obs_result_counts = idata.observed_data["observed_results_district"].values
                # Get total district counts directly from model's data containers
                if "observed_N_results_district" not in self.data_containers:
                     raise KeyError("'observed_N_results_district' not found in self.data_containers")
                obs_result_n = self.data_containers["observed_N_results_district"].eval()

                with np.errstate(divide='ignore', invalid='ignore'):
                    obs_result_probs = obs_result_counts / obs_result_n[:, np.newaxis]
                obs_result_probs = np.nan_to_num(obs_result_probs)

                print("\n[District Result Data]")
                print(f"  Shape pred_result_probs (district): {pred_result_probs.shape}")
                print(f"  Shape obs_result_probs (district): {obs_result_probs.shape}")
                print(f"  Shape obs_result_counts (district): {obs_result_counts.shape}")

                if obs_result_counts.shape[0] > 0:
                    metrics["result_district_mae"] = calculate_mae(pred_result_probs, obs_result_probs)
                    metrics["result_district_rmse"] = calculate_rmse(pred_result_probs, obs_result_probs)
                    individual_result_log_scores = calculate_log_score(pred_result_probs, obs_result_counts)
                    metrics["result_district_log_score"] = np.mean(individual_result_log_scores[np.isfinite(individual_result_log_scores)])
                    metrics["result_district_rps"] = calculate_rps(pred_result_probs, obs_result_probs)
                    # Use pp_result_da which has (chain, draw, obs, party) dimensions
                    metrics["result_district_calibration"] = calculate_calibration_data(pp_result_da, obs_result_probs)
                else:
                    print("Warning: No district result observations found. Skipping district result metrics.")
                    metrics["result_district_mae"] = np.nan
                    metrics["result_district_rmse"] = np.nan
                    metrics["result_district_log_score"] = np.nan
                    metrics["result_district_rps"] = np.nan
                    metrics["result_district_calibration"] = {"bins": [], "observed_freq": [], "expected_freq": [], "bin_counts": []}

            except KeyError as e:
                print(f"Warning: Missing district result variable in InferenceData: {e}. Skipping district result metrics.")
                metrics["result_district_mae"] = np.nan
                metrics["result_district_rmse"] = np.nan
                metrics["result_district_log_score"] = np.nan
                metrics["result_district_rps"] = np.nan
                metrics["result_district_calibration"] = {"bins": [], "observed_freq": [], "expected_freq": [], "bin_counts": []}
            except Exception as e:
                 print(f"Error extracting district result data: {e}")
                 # Handle error state for district result metrics
                 metrics["result_district_mae"] = np.nan
                 metrics["result_district_rmse"] = np.nan
                 metrics["result_district_log_score"] = np.nan
                 metrics["result_district_rps"] = np.nan
                 metrics["result_district_calibration"] = {"bins": [], "observed_freq": [], "expected_freq": [], "bin_counts": []}
        else:
            print("Warning: District result probabilities ('p_results_district' in posterior_predictive) or counts ('observed_results_district' in observed_data) not found. Skipping district result metrics.")
            metrics["result_district_mae"] = np.nan
            metrics["result_district_rmse"] = np.nan
            metrics["result_district_log_score"] = np.nan
            metrics["result_district_rps"] = np.nan
            metrics["result_district_calibration"] = {"bins": [], "observed_freq": [], "expected_freq": [], "bin_counts": []}


        print("--- End Debugging Metric Calculation Inputs ---")
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
                    dask="parallelized",
                    output_dtypes=[district_latent_mean_at_date.dtype]
                ).rename("district_adjusted_popularity")

                # Assign coordinates from the input DataArray to the output
                adjusted_popularity = adjusted_popularity.assign_coords(district_latent_mean_at_date.coords)


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

# ... (rest of the file remains the same - placeholders, etc.) ...

        # ... (rest of the file remains the same - placeholders, etc.) ... 