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
    A Bayesian election model where party support follows a Gaussian Process
    over calendar time.
    """

    def __init__(self, dataset: ElectionDataset, **kwargs):
        """
        Initializes the DynamicGPElectionModel.

        Args:
            dataset: An ElectionDataset object containing polls, results, etc.
            **kwargs: Additional keyword arguments for model configuration.
                      Expected keys: 'gp_lengthscale', 'gp_kernel', 'hsgp_m', 'hsgp_c'.
        """
        super().__init__(dataset, **kwargs)

        # Configuration for the Calendar Time GP
        self.gp_config = {
            "lengthscale": kwargs.get("gp_lengthscale", 180.0), # Default lengthscale in days
            "kernel": kwargs.get("gp_kernel", "Matern52"),
            "hsgp_m": kwargs.get("hsgp_m", [100]), # Basis functions for HSGP
            "hsgp_c": kwargs.get("hsgp_c", 2.0), # Expansion factor for HSGP
        }

        # Attributes to be set by _build_coords and _build_data_containers
        self.pollster_id = None
        self.calendar_time_poll_id = None
        self.calendar_time_result_id = None
        self.observed_election_indices = None # Indices relative to all_election_dates
        self.calendar_time_numeric = None # Numerical representation of calendar time coord
        self.gp_calendar_time = None # The HSGP object


    def _build_coords(self, polls: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray]:
        """
        Build the coordinates for the PyMC model, including a calendar time dimension.

        Args:
            polls: Optional DataFrame of polls to use instead of self.polls_train.

        Returns:
            Tuple containing:
            - pollster_id: Array mapping polls to pollster coord indices.
            - calendar_time_poll_id: Array mapping polls to calendar_time coord indices.
            - calendar_time_result_id: Array mapping historical results to calendar_time coord indices.
            - COORDS: Dictionary of coordinates for the PyMC model.
            - observed_election_indices: Indices of historical elections within the 'all_elections' list.
        """
        data_polls = polls if polls is not None else self.polls_train
        historical_results = self.results_oos # From base class

        # --- Create Unified Calendar Time Coordinate ---
        poll_dates = pd.to_datetime(data_polls['date']).unique()
        all_election_dates_dt = pd.to_datetime(self.all_election_dates).unique()
        historical_election_dates_dt = pd.to_datetime(self.historical_election_dates).unique()

        unique_dates = pd.to_datetime(
             np.union1d(poll_dates, all_election_dates_dt)
        ).unique().sort_values()

        min_date = unique_dates.min()
        calendar_time_numeric = (unique_dates - min_date).days.values
        self.calendar_time_numeric = calendar_time_numeric

        date_to_calendar_index = {date: i for i, date in enumerate(unique_dates)}

        # --- Standard Coordinates ---
        COORDS = {
            "observations": data_polls.index,
            "parties_complete": self.political_families,
            "calendar_time": unique_dates.strftime('%Y-%m-%d'), # Use strings for coords
            "elections_observed": [d.strftime('%Y-%m-%d') for d in pd.to_datetime(self.historical_election_dates)], # Use strings
        }

        pollster_id, COORDS["pollsters"] = data_polls["pollster"].factorize(sort=True)

        # --- Map Polls and Results to Calendar Time ---
        poll_dates_dt = pd.to_datetime(data_polls['date'])
        calendar_time_poll_id = poll_dates_dt.map(date_to_calendar_index).values
        if np.isnan(calendar_time_poll_id).any():
            missing_dates = data_polls.loc[np.isnan(calendar_time_poll_id), 'date'].unique()
            raise ValueError(f"Some poll dates could not be mapped to calendar_time index: {missing_dates}")
        calendar_time_poll_id = calendar_time_poll_id.astype(int)

        historical_results_dates_dt = pd.to_datetime(historical_results['election_date'])
        calendar_time_result_id = historical_results_dates_dt.map(date_to_calendar_index).values
        if np.isnan(calendar_time_result_id).any():
             missing_dates = historical_results.loc[np.isnan(calendar_time_result_id), 'election_date'].unique()
             raise ValueError(f"Some historical result dates could not be mapped to calendar_time index: {missing_dates}")
        calendar_time_result_id = calendar_time_result_id.astype(int)

        # Find indices of observed elections relative to *all* election dates
        # Compare based on standardized date strings
        all_election_dates_iso = [d.date().isoformat() for d in pd.to_datetime(self.all_election_dates)]
        observed_election_dates_iso = set(COORDS["elections_observed"]) # Already YYYY-MM-DD strings
        # print(f"DEBUG _build_coords: Comparing {all_election_dates_iso[0]} in {list(observed_election_dates_iso)[:5]}...") # Debug print removed
        self.observed_election_indices = [i for i, election_iso in enumerate(all_election_dates_iso)
                                          if election_iso in observed_election_dates_iso]
        # print(f"DEBUG: Value of self.observed_election_indices INSIDE _build_coords: {self.observed_election_indices}") # Debug print removed

        print("\n=== MODEL COORDINATES ===")
        for key, value in COORDS.items():
            print(f"{key}: length={len(value)}")
        print(f"Calendar time range: {COORDS['calendar_time'][0]} to {COORDS['calendar_time'][-1]}")
        print(f"Number of unique days: {len(calendar_time_numeric)}")
        print(f"Observed election indices relative to all_elections: {self.observed_election_indices}")

        return (pollster_id, calendar_time_poll_id, calendar_time_result_id,
                COORDS, self.observed_election_indices)


    def _build_data_containers(self,
                              polls: pd.DataFrame = None
                             ) -> Dict[str, pm.Data]:
        """
        Build the data containers for the PyMC model.

        Args:
            polls: Optional DataFrame of polls to use instead of self.polls_train.

        Returns:
            Dictionary of PyMC data containers.
        """
        current_polls = polls if polls is not None else self.polls_train

        if polls is None:
            non_competing_polls_additive_np = self.non_competing_polls_additive_base
            is_here_polls_np = self.is_here_polls_base
        else:
            is_here_polls = current_polls[self.political_families] > 0
            non_competing_polls_additive_np = np.where(is_here_polls, 0, -10).astype(np.float64)
            is_here_polls_np = is_here_polls.astype(int).to_numpy()

        # print(f"DEBUG: Type of self.observed_election_indices: {type(self.observed_election_indices)}") # Debug print removed
        # print(f"DEBUG: Value of self.observed_election_indices: {self.observed_election_indices}") # Debug print removed
        # print(f"DEBUG: Shape of self.non_competing_parties_results_base BEFORE filtering: {self.non_competing_parties_results_base.shape}") # Debug print removed
        non_competing_parties_results_filtered_np = self.non_competing_parties_results_base[self.observed_election_indices].astype(np.float64)
        print(f"Shape of filtered non_competing_parties_results (for observed elections): {non_competing_parties_results_filtered_np.shape}") # Keep this print

        results_N_historical = self.results_oos["sample_size"].to_numpy()
        observed_results_historical = self.results_oos[self.political_families].to_numpy()

        data_containers = dict(
            calendar_time_poll_idx=pm.Data("calendar_time_poll_idx", self.calendar_time_poll_id, dims="observations"),
            calendar_time_result_idx=pm.Data("calendar_time_result_idx", self.calendar_time_result_id, dims="elections_observed"),
            pollster_idx=pm.Data("pollster_idx", self.pollster_id, dims="observations"),
            observed_N_polls=pm.Data("observed_N_polls", current_polls["sample_size"].to_numpy(), dims="observations"),
            observed_polls=pm.Data(
                "observed_polls",
                current_polls[self.political_families].to_numpy(),
                dims=("observations", "parties_complete"),
            ),
            observed_N_results=pm.Data(
                "observed_N_results",
                results_N_historical,
                dims="elections_observed"
            ),
            observed_results=pm.Data(
                "observed_results",
                observed_results_historical,
                dims=("elections_observed", "parties_complete"),
            ),
            non_competing_parties_results=pm.Data(
                "non_competing_parties_results",
                non_competing_parties_results_filtered_np, # Use the filtered mask
                dims=("elections_observed", "parties_complete"),
            ),
            non_competing_polls_additive=pm.Data(
                "non_competing_polls_additive",
                non_competing_polls_additive_np,
                dims=("observations", "parties_complete"),
            ),
            non_competing_polls_multiplicative=pm.Data(
                "non_competing_polls_multiplicative",
                is_here_polls_np,
                dims=("observations", "parties_complete"),
            ),
        )

        return data_containers


    def build_model(self, polls: pd.DataFrame = None) -> pm.Model:
        """
        Build the PyMC model with a GP over calendar time.

        Args:
            polls: Optional DataFrame of polls to use for building coordinates
                   and data containers (defaults to self.polls_train).

        Returns:
            The compiled PyMC model.
        """
        (
            self.pollster_id,
            self.calendar_time_poll_id,
            self.calendar_time_result_id,
            self.coords,
            self.observed_election_indices,
        ) = self._build_coords(polls)

        gp_lengthscale = float(self.gp_config["lengthscale"])
        print(f"Using GP lengthscale: {gp_lengthscale} days")

        with pm.Model(coords=self.coords) as model:
            data_containers = self._build_data_containers(polls)

            # --------------------------------------------------------
            #        DYNAMIC BASELINE: GP over Calendar Time
            # --------------------------------------------------------
            if self.gp_config["kernel"] == "Matern52":
                 cov_func_calendar = pm.gp.cov.Matern52(input_dim=1, ls=gp_lengthscale)
            elif self.gp_config["kernel"] == "ExpQuad":
                 cov_func_calendar = pm.gp.cov.ExpQuad(input_dim=1, ls=gp_lengthscale)
            else:
                 raise ValueError(f"Unsupported GP kernel: {self.gp_config['kernel']}")

            self.gp_calendar_time = pm.gp.HSGP(
                cov_func=cov_func_calendar,
                m=self.gp_config["hsgp_m"],
                c=self.gp_config["hsgp_c"]
            )

            phi_calendar, sqrt_psd_calendar = self.gp_calendar_time.prior_linearized(
                 X=self.calendar_time_numeric[:, None]
            )

            coord_name_gp_basis = f"gp_basis_calendar"
            if coord_name_gp_basis not in model.coords:
                model.add_coords({coord_name_gp_basis: np.arange(self.gp_calendar_time.n_basis_vectors)})

            gp_coef_calendar_raw = pm.Normal(
                "gp_coef_calendar_raw",
                mu=0,
                sigma=1,
                dims=(coord_name_gp_basis, "parties_complete")
            )

            gp_coef_calendar = pm.Deterministic(
                "gp_coef_calendar",
                gp_coef_calendar_raw - gp_coef_calendar_raw.mean(axis=1, keepdims=True),
                 dims=(coord_name_gp_basis, "parties_complete")
            )

            party_calendar_effect = pm.Deterministic(
                "party_calendar_effect",
                pt.einsum('cb,bp->cp',
                          phi_calendar,
                          gp_coef_calendar * sqrt_psd_calendar[:, None]
                         ),
                dims=("calendar_time", "parties_complete")
            )

            # --------------------------------------------------------
            #          HOUSE EFFECTS (Pollster Bias) - Temporarily commented out
            # --------------------------------------------------------
            # Standard deviation of house effects
            # house_effects_sd = pm.HalfNormal("house_effects_sd", sigma=0.1) # Tighter prior
            house_effects_sd = pm.HalfNormal("house_effects_sd", sigma=0.1) # Tighter prior

            # House effects, constrained to sum to zero across parties for each pollster
            # house_effects = pm.ZeroSumNormal(
            #     "house_effects",
            #     sigma=house_effects_sd,
            #     dims=("pollsters", "parties_complete"),
            #     shape=(len(self.coords["pollsters"]), len(self.political_families))
            # )
            house_effects = pm.ZeroSumNormal(
                "house_effects",
                sigma=house_effects_sd,
                dims=("pollsters", "parties_complete"),
                shape=(len(self.coords["pollsters"]), len(self.political_families))
            )

            # --------------------------------------------------------
            #                      POLL LIKELIHOOD
            # --------------------------------------------------------
            latent_mu_polls = pm.Deterministic(
                "latent_mu_polls",
                (
                    party_calendar_effect[data_containers["calendar_time_poll_idx"]]
                    + data_containers['non_competing_polls_additive']
                ),
                dims=("observations", "parties_complete")
            )

            noisy_mu_polls = pm.Deterministic(
                "noisy_mu_polls",
                (
                    latent_mu_polls
                    # + house_effects[data_containers["pollster_idx"]] # Temporarily remove house effect
                    + house_effects[data_containers["pollster_idx"]]
                ) * data_containers['non_competing_polls_multiplicative'],
                dims=("observations", "parties_complete")
            )

            noisy_popularity_polls = pm.Deterministic(
                "noisy_popularity_polls",
                pt.special.softmax(noisy_mu_polls, axis=1),
                dims=("observations", "parties_complete"),
            )

            concentration_polls = pm.Gamma("concentration_polls", alpha=100, beta=0.1)

            N_approve = pm.DirichletMultinomial(
                "N_approve",
                a=(concentration_polls * noisy_popularity_polls),
                n=data_containers["observed_N_polls"],
                observed=data_containers["observed_polls"],
                dims=("observations", "parties_complete"),
            )

            # --------------------------------------------------------
            #                 ELECTION RESULT LIKELIHOOD
            # --------------------------------------------------------
            latent_mu_results = pm.Deterministic(
                "latent_mu_results",
                 party_calendar_effect[data_containers["calendar_time_result_idx"]]
                 + data_containers['non_competing_parties_results'],
                dims=("elections_observed", "parties_complete")
            )

            latent_pop_results = pm.Deterministic(
                "latent_pop_results",
                pt.special.softmax(latent_mu_results, axis=1),
                dims=("elections_observed", "parties_complete"),
            )

            concentration_results = pm.Gamma("concentration_results", alpha=100, beta=0.05)

            R = pm.DirichletMultinomial(
                "R",
                n=data_containers["observed_N_results"],
                a=(concentration_results * latent_pop_results),
                observed=data_containers["observed_results"],
                dims=("elections_observed", "parties_complete")
            )

            # --------------------------------------------------------
            #       (Optional) Deterministics for easier analysis
            # --------------------------------------------------------
            latent_popularity_calendar_trajectory = pm.Deterministic(
                "latent_popularity_calendar_trajectory",
                pt.special.softmax(party_calendar_effect, axis=1),
                 dims=("calendar_time", "parties_complete")
             )

        self.model = model
        print("\nModel building complete.")
        return model

    def posterior_predictive_check(self, posterior):
        """Perform posterior predictive checks (TODO: Implement)."""
        print("Warning: posterior_predictive_check not yet implemented for DynamicGPElectionModel.")
        return {}

    def predict_latent_trajectory(self,
                                idata: az.InferenceData,
                                start_date: pd.Timestamp,
                                end_date: pd.Timestamp) -> az.InferenceData:
        """Predicts the latent popularity trajectory (TODO: Implement)."""
        print("Warning: predict_latent_trajectory not yet implemented for DynamicGPElectionModel.")
        return az.InferenceData()

    def predict(self, oos_data: pd.DataFrame) -> az.InferenceData:
        """Generate predictions for out-of-sample poll data (TODO: Implement)."""
        print("Warning: predict (for polls) not yet implemented for DynamicGPElectionModel.")
        return az.InferenceData()

    def predict_history(self, elections_to_predict: List[str]) -> az.InferenceData:
        """Calculate predictive accuracy for historical elections (TODO: Implement)."""
        print("Warning: predict_history not yet implemented for DynamicGPElectionModel.")
        return az.InferenceData() 