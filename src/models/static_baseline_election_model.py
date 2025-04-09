import os
from typing import Dict, List, Tuple

import arviz as az
import arviz
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from scipy.special import softmax

from src.data.dataset import ElectionDataset
from src.models.base_model import BaseElectionModel


class StaticBaselineElectionModel(BaseElectionModel):
    """A Bayesian election model with a static baseline component."""

    def __init__(self, dataset: ElectionDataset, **kwargs):
        # Call the base class initializer, which now handles common data loading
        super().__init__(dataset, **kwargs)

        # Specific configuration for this model type
        self.gp_config = {
            "baseline_lengthscale": kwargs.get("baseline_lengthscale", dataset.baseline_timescales),
            "election_lengthscale": kwargs.get("election_lengthscale", dataset.election_timescales),
            "kernel": kwargs.get("kernel", "matern52"),
            "zerosum": kwargs.get("zerosum", True),
            "variance_limit": kwargs.get("variance_limit", 0.8),
        }
        
        # Initialize specific model attributes (already in base, but good to be explicit?)
        # No, these are set by _build_coords which is called later
        self.pollster_id = None
        self.countdown_id = None
        self.election_id = None
        self.observed_election_indices = None
        self.gp_baseline = None # Specific to this GP model

    def _build_coords(self, polls: pd.DataFrame = None):
        """Build the coordinates for the PyMC model, using base class data."""
        # Use provided polls or default to the base class's polls_train
        data = polls if polls is not None else self.polls_train

        COORDS = {
            "observations": data.index,
            # Use political_families from base class
            "parties_complete": self.political_families,
        }
        pollster_id, COORDS["pollsters"] = data["pollster"].factorize(sort=True)

        # Handle countdown values
        countdown_values = data["countdown"].values
        if not np.isfinite(countdown_values.max()):
            max_countdown = len(countdown_values) - 1
        else:
            max_countdown = int(countdown_values.max())

        countdown_id = countdown_values.astype(int)

        # For nowcasting, use the original countdown dimension from stored coords
        # Check if self.coords was populated by a previous fit
        if polls is not None and hasattr(self, 'coords') and self.coords and 'countdown' in self.coords:
            COORDS["countdown"] = self.coords["countdown"]
        else:
            COORDS["countdown"] = np.arange(max_countdown + 1)

        # Use ALL election dates from base class
        COORDS["elections"] = self.all_election_dates # Use all dates from base
        print(f"Setting 'elections' coord to ALL election cycles: {COORDS['elections']}")

        # Get election indices for the data, mapping to the index in all_election_dates
        election_dates_in_data = data["election_date"].unique()
        election_id = np.zeros(len(data), dtype=int)
        for i, date in enumerate(data["election_date"]):
            # Find the index of this date in the *all_election_dates* from base class
            for j, cycle_date_str in enumerate(self.all_election_dates): # Use all_election_dates
                 if pd.to_datetime(cycle_date_str) == pd.to_datetime(date):
                     election_id[i] = j
                     break
                 elif j == len(self.all_election_dates) - 1:
                      print(f"Warning: Poll date {date} associated election date not found in all_election_dates: {self.all_election_dates}. Assigning index 0.")
                      election_id[i] = 0

        # Coordinate for elections where we HAVE observed results (historical only) from base class
        COORDS["elections_observed"] = self.historical_election_dates
        print(f"Setting 'elections_observed' coord to historical elections only: {COORDS['elections_observed']}")

        # Get indices for observed elections within the main 'elections' coordinate
        observed_election_indices = [i for i, election in enumerate(COORDS["elections"]) if election in COORDS["elections_observed"]]
        print(f"Indices for observed elections: {observed_election_indices}")

        return pollster_id, countdown_id, election_id, COORDS, observed_election_indices

    def _build_data_containers(self, polls: pd.DataFrame = None):
        """Build the data containers for the PyMC model, using base class data."""

        # Use provided polls or default to the base class's polls_train
        current_polls = polls if polls is not None else self.polls_train

        # Use base class masks if using default polls, otherwise recalculate
        if polls is None:
            non_competing_polls_additive_np = self.non_competing_polls_additive_base
            is_here_polls_np = self.is_here_polls_base
        else:
            # Recalculate masks for the provided polls DataFrame
            is_here_polls = current_polls[self.political_families] > 0
            non_competing_polls_additive_np = np.where(is_here_polls, 0, -10).astype(np.int32)
            is_here_polls_np = is_here_polls.astype(int).to_numpy()

        # Use the pre-calculated results mask from the base class
        non_competing_parties_results_np = self.non_competing_parties_results_base

        # Use predictors from base class
        # Note: campaign_preds might need re-indexing if 'polls' is provided and differs from polls_train
        # For simplicity now, assume campaign_preds aligns with whatever 'current_polls' is.
        # A more robust solution might involve passing predictors alongside 'polls'.
        stdz_gdp_campaign = self.campaign_preds["gdp"].to_numpy()
        if polls is not None:
            # Attempt to align predictors if custom polls are given
            try:
                 stdz_gdp_campaign = self.campaign_preds.reindex(current_polls.index)["gdp"].fillna(0).to_numpy()
                 print("Aligned campaign GDP predictors to provided polls.")
            except Exception as e:
                 print(f"Warning: Could not align campaign GDP predictors to provided polls. Using original alignment. Error: {e}")
                 # Fallback to original alignment - this might be wrong if polls != polls_train
                 stdz_gdp_campaign = self.campaign_preds["gdp"].to_numpy()[:len(current_polls)] # Risky slicing


        # Use results predictors and results_oos (historical results) from base class
        election_gdp_results = self.results_preds["gdp"].to_numpy()
        results_N_historical = self.results_oos["sample_size"].to_numpy()
        observed_results_historical = self.results_oos[self.political_families].to_numpy()
        government_status_np = self.government_status.values.astype(int) # From base

        # Data containers for inference
        data_containers = dict(
            election_idx=pm.Data("election_idx", self.election_id, dims="observations"),
            pollster_idx=pm.Data("pollster_idx", self.pollster_id, dims="observations"),
            countdown_idx=pm.Data("countdown_idx", self.countdown_id, dims="observations"),
            # Predictors from base class (potentially re-aligned)
            stdz_gdp=pm.Data("stdz_gdp", stdz_gdp_campaign, dims="observations"),
            election_gdp=pm.Data("election_gdp", election_gdp_results, dims="elections"),
            # Observed poll data (from current_polls)
            observed_N=pm.Data("observed_N", current_polls["sample_size"].to_numpy(), dims="observations"),
            observed_polls=pm.Data(
                "observed_polls",
                current_polls[self.political_families].to_numpy(),
                dims=("observations", "parties_complete"),
            ),
            # Historical results data from base class
            results_N=pm.Data(
                "results_N",
                results_N_historical,
                dims="elections_observed"
            ),
            observed_results=pm.Data(
                "observed_results",
                observed_results_historical,
                dims=("elections_observed", "parties_complete"),
            ),
            # Masks (potentially re-calculated for polls, base used for results)
            non_competing_parties_results=pm.Data(
                "non_competing_parties_results",
                non_competing_parties_results_np, # Use base mask
                dims=("elections", "parties_complete"),
            ),
            non_competing_polls_additive=pm.Data(
                "non_competing_polls_additive",
                non_competing_polls_additive_np, # Use potentially recalculated mask
                dims=("observations", "parties_complete"),
            ),
            non_competing_polls_multiplicative=pm.Data(
                "non_competing_polls_multiplicative",
                is_here_polls_np, # Use potentially recalculated mask
                dims=("observations", "parties_complete"),
            ),
            # Government status from base class
            government_status=pm.Data(
                "government_status",
                government_status_np,
                dims=("elections", "parties_complete"),
            )
        )

        return data_containers

    def build_model(self, polls: pd.DataFrame = None) -> pm.Model:
        """Build the PyMC model, using base class data where possible."""
        # Build coordinates and store instance variables (pollster_id, etc.)
        (
            self.pollster_id,
            self.countdown_id,
            self.election_id,
            self.coords, # Sets self.coords
            self.observed_election_indices,
        ) = self._build_coords(polls)

        # Debug info about dimensions
        print("\n=== MODEL DIMENSIONS ===")
        for key, value in self.coords.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"{key}: shape={len(value)}")

        # Debug info about results_oos (now from base class)
        print(f"results_oos shape: {self.results_oos.shape}")
        print(f"results_oos election dates: {self.results_oos['election_date'].unique()}")

        # Check if results_oos and elections_observed match in length
        num_oos = len(self.results_oos) # Use base attribute
        num_observed = len(self.coords["elections_observed"])
        if num_oos != num_observed:
            print(f"WARNING: Mismatch between results_oos ({num_oos}) and elections_observed ({num_observed})")

        # Ensure GP timescales are lists (using self.gp_config)
        if not isinstance(self.gp_config["baseline_lengthscale"], list):
            self.gp_config["baseline_lengthscale"] = [self.gp_config["baseline_lengthscale"]]
        if not isinstance(self.gp_config["election_lengthscale"], list):
            self.gp_config["election_lengthscale"] = [self.gp_config["election_lengthscale"]]

        with pm.Model(coords=self.coords) as model:
            # Build data containers using potentially provided polls
            data_containers = self._build_data_containers(polls)

            # --------------------------------------------------------
            #                   BASELINE COMPONENTS
            # --------------------------------------------------------
            # Use political_families from base class
            party_baseline = pm.ZeroSumNormal(
                "party_baseline", sigma=0.5, dims="parties_complete"
            )

            # Election-specific party baseline.
            election_party_baseline = pm.ZeroSumNormal(
                "election_party_baseline",
                sigma=0.15,
                dims=("elections", "parties_complete")
            )

            # --------------------------------------------------------
            #               TIME-VARYING COMPONENTS (Election Specific GP)
            # --------------------------------------------------------

            baseline_lengthscale = self.gp_config["baseline_lengthscale"][0]
            print(f"Using single baseline timescale: {baseline_lengthscale}")

            cov_func_baseline = pm.gp.cov.Matern52(input_dim=1, ls=baseline_lengthscale)
            # Note: self.gp_baseline is stored later
            gp_baseline_local = pm.gp.HSGP(cov_func=cov_func_baseline, m=[20], c=2.0)
            # Use countdown coordinate from self.coords
            phi_baseline, sqrt_psd_baseline = gp_baseline_local.prior_linearized(X=self.coords["countdown"][:, None])

            coord_name = f"gp_basis_baseline"
            if coord_name not in model.coords:
                model.add_coords({coord_name: np.arange(gp_baseline_local.n_basis_vectors)})

            gp_coef_baseline = pm.Normal(
                f"gp_coef_baseline",
                mu=0,
                sigma=1,
                dims=("elections", coord_name, "parties_complete")
            )

            party_time_effect = pm.Deterministic(
                f"party_time_effect",
                pt.einsum('cb,ebp->ecp',
                          phi_baseline,
                          gp_coef_baseline * sqrt_psd_baseline[:, None]
                         ),
                dims=("elections", "countdown", "parties_complete")
            )

            party_time_effect_weighted = party_time_effect

            # --------------------------------------------------------
            #          HOUSE EFFECTS & POLL BIAS
            # --------------------------------------------------------
            house_effects_sd = pm.HalfNormal("house_effects_sd", sigma=0.15)
            house_effects = pm.ZeroSumNormal(
                "house_effects",
                sigma=house_effects_sd,
                dims=("pollsters", "parties_complete"),
                # Use political_families from base class
                shape=(len(self.coords["pollsters"]), len(self.political_families))
            )

            # --------------------------------------------------------
            #                      POLL RESULTS
            # --------------------------------------------------------

            # Compute latent_mu (Using election-specific GP)
            latent_mu = pm.Deterministic(
                "latent_mu",
                (
                    party_baseline[None, :]
                    + election_party_baseline[data_containers["election_idx"]]
                    + party_time_effect_weighted[data_containers["election_idx"], data_containers["countdown_idx"]]
                    + data_containers['non_competing_polls_additive'] # From data container
                ),
                dims=("observations", "parties_complete")
            )

            latent_popularity = pm.Deterministic(
                "latent_popularity",
                pt.special.softmax(latent_mu, axis=1),
                dims=("observations", "parties_complete"),
            )

            noisy_mu = pm.Deterministic(
                "noisy_mu",
                (
                    latent_mu
                    + house_effects[data_containers["pollster_idx"]]
                ) * data_containers['non_competing_polls_multiplicative'], # From data container
                dims=("observations", "parties_complete")
            )

            noisy_popularity = pm.Deterministic(
                "noisy_popularity",
                pt.special.softmax(noisy_mu, axis=1),
                dims=("observations", "parties_complete"),
            )

            concentration_polls = pm.Gamma("concentration_polls", alpha=100, beta=0.1)

            N_approve = pm.DirichletMultinomial(
                "N_approve",
                a=concentration_polls * noisy_popularity,
                n=data_containers["observed_N"], # From data container
                observed=data_containers["observed_polls"], # From data container
                dims=("observations", "parties_complete"),
            )

            # --------------------------------------------------------
            #                    ELECTION RESULTS
            # --------------------------------------------------------

            latent_mu_t0 = pm.Deterministic(
                "latent_mu_t0",
                (
                    party_baseline[None, :]
                    + election_party_baseline
                    + party_time_effect_weighted[:, 0]
                    + data_containers['non_competing_parties_results'] # From data container
                ),
                dims=("elections", "parties_complete")
            )

            latent_popularity_full_trajectory_mu = pm.Deterministic(
                "latent_popularity_full_trajectory_mu",
                (
                     party_baseline[None, None, :]
                     + election_party_baseline[:, None, :]
                     + party_time_effect_weighted
                ),
                dims=("elections", "countdown", "parties_complete")
            )

            latent_popularity_trajectory = pm.Deterministic(
                "latent_popularity_trajectory",
                pt.special.softmax(latent_popularity_full_trajectory_mu, axis=2),
                dims=("elections", "countdown", "parties_complete")
            )

            latent_pop_t0 = pm.Deterministic(
                "latent_pop_t0",
                pt.special.softmax(latent_mu_t0, axis=1),
                dims=("elections", "parties_complete"),
            )

            concentration_results = pm.Gamma("concentration_results", alpha=100, beta=0.05)

            # Select ONLY the latent popularity values corresponding to the observed elections
            # Use observed_election_indices stored from _build_coords
            latent_pop_t0_observed = latent_pop_t0[self.observed_election_indices]

            R = pm.DirichletMultinomial(
                "R",
                n=data_containers["results_N"], # From data container
                a=concentration_results * latent_pop_t0_observed,
                observed=data_containers["observed_results"], # From data container
                dims=("elections_observed", "parties_complete")
            )

            # Store gp_baseline instance for later use in prediction
            self.gp_baseline = gp_baseline_local # Store the local instance

        self.model = model # Store the built model instance
        return model

    def posterior_predictive_check(self, posterior):
        """
        Perform posterior predictive checks.

        Parameters:
        -----------
        posterior : arviz.InferenceData
            The posterior samples containing posterior predictive data.

        Returns:
        --------
        ppc_results : dict
            A dictionary containing various posterior predictive check results.
        """
        # Build the model using the training data stored in the base class
        if self.model is None:
             print("Building model for PPC check using self.polls_train")
             _ = self.build_model(self.polls_train)
        # Commenting out the elif block referencing undefined 'polls' variable
        # elif polls is not None: # If specific polls were used for the trace, rebuild? Risky.
        #      print("Warning: PPC check assumes model was built with self.polls_train")
        #      # Ideally, the model used for sampling 'posterior' should be used.
        #      # This function might need refactoring if model can be built with different data.
        #      pass # Use existing self.model

        ppc = posterior.posterior_predictive
        ppc_results = {}

        print("Available keys in posterior_predictive:", ppc.data_vars.keys())

        # Use base class attributes for observed data comparison
        for i, party in enumerate(self.political_families):
            observed_polls_proportions = self.polls_train[party].values / self.polls_train['sample_size'].values
            # Use results_oos for historical results comparison
            observed_results_proportions = self.results_oos[party].values / self.results_oos['sample_size'].values

            # Check if 'N_approve' exists in ppc
            if 'N_approve' not in ppc:
                 print(f"WARNING: 'N_approve' not found in posterior predictive. Skipping PPC for {party}.")
                 continue # Skip party if required variable is missing

            # Assume N_approve corresponds to self.polls_train structure
            predicted_counts = ppc['N_approve'].values[:, :, :, i]
            predicted_proportions = predicted_counts / self.polls_train['sample_size'].values # Divide by original N

            # Calculate mean absolute error for polls
            mae_polls = np.mean(np.abs(observed_polls_proportions - predicted_proportions.mean(axis=(0, 1))))
            ppc_results[f'{party}_mae_polls'] = mae_polls

            # Calculate coverage of 95% credible interval for polls
            lower, upper = np.percentile(predicted_proportions, [2.5, 97.5], axis=(0, 1))
            coverage_polls = np.mean((observed_polls_proportions >= lower) & (observed_polls_proportions <= upper))
            ppc_results[f'{party}_coverage_polls'] = coverage_polls

            # --- MAE/Plotting for Results ---
            # This part is tricky. N_approve is based on polls_train.
            # How to compare N_approve prediction with observed_results_proportions?
            # Option 1: Compare latent_pop_t0 (if sampled) with observed_results.
            # Option 2: Compare predicted 'R' (if sampled) with observed_results.
            # Option 3: Approximate - Use the last part of N_approve? (Incorrect assumptions)

            # Let's assume 'R' (the results variable) is in posterior_predictive for results comparison
            if 'R' in ppc:
                predicted_results_counts = ppc['R'].values[:, :, :, i] # Shape (chain, draw, elections_observed)
                # Divide by the N corresponding to observed results (results_N from base class)
                predicted_results_proportions = predicted_results_counts / self.results_oos['sample_size'].values

                # Calculate mean absolute error for results
                mae_results = np.mean(np.abs(observed_results_proportions - predicted_results_proportions.mean(axis=(0, 1))))
                ppc_results[f'{party}_mae_results'] = mae_results
                
                plot_results_pred = predicted_results_proportions.mean(axis=(0,1))
            else:
                print(f"Warning: 'R' not found in posterior predictive. Cannot calculate MAE/plot for results for {party}.")
                mae_results = np.nan
                ppc_results[f'{party}_mae_results'] = mae_results
                plot_results_pred = None # No results prediction available to plot

            # Plot observed vs. predicted
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.scatter(observed_polls_proportions, predicted_proportions.mean(axis=(0, 1)), label='Polls (Predicted N_approve)', alpha=0.5)
            if plot_results_pred is not None:
                # Only plot if we have results predictions
                plt.scatter(observed_results_proportions, plot_results_pred, label='Results (Predicted R)', marker='x', s=100, color='orange')
            else:
                 # Optionally plot observed results even if no prediction is available
                 plt.scatter(observed_results_proportions, np.full_like(observed_results_proportions, np.nan), label='Results (Observed Only)', marker='s', s=50, color='gray')

            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('Observed Proportion')
            plt.ylabel('Predicted Proportion (Mean)')
            plt.title(f'Observed vs. Predicted Proportion for {party}')
            plt.legend()
            plt.savefig(f'ppc_plot_{party}.png')
            plt.close()

        return ppc_results

    def predict_latent_trajectory(self,
                                idata: az.InferenceData,
                                start_date: pd.Timestamp,
                                end_date: pd.Timestamp) -> az.InferenceData:
        """Predicts the latent popularity trajectory over a specified date range.
           Relies on base class attributes like political_families, coords (if available).
        """
        print(f"\n--- Predicting Latent Trajectory from {start_date.date()} to {end_date.date()} ---")

        # Check for necessary base attributes and model components
        if not hasattr(self, 'gp_baseline') or self.gp_baseline is None:
            print("Warning: gp_baseline not found. Attempting to rebuild model with self.polls_train...")
            try:
                self.build_model(self.polls_train) # Rebuild to set gp_baseline and coords
                if not hasattr(self, 'gp_baseline') or self.gp_baseline is None:
                     raise RuntimeError("Failed to set gp_baseline even after rebuild.")
                if not hasattr(self, 'coords') or not self.coords:
                     raise RuntimeError("Failed to set coords even after rebuild.")
            except Exception as e:
                raise RuntimeError(f"Required attributes (`gp_baseline`, `coords`) not found and couldn't be rebuilt: {e}")

        original_model_coords = self.coords
        gp_basis_coord_name = "gp_basis_baseline" # Name used in build_model

        # Check required coordinates exist
        if 'elections' not in original_model_coords or len(original_model_coords['elections']) == 0:
             raise RuntimeError("Model 'elections' coordinate not found or empty.")
        if 'countdown' not in original_model_coords or len(original_model_coords['countdown']) == 0:
            raise ValueError("Original countdown coordinate is missing or empty.")
        if gp_basis_coord_name not in original_model_coords:
             raise ValueError(f"Original GP basis coordinate ('{gp_basis_coord_name}') is missing.")

        # --- 1. Define Prediction Coordinates ---
        reference_end_date_str = original_model_coords['elections'][-1]
        try: reference_end_date = pd.to_datetime(reference_end_date_str)
        except Exception as e: raise ValueError(f"Could not parse ref date '{reference_end_date_str}': {e}")

        pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
        if pred_days.empty:
             print("Warning: No days found in the specified prediction range.")
             return az.InferenceData()

        pred_countdown_values = (reference_end_date - pred_days).days.values
        min_coord_countdown = original_model_coords['countdown'].min()
        max_coord_countdown = original_model_coords['countdown'].max()
        pred_countdown_values_clipped = np.clip(pred_countdown_values, min_coord_countdown, max_coord_countdown)

        # Find unique, valid countdown values needed for prediction
        pred_countdown_unique_vals = np.unique(pred_countdown_values_clipped)
        valid_pred_countdown_coord_vals = pred_countdown_unique_vals[
            np.isin(pred_countdown_unique_vals, original_model_coords['countdown'])
        ]

        if len(valid_pred_countdown_coord_vals) == 0:
             print(f"Warning: No valid countdown indices found for prediction range.")
             # ... (rest of warning message) ...
             return az.InferenceData()

        # Filter prediction days and map countdown values to the new minimal coordinate
        valid_days_mask = np.isin(pred_countdown_values_clipped, valid_pred_countdown_coord_vals)
        pred_days_final = pred_days[valid_days_mask]
        pred_countdown_values_final = pred_countdown_values_clipped[valid_days_mask]

        if pred_days_final.empty:
             print("Warning: No observation days remaining after filtering.")
             return az.InferenceData()

        pred_model_countdown_coord = np.sort(valid_pred_countdown_coord_vals)
        countdown_map_to_pred_coord = {val: i for i, val in enumerate(pred_model_countdown_coord)}
        countdown_indices_for_obs = pd.Series(pred_countdown_values_final).map(countdown_map_to_pred_coord).values.astype(int)

        # Prediction coordinates using base attributes and filtered coords
        prediction_coords = {
            "observations": pred_days_final.values,
            "parties_complete": self.political_families, # Use base attribute
            "elections": original_model_coords["elections"],
            "countdown": pred_model_countdown_coord, # Minimal countdown set
            gp_basis_coord_name: original_model_coords[gp_basis_coord_name],
        }
        print(f"Prediction coordinates defined for {len(pred_days_final)} days.")
        print(f"Using {len(pred_model_countdown_coord)} unique countdown indices.")

        # --- 2. Build Prediction Model Context ---
        with pm.Model(coords=prediction_coords) as pred_model:
            # Flat variables for posterior injection
            party_baseline = pm.Flat("party_baseline", dims="parties_complete")
            _election_party_baseline = pm.Flat("election_party_baseline", dims=("elections", "parties_complete"))
            _gp_coef_baseline = pm.Flat("_gp_coef_baseline", dims=("elections", gp_basis_coord_name, "parties_complete"))

            # Data containers for indexing within the prediction model
            countdown_idx_data = pm.Data("countdown_idx_data", countdown_indices_for_obs, dims="observations")
            last_election_idx = len(original_model_coords["elections"]) - 1
            election_indices_pred = np.full(len(pred_days_final), last_election_idx, dtype=int)
            election_idx_data = pm.Data("election_idx_data", election_indices_pred, dims="observations")

            # Reconstruct Deterministic variables using the prediction coordinates
            # Use the stored gp_baseline object
            phi_baseline_pred, sqrt_psd_baseline_pred = self.gp_baseline.prior_linearized(
                X=prediction_coords["countdown"][:, None] # Use prediction countdown coord
            )

            # Ensure sqrt_psd shape matches gp_basis coord length in prediction_coords
            if len(sqrt_psd_baseline_pred) != len(prediction_coords[gp_basis_coord_name]):
                  raise ValueError("Mismatch between sqrt_psd length and GP basis coordinate length during latent prediction.")

            party_time_effect_pred = pm.Deterministic(
                "party_time_effect_pred",
                pt.einsum('cb,ebp->ecp',
                          phi_baseline_pred,
                          _gp_coef_baseline * sqrt_psd_baseline_pred[:, None]),
                dims=("elections", "countdown", "parties_complete") # Dims match prediction coords
            )

            latent_mu = pm.Deterministic(
                 "latent_mu",
                 (
                     party_baseline[None, :] # Flat posterior
                     + _election_party_baseline[election_idx_data] # Flat posterior, indexed
                     + party_time_effect_pred[election_idx_data, countdown_idx_data] # Reconstructed, indexed
                 ),
                 dims=("observations", "parties_complete")
             )
            latent_popularity = pm.Deterministic( # Variable of interest
                "latent_popularity",
                pt.special.softmax(latent_mu, axis=1),
                dims=("observations", "parties_complete")
            )

        # --- 3. Sample Posterior Predictive ---
        print("Sampling posterior predictive for latent_popularity...")

        # Check required variables in posterior
        needed_vars_latent = ["party_baseline", "election_party_baseline", "gp_coef_baseline"]
        print("\n=== Checking required posterior variables ===")
        missing_for_latent = []
        for var in needed_vars_latent:
             if var in idata.posterior: print(f"✓ Found {var} in posterior")
             else:
                  print(f"✗ Missing {var} in posterior")
                  missing_for_latent.append(var)
        if missing_for_latent:
             raise ValueError(f"Missing required variables in posterior for latent prediction: {missing_for_latent}")

        # Prepare posterior values dictionary for injection
        posterior_values = {
            "party_baseline": idata.posterior["party_baseline"].values,
            # Use the Flat names defined in the prediction model context
            "_election_party_baseline": idata.posterior["election_party_baseline"].values,
            "_gp_coef_baseline": idata.posterior["gp_coef_baseline"].values,
        }

        with pred_model:
             pred_trace = pm.sample_posterior_predictive(
                 idata,
                 var_names=["latent_popularity"],
                 return_inferencedata=True,
                 predictions=True,
                 predictions_constant_data=posterior_values # Inject posterior values
             )

        print("Prediction finished.")

        if hasattr(pred_trace, 'predictions') and not hasattr(pred_trace, 'posterior_predictive'):
            pred_trace.posterior_predictive = pred_trace.predictions.copy()

        return pred_trace

    def predict(self, oos_data: pd.DataFrame) -> az.InferenceData:
        """Generate predictions for out-of-sample poll data using base attributes."""
        if self.trace is None: raise ValueError("Model must be fit before predicting.")
        if not hasattr(self, 'coords') or not self.coords: raise ValueError("Model coords not found.")
        if not hasattr(self, 'model') or self.model is None: raise ValueError("Model object not found.")

        # Use the *original* model context (self.model)
        with self.model:
            # Prediction coordinates based on oos_data index
            coords_pred = { "observations": oos_data.index } # minimal needed for pm.set_data

            # Map prediction data indices to *original* model coordinates
            original_pollsters = self.coords.get("pollsters", np.array([]))
            oos_pollsters = oos_data["pollster"].unique()
            new_pollsters = [p for p in oos_pollsters if p not in original_pollsters]
            pollster_map = {name: i for i, name in enumerate(original_pollsters)}
            pollster_indices_pred = oos_data["pollster"].map(lambda p: pollster_map.get(p, 0)).values # Map unknown to 0
            if new_pollsters: print(f"Warning: OOS data contains new pollsters {new_pollsters}. Mapping to index 0.")

            min_countdown = self.coords.get("countdown", np.array([0])).min()
            max_countdown = self.coords.get("countdown", np.array([0])).max()
            countdown_indices_pred = oos_data["countdown"].astype(int).values.clip(min_countdown, max_countdown)

            original_elections = self.coords.get("elections", np.array([]))
            try:
                 # Ensure consistent types for mapping (e.g., both strings)
                 oos_election_dates_str = oos_data["election_date"].astype(str).values
                 election_indices_pred = pd.Index(original_elections).get_indexer(oos_election_dates_str)
            except KeyError as e:
                 raise ValueError(f"Error mapping pred election dates: {e}. OOS dates must be in model coords['elections'].")
            if np.any(election_indices_pred == -1):
                 missing_dates = oos_data.loc[election_indices_pred == -1, "election_date"].unique()
                 raise ValueError(f"OOS data has election dates not in model: {missing_dates}")

            # Prepare data containers for pm.set_data, names must match original pm.Data calls
            # Use base attribute for parties
            is_here_pred = oos_data[self.political_families].astype(bool).astype(int)
            non_competing_add_pred = is_here_pred.replace(0, -10).replace(1, 0).to_numpy()
            non_competing_mult_pred = is_here_pred.to_numpy()

            prediction_data_dict = {
                "election_idx": election_indices_pred,
                "pollster_idx": pollster_indices_pred,
                "countdown_idx": countdown_indices_pred,
                "observed_N": oos_data["sample_size"].to_numpy(),
                "non_competing_polls_additive": non_competing_add_pred,
                "non_competing_polls_multiplicative": non_competing_mult_pred,
                # Add other pm.Data variables if needed, ensuring alignment
                # Example: stdz_gdp - needs careful alignment with oos_data
                # If self.campaign_preds exists and aligns with self.polls_train index:
                # \'stdz_gdp\': self.campaign_preds[\'gdp\'].reindex(oos_data.index).fillna(0).values # Risky if indices don\'t match
            }
            # Only add stdz_gdp if it exists in campaign_preds
            if hasattr(self, 'campaign_preds') and 'gdp' in self.campaign_preds.columns:
                 try:
                     # Align based on index, assuming oos_data has same index type as campaign_preds
                     aligned_gdp = self.campaign_preds['gdp'].reindex(oos_data.index).fillna(0).values
                     prediction_data_dict['stdz_gdp'] = aligned_gdp
                     print("Aligned \'stdz_gdp\' for prediction.")
                 except Exception as e:
                     print(f"Warning: Could not align \'stdz_gdp\' for prediction: {e}. Skipping.")

            # Set data using pm.set_data within the original model context
            # Pass the minimal coords_pred needed for the observations dimension
            pm.set_data(prediction_data_dict, model=self.model, coords=coords_pred)

            # Sample posterior predictive
            pred = pm.sample_posterior_predictive(
                self.trace,
                var_names=["noisy_popularity", "latent_popularity", "N_approve"],
                return_inferencedata=True,
                predictions=True
            )

        # Optional: Add observed data to the prediction trace
        try:
            oos_data_aligned = oos_data.set_index(pd.Index(coords_pred["observations"], name="observations"))
            # Use base attribute for parties
            observed_pred_xr = oos_data_aligned[self.political_families + ['sample_size']].to_xarray()
            pred.add_groups({"observed_data_pred": observed_pred_xr}, coords=coords_pred)
        except Exception as e:
            print(f"Warning: Could not add observed prediction data to trace: {e}")

        if hasattr(pred, 'predictions') and not hasattr(pred, 'posterior_predictive'):
             pred.posterior_predictive = pred.predictions.copy()

        return pred


    def predict_history(self, elections_to_predict: List[str]):
        """Calculate predictive accuracy for historical elections using base attributes."""
        if self.trace is None: raise ValueError("Model must be fit before predicting history.")
        if not hasattr(self, 'coords') or not self.coords: raise ValueError("Model coords not found.")
        if not hasattr(self, 'model') or self.model is None: raise ValueError("Model object not found.")

        elections_to_predict_dt = pd.to_datetime(elections_to_predict)

        # --- Coordinates for Prediction ---
        # Align with the dimensions of the data being set ('results_N', 'non_competing_parties_results')
        # These likely correspond to 'elections_observed' in the original model.
        # Ensure the coordinate name here matches the dimension name in pm.Data calls.
        observed_election_dim_name = "elections_observed" # Name used in R likelihood dim
        coords_pred = {
            observed_election_dim_name: elections_to_predict, # The specific elections
            # Use base attribute for parties
            "parties_complete": self.political_families,
            # Other coords might be needed by sample_posterior_predictive if R depends on them indirectly
            # Check original model definition for R's dependencies
            # 'latent_pop_t0_observed' depends on 'latent_pop_t0' which depends on time effect, baselines etc.
            # Include coords needed for those components.
            "elections": self.coords.get("elections", []),
            "countdown": self.coords.get("countdown", []), # Needed for time effect at t=0
             "gp_basis_baseline": self.coords.get("gp_basis_baseline", []), # Needed for time effect
             # Pollsters might not be directly needed if R doesn't depend on house effects
             # "pollsters": self.coords.get("pollsters", [])
        }
        # Remove empty coords
        coords_pred = {k: v for k, v in coords_pred.items() if len(v) > 0}
        print(f"Using prediction coordinates: {list(coords_pred.keys())}")


        # --- Prepare Data for pm.set_data ---
        # Filter original historical results (self.results_oos from base class)
        results_subset = self.results_oos[
            self.results_oos['election_date'].isin(elections_to_predict_dt)
        ].copy()

        # Ensure results_subset is sorted according to elections_to_predict
        results_subset['election_date_str'] = results_subset['election_date'].astype(str)
        results_subset = results_subset.set_index('election_date_str').reindex(elections_to_predict).reset_index()

        if results_subset['sample_size'].isnull().any():
             missing_dates = results_subset.loc[results_subset['sample_size'].isnull(), 'election_date_str'].tolist()
             raise ValueError(f"Missing sample size in results_oos for elections: {missing_dates}")

        # Prepare non_competing mask for the prediction subset
        # Use base attribute for parties
        # This mask needs to align with the 'elections' dimension used by latent_mu_t0
        # We need the original non_competing_parties_results mask and slice it.

        prediction_data_dict = {
             # Data for 'results_N', aligned with observed_election_dim_name
             "results_N": results_subset["sample_size"].to_numpy(),

             # Data for 'non_competing_parties_results'
             # Provide the *original* full mask from the base class.
             # pm.set_data should handle using the relevant parts based on coords_pred.
             "non_competing_parties_results": self.non_competing_parties_results_base,

             # We might also need to update indices if R depends on them?
             # Check R definition again: It uses latent_pop_t0_observed, which itself uses
             # self.observed_election_indices to slice latent_pop_t0.
             # When predicting, pm.sample_posterior_predictive internally handles indexing
             # based on the dimensions of the variable being predicted ('R' has dim elections_observed).
             # So we likely don't need to set election_idx etc. here.
        }


        # --- Set Data and Predict ---
        with self.model:
             # Set data, providing the coords for the dimensions being updated/predicted
             # The coords tell set_data how to interpret the provided data arrays.
             pm.set_data(prediction_data_dict, model=self.model, coords=coords_pred)

             # Sample posterior predictive for 'R'
             pred_history = pm.sample_posterior_predictive(
                self.trace,
                var_names=["R"], # Predict the results variable
                return_inferencedata=True,
                predictions=True
             )

        # --- Add Observed Data for Comparison ---
        try:
            # Prepare observed results as xarray aligned with prediction coords
            # Use base attribute for parties
            observed_results_xr = results_subset.set_index('election_date_str')[self.political_families].to_xarray()
            # Rename the dimension to match the prediction coordinate name
            observed_results_xr = observed_results_xr.rename({'election_date_str': observed_election_dim_name})
            pred_history.add_groups({"observed_data_history": observed_results_xr}, coords=coords_pred)
        except Exception as e:
            print(f"Warning: Could not add observed history data to trace: {e}")

        if hasattr(pred_history, 'predictions') and not hasattr(pred_history, 'posterior_predictive'):
             pred_history.posterior_predictive = pred_history.predictions.copy()

        return pred_history