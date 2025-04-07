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


class ElectionModel(BaseElectionModel):
    """A Bayesian election model"""

    def __init__(self, dataset: ElectionDataset, **kwargs):
        super().__init__(dataset, **kwargs)
        
        self.gp_config = {
            "baseline_lengthscale": kwargs.get("baseline_lengthscale", dataset.baseline_timescales),
            "election_lengthscale": kwargs.get("election_lengthscale", dataset.election_timescales),            
            "kernel": kwargs.get("kernel", "matern52"),
            "zerosum": kwargs.get("zerosum", True),
            "variance_limit": kwargs.get("variance_limit", 0.8),
        }
        self.pollster_id = None
        self.countdown_id = None
        self.election_id = None
        self.observed_election_indices = None
        self.gp_baseline = None

    def _build_coords(self, polls: pd.DataFrame = None):
        """Build the coordinates for the PyMC model"""
        data = polls if polls is not None else self.dataset.polls_train

        COORDS = {
            "observations": data.index,
            "parties_complete": self.dataset.political_families,
        }
        pollster_id, COORDS["pollsters"] = data["pollster"].factorize(sort=True)
        
        # Handle countdown values
        countdown_values = data["countdown"].values
        if not np.isfinite(countdown_values.max()):
            # If max is not finite, use the length of countdown values
            max_countdown = len(countdown_values) - 1
        else:
            max_countdown = int(countdown_values.max())
            
        countdown_id = countdown_values.astype(int)
        
        # For nowcasting, use the original countdown dimension to match the posterior samples
        if polls is not None and hasattr(self, 'coords') and 'countdown' in self.coords:
            COORDS["countdown"] = self.coords["countdown"]
        else:
            COORDS["countdown"] = np.arange(max_countdown + 1)
        
        # Use ALL election dates (historical + target) for the main election coordinate
        COORDS["elections"] = self.dataset.all_election_dates # Use all dates
        print(f"Setting 'elections' coord to ALL election cycles: {COORDS['elections']}") # Updated print

        # Get election indices for the data, mapping to the index in all_election_dates
        election_dates_in_data = data["election_date"].unique()
        election_id = np.zeros(len(data), dtype=int)
        for i, date in enumerate(data["election_date"]):
            # Find the index of this date in the *all_election_dates*
            for j, cycle_date_str in enumerate(self.dataset.all_election_dates): # Use all_election_dates
                 # Ensure comparison is between datetime objects or consistent types
                 if pd.to_datetime(cycle_date_str) == pd.to_datetime(date):
                     election_id[i] = j
                     break
                 elif j == len(self.dataset.all_election_dates) - 1: # If loop finishes without break
                      print(f"Warning: Poll date {date} associated election date not found in all_election_dates: {self.dataset.all_election_dates}. Assigning index 0.")
                      # Decide on fallback: assign to last known? first? raise error?
                      election_id[i] = 0 # Fallback to first election index

        # Coordinate for elections where we HAVE observed results (historical only)
        COORDS["elections_observed"] = self.dataset.historical_election_dates
        print(f"Setting 'elections_observed' coord to historical elections only: {COORDS['elections_observed']}")

        # Get indices for observed elections within the main 'elections' coordinate
        observed_election_indices = [i for i, election in enumerate(COORDS["elections"]) if election in COORDS["elections_observed"]]
        print(f"Indices for observed elections: {observed_election_indices}") # Debug print

        return pollster_id, countdown_id, election_id, COORDS, observed_election_indices

    def _build_data_containers(self, polls: pd.DataFrame = None):
        """Build the data containers for the PyMC model
        
        Parameters
        ----------
        polls : pd.DataFrame, optional
            Poll data to use. If None, use the training data.
        """
        if polls is None:
            polls = self.dataset.polls_train
        is_here_polls = polls[self.dataset.political_families] > 0 # True if votes > 0
        non_competing_polls_additive_np = np.where(is_here_polls, 0, -10).astype(np.int32)

        # Create non_competing_parties_results numpy array aligned with ALL election dates
        reindexed_results = (
            self.dataset.results_mult[self.dataset.political_families]
            .reindex(pd.to_datetime(self.dataset.all_election_dates))
        )
        # is_competing_mask is True only for parties with > 0 votes in historical results
        # NaN > 0 is False, so future date row will correctly be False here
        is_competing_mask = reindexed_results > 0
        non_competing_parties_results_np = np.where(is_competing_mask, 0, -10).astype(np.int32) # Shape (6, 8)

        print(f"Shape of non_competing_parties_results_np (aligned with all elections): {non_competing_parties_results_np.shape}")
        
        # Data containers for inference
        data_containers = dict(
            election_idx=pm.Data("election_idx", self.election_id, dims="observations"),
            pollster_idx=pm.Data("pollster_idx", self.pollster_id, dims="observations"),
            countdown_idx=pm.Data("countdown_idx", self.countdown_id, dims="observations"),
            stdz_gdp=pm.Data("stdz_gdp", self.dataset.campaign_preds["gdp"].to_numpy(), dims="observations"),
            election_gdp=pm.Data("election_gdp", self.dataset.results_preds["gdp"].to_numpy(), dims="elections"),
            observed_N=pm.Data("observed_N", polls["sample_size"].to_numpy(), dims="observations"),
            observed_polls=pm.Data(
                "observed_polls",
                polls[self.dataset.political_families].to_numpy(),
                dims=("observations", "parties_complete"),
            ),
            results_N=pm.Data(
                "results_N", 
                self.dataset.results_oos["sample_size"].to_numpy(),  # ONLY historical elections
                dims="elections_observed"
            ),
            observed_results=pm.Data(
                "observed_results",
                self.dataset.results_oos[self.dataset.political_families].to_numpy(),  # ONLY historical elections
                dims=("elections_observed", "parties_complete"),
            ),
            non_competing_parties_results=pm.Data(
                "non_competing_parties_results",
                non_competing_parties_results_np, # Pass the correctly shaped (6, 8) numpy array
                dims=("elections", "parties_complete"), # Dims now match the data shape
            ),
            non_competing_polls_additive=pm.Data(
                "non_competing_polls_additive",
                non_competing_polls_additive_np, # Pass the refined polls mask
                dims=("observations", "parties_complete"),
            ),
            non_competing_polls_multiplicative=pm.Data(
                "non_competing_polls_multiplicative",
                is_here_polls.astype(int).to_numpy(), # Use boolean mask directly for multiplicative
                dims=("observations", "parties_complete"),
            ),
            government_status=pm.Data(
                "government_status",
                self.dataset.government_status.values.astype(int),
                dims=("elections", "parties_complete"),
            )
        )

        return data_containers

    def build_model(self, polls: pd.DataFrame = None) -> pm.Model:
        """Build the PyMC model
        
        Parameters
        ----------
        polls : pd.DataFrame, optional
            Poll data to use. If None, use the training data.
        """
        (
            self.pollster_id,
            self.countdown_id,
            self.election_id,
            self.coords,
            self.observed_election_indices,
        ) = self._build_coords(polls)

        # Debug info about dimensions
        print(f"\n=== MODEL DIMENSIONS ===")
        for key, value in self.coords.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"{key}: shape={len(value)}")
        
        # Debug info about results_oos
        print(f"results_oos shape: {self.dataset.results_oos.shape}")
        print(f"results_oos election dates: {self.dataset.results_oos['election_date'].unique()}")
        
        # Check if results_oos and elections_observed match in length
        num_oos = len(self.dataset.results_oos)
        num_observed = len(self.coords["elections_observed"])
        if num_oos != num_observed:
            print(f"WARNING: Mismatch between results_oos ({num_oos}) and elections_observed ({num_observed})")
            
        # Ensure baseline_lengthscale is a list
        if not isinstance(self.gp_config["baseline_lengthscale"], list):
            self.gp_config["baseline_lengthscale"] = [self.gp_config["baseline_lengthscale"]]
            
        # Ensure election_lengthscale is a list
        if not isinstance(self.gp_config["election_lengthscale"], list):
            self.gp_config["election_lengthscale"] = [self.gp_config["election_lengthscale"]]

        with pm.Model(coords=self.coords) as model:
            data_containers = self._build_data_containers(polls)

            # --------------------------------------------------------
            #                   BASELINE COMPONENTS
            # --------------------------------------------------------
            party_baseline = pm.ZeroSumNormal(
                "party_baseline", sigma=0.5, dims="parties_complete" # Simpler prior for now
            )

            # Count actual number of elections in the dataset for proper dimensioning
            # num_elections = len(self.dataset.historical_election_dates) # Removed, no longer needed here
            # print(f"Setting election dimensions to match historical elections: {num_elections}") # Removed outdated print

            # Election-specific party baseline. TIGHTER fixed prior.
            election_party_baseline = pm.ZeroSumNormal(
                "election_party_baseline",
                sigma=0.15, 
                dims=("elections", "parties_complete") # Shape inferred from dims
                # shape=(num_elections, len(self.dataset.political_families)) # Removed conflicting shape arg
            )

            # --------------------------------------------------------
            #               TIME-VARYING COMPONENTS (Election Specific GP)
            # --------------------------------------------------------

            # Use only the FIRST baseline timescale
            baseline_lengthscale = self.gp_config["baseline_lengthscale"][0]
            print(f"Using single baseline timescale: {baseline_lengthscale}")

            cov_func_baseline = pm.gp.cov.Matern52(input_dim=1, ls=baseline_lengthscale)
            gp_baseline = pm.gp.HSGP(cov_func=cov_func_baseline, m=[20], c=2.0)
            phi_baseline, sqrt_psd_baseline = gp_baseline.prior_linearized(X=self.coords["countdown"][:, None])

            coord_name = f"gp_basis_baseline" # Simplified name
            if coord_name not in model.coords:
                model.add_coords({coord_name: np.arange(gp_baseline.n_basis_vectors)})

            gp_coef_baseline = pm.Normal(
                f"gp_coef_baseline", # Simplified name
                mu=0,
                sigma=1,
                # dims=(coord_name, "parties_complete") # Old dims
                dims=("elections", coord_name, "parties_complete") # New dims with elections
            )

            party_time_effect = pm.Deterministic(
                f"party_time_effect", # Simplified name
                # pt.dot(phi_baseline, gp_coef_baseline * sqrt_psd_baseline[:, None]), # Old calc
                pt.einsum('cb,ebp->ecp',
                          phi_baseline, # shape (countdown, gp_basis)
                          gp_coef_baseline * sqrt_psd_baseline[:, None] # shape (elections, gp_basis, parties_complete)
                         ),
                # dims=("countdown", "parties_complete") # Old dims
                dims=("elections", "countdown", "parties_complete") # New dims
            )

            # Removed party_time_weight for simplicity
            party_time_effect_weighted = party_time_effect # Directly use the effect

            # --------------------------------------------------------
            #          HOUSE EFFECTS & POLL BIAS
            # --------------------------------------------------------
            # Re-introduce house effects (pollster-specific only)
            house_effects_sd = pm.HalfNormal("house_effects_sd", sigma=0.15) # Tighter prior
            house_effects = pm.ZeroSumNormal(
                "house_effects",
                sigma=house_effects_sd,
                dims=("pollsters", "parties_complete"),
                shape=(len(self.coords["pollsters"]), len(self.dataset.political_families)) # Ensure correct shape
            )
            # Removed house_election_effects for simplicity
            # Removed poll_bias

            # --------------------------------------------------------
            #                      POLL RESULTS
            # --------------------------------------------------------

            # Compute latent_mu (Using election-specific GP)
            latent_mu = pm.Deterministic(
                "latent_mu",
                (
                    party_baseline[None, :]
                    + election_party_baseline[data_containers["election_idx"]] # Re-added baseline
                    # Index election-specific GP effect
                    + party_time_effect_weighted[data_containers["election_idx"], data_containers["countdown_idx"]]
                    + data_containers['non_competing_polls_additive']
                ),
                dims=("observations", "parties_complete")
            )

            # Apply softmax over parties for each observation
            latent_popularity = pm.Deterministic(
                "latent_popularity",
                pt.special.softmax(latent_mu, axis=1),
                dims=("observations", "parties_complete"),
            )

            # noisy_mu now includes house effects
            noisy_mu = pm.Deterministic(
                "noisy_mu",
                (
                    latent_mu
                    + house_effects[data_containers["pollster_idx"]] # Add only general house effect
                    # Removed house_election_effects
                ) * data_containers['non_competing_polls_multiplicative'],
                 # Note: non_competing_polls_multiplicative handles zeroing out non-competing parties
                dims=("observations", "parties_complete")
            )

            # Apply softmax over parties for each observation
            noisy_popularity = pm.Deterministic(
                "noisy_popularity",
                pt.special.softmax(noisy_mu, axis=1),
                dims=("observations", "parties_complete"),
            )

            # Use Gamma priors for concentration parameters based on ESS intuition
            # For polls: Mean=1000, SD=100 -> alpha=100, beta=0.1
            concentration_polls = pm.Gamma("concentration_polls", alpha=100, beta=0.1)

            # Generate counts from Dirichlet-Multinomial
            N_approve = pm.DirichletMultinomial(
                "N_approve",
                a=concentration_polls * noisy_popularity,
                n=data_containers["observed_N"],
                observed=data_containers["observed_polls"],
                dims=("observations", "parties_complete"),
            )

            # --------------------------------------------------------
            #                    ELECTION RESULTS
            # --------------------------------------------------------

            # Compute latent_mu_t0 (Using election-specific GP)
            latent_mu_t0 = pm.Deterministic(
                "latent_mu_t0",
                (
                    party_baseline[None, :]
                    + election_party_baseline # Re-added baseline
                    + party_time_effect_weighted[:, 0] # Effect at election day (t=0) for all elections
                    + data_containers['non_competing_parties_results']
                ),
                dims=("elections", "parties_complete")
            )

            # --- Define the FULL latent popularity trajectory over elections and countdown ---
            latent_popularity_full_trajectory_mu = pm.Deterministic(
                "latent_popularity_full_trajectory_mu",
                (
                     party_baseline[None, None, :]
                     + election_party_baseline[:, None, :] # Broadcast elections
                     + party_time_effect_weighted # Already has (elections, countdown, parties_complete)
                ),
                 # Ensure non_competing_parties_additive handled if needed
                 # Currently non_competing handled later or assumed 0 for full trajectory
                dims=("elections", "countdown", "parties_complete")
            )

            latent_popularity_trajectory = pm.Deterministic(
                "latent_popularity_trajectory",
                pt.special.softmax(latent_popularity_full_trajectory_mu, axis=2), # Softmax over parties_complete
                dims=("elections", "countdown", "parties_complete")
            )
            # --- End full trajectory definition ---

            # Apply softmax over parties for each observation
            latent_pop_t0 = pm.Deterministic(
                "latent_pop_t0",
                pt.special.softmax(latent_mu_t0, axis=1),
                dims=("elections", "parties_complete"),
            )

            # Use Gamma priors for concentration parameters based on ESS intuition
            # For results: Mean=2000, SD=200 -> alpha=100, beta=0.05
            concentration_results = pm.Gamma("concentration_results", alpha=100, beta=0.05)

            # DirichletMultinomial for the observed results
            # Select ONLY the latent popularity values corresponding to the observed elections
            latent_pop_t0_observed = latent_pop_t0[self.observed_election_indices]
            
            R = pm.DirichletMultinomial(
                "R",
                n=data_containers["results_N"], # Uses elections_observed dim
                a=concentration_results * latent_pop_t0_observed,  # Use the SLICED latent pop
                observed=data_containers["observed_results"], # Uses elections_observed dim
                dims=("elections_observed", "parties_complete") # Explicitly use observed dim
            )

            # Store gp_baseline instance for later use in prediction
            self.gp_baseline = gp_baseline 

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
        # Build the model
        model = self.build_model(self.dataset.polls_train)
        
        ppc = posterior.posterior_predictive
        ppc_results = {}

        # Print available keys in ppc for debugging
        print("Available keys in posterior_predictive:", ppc.data_vars.keys())

        # Compare observed data to posterior predictive distribution
        for i, party in enumerate(self.dataset.political_families):
            observed_polls = self.dataset.polls_train[party].values / self.dataset.polls_train['sample_size'].values
            observed_results = self.dataset.results_mult[party].values / self.dataset.results_mult['sample_size'].values
            
            # Use 'N_approve' from posterior_predictive
            predicted = ppc['N_approve'].values[:, :, :, i] / self.dataset.polls_train['sample_size'].values

            # Calculate mean absolute error for polls
            mae_polls = np.mean(np.abs(observed_polls - predicted.mean(axis=(0, 1))))
            ppc_results[f'{party}_mae_polls'] = mae_polls

            # Calculate coverage of 95% credible interval for polls
            lower, upper = np.percentile(predicted, [2.5, 97.5], axis=(0, 1))
            coverage_polls = np.mean((observed_polls >= lower) & (observed_polls <= upper))
            ppc_results[f'{party}_coverage_polls'] = coverage_polls

            # Calculate mean absolute error for results
            mae_results = np.mean(np.abs(observed_results - predicted.mean(axis=(0, 1))[-len(observed_results):]))
            ppc_results[f'{party}_mae_results'] = mae_results

            # Plot observed vs. predicted for polls
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.scatter(observed_polls, predicted.mean(axis=(0, 1)), label='Polls', alpha=0.5)
            plt.scatter(observed_results, predicted.mean(axis=(0, 1))[-len(observed_results):], label='Results', marker='x', s=100)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('Observed')
            plt.ylabel('Predicted')
            plt.title(f'Observed vs. Predicted for {party}')
            plt.legend()
            plt.savefig(f'ppc_plot_{party}.png')
            plt.close()

        return ppc_results 

    def nowcast_party_support(
        self, 
        idata: arviz.InferenceData, 
        current_polls: pd.DataFrame, 
        latest_election_date: str,
        model_output_dir: str = "."
    ) -> Tuple[arviz.InferenceData, Dict, Dict]:
        """
        Nowcast the current party support based on most recent polls.
        
        Following the approach from PyMC Labs' blog post on out-of-model predictions:
        https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc/
        
        Parameters
        ----------
        idata: arviz.InferenceData
            Posterior trace from a previously fit model
        current_polls: pd.DataFrame
            Current polls to use for nowcasting
        latest_election_date: str
            The date of the most recent election
            
        Returns
        -------
        tuple
            (posterior_predictive, prediction_coords, prediction_dims)
        """
        print("Running nowcast with current poll data...")
        
        # Prepare the current poll data
        current_polls = current_polls.copy()
        
        # Debug info about parties
        print("\n=== PARTY CONSISTENCY CHECK ===")
        print(f"Original model parties: {self.dataset.political_families}")
        print(f"Current polls parties: {[col for col in current_polls.columns if col in self.dataset.political_families]}")
        
        # Verify all required parties are present
        missing_parties = set(self.dataset.political_families) - set(current_polls.columns)
        if missing_parties:
            raise ValueError(f"Missing required parties in current polls: {missing_parties}")
        
        if not hasattr(self, 'model') or self.model is None:
             raise ValueError("Original model structure (`self.model`) not found or not built.")

        # Get reference to original model
        # original_model = self.model # Use the instance's model attribute
        
        # Check if model has coords, if not, try building it simply
        if not hasattr(self.model, 'coords') or not self.model.coords:
             print("Warning: Original model coordinates not found. Attempting a simple build.")
             try:
                 # Use the stored dataset reference
                 self._build_coords(self.dataset.polls_train) 
                 # A minimal build might be needed, or rely on coords stored in self.coords
             except Exception as e:
                 raise RuntimeError(f"Failed to establish model coordinates for nowcasting: {e}")
        
        # If self.model.coords is still not populated, use self.coords
        original_model_coords = self.model.coords if hasattr(self.model, 'coords') and self.model.coords else self.coords
        if not original_model_coords:
             raise RuntimeError("Failed to obtain model coordinates.")

        # Map current polls to model indices
        pollster_indices = np.zeros(len(current_polls), dtype=int)
        
        # Get pollster list directly from the posterior to ensure exact match
        if hasattr(idata, 'posterior') and 'house_effects' in idata.posterior:
            pollsters_list = idata.posterior.coords['pollsters'].values
            print(f"\n=== Using pollster list DIRECTLY FROM POSTERIOR ===")
        else:
            pollsters_list = np.atleast_1d(original_model_coords["pollsters"])
            print(f"\n=== Using pollster list from original model ===")
        
        # Check for new pollsters and print diagnostic info
        print("\n=== POLLSTER MAPPING DIAGNOSTICS ===")
        print(f"Posterior/model pollsters: {pollsters_list}")
        print(f"Current polls pollsters: {current_polls['pollster'].unique()}")
        
        unknown_pollsters = []
        for i, pollster in enumerate(current_polls["pollster"]):
            if pollster in pollsters_list:
                pollster_indices[i] = list(pollsters_list).index(pollster)
            else:
                unknown_pollsters.append(pollster)
                print(f"Warning: Pollster {pollster} not in original model. Using index 0.")
                pollster_indices[i] = 0
        
        if unknown_pollsters:
            print(f"Found {len(unknown_pollsters)} unknown pollsters: {unknown_pollsters}")
            print("This might cause sampling issues with house_effects variable.")
            
        # Ensure we're using exactly the same pollster list as in posterior
        prediction_coords = {
            "observations": pd.to_datetime(current_polls['date']).values,  # Use actual datetime objects
            "parties_complete": self.dataset.political_families,
            "pollsters": pollsters_list,  # Use exactly the same pollster list from posterior
            "elections": original_model_coords["elections"],
            "countdown": original_model_coords["countdown"],
        }
        
        # Debug: print house effects for current pollsters
        if hasattr(idata, 'posterior') and 'house_effects' in idata.posterior:
            print("\n=== DEBUG: HOUSE EFFECTS FOR CURRENT POLLSTERS ===")
            import matplotlib.pyplot as plt
            import os
            
            # Set the output directory
            self.model_output_dir = model_output_dir
            
            # Create debug plots directory
            debug_dir = f"{self.model_output_dir}/debug_plots"
            os.makedirs(debug_dir, exist_ok=True)
            
            # House effects by pollster
            he = idata.posterior['house_effects'].mean(dim=["chain", "draw"])
            for pollster in current_polls['pollster'].unique():
                if pollster in pollsters_list:
                    pollster_idx = list(pollsters_list).index(pollster)
                    pollster_he = he[pollster_idx]
                    print(f"\nHouse effects for {pollster}:")
                    for i, party in enumerate(self.dataset.political_families):
                        print(f"  {party}: {pollster_he[i].values:.4f}")
                else:
                    print(f"\nHouse effects for {pollster}: Using fallback (index 0)")
            
            # Plot components to understand latent popularity
            print("\n=== DEBUG: COMPONENT VISUALIZATIONS ===")
            
            # 1. Plot party_baseline
            party_baseline = idata.posterior['party_baseline'].mean(dim=["chain", "draw"])
            plt.figure(figsize=(10, 6))
            plt.bar(self.dataset.political_families, party_baseline.values)
            plt.title("Party Baseline")
            plt.xlabel("Party")
            plt.ylabel("Value")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "party_baseline.png"))
            plt.close()
            
            # 2. Plot party_time_effect_weighted
            if 'party_time_effect_weighted' in idata.posterior:
                pte = idata.posterior['party_time_effect_weighted'].mean(dim=["chain", "draw"])
                countdown_vals = original_model_coords["countdown"]
                for party_idx, party in enumerate(self.dataset.political_families):
                    plt.figure(figsize=(12, 6))
                    plt.plot(countdown_vals, pte.sel(parties_complete=party).values)
                    plt.title(f"Party Time Effect for {party}")
                    plt.xlabel("Countdown Days")
                    plt.ylabel("Effect Size")
                    plt.tight_layout()
                    plt.savefig(os.path.join(debug_dir, f"party_time_effect_{party}.png"))
                    plt.close()
            
            # 3. Plot poll_bias
            poll_bias = idata.posterior['poll_bias'].mean(dim=["chain", "draw"])
            plt.figure(figsize=(10, 6))
            plt.bar(self.dataset.political_families, poll_bias.values)
            plt.title("Poll Bias")
            plt.xlabel("Party")
            plt.ylabel("Value")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "poll_bias.png"))
            plt.close()
            
            # 4. Plot house_effects heatmap
            he_df = he.to_dataframe(name="value").reset_index()
            pivot_df = he_df.pivot(index="pollsters", columns="parties_complete", values="value")
            plt.figure(figsize=(12, 8))
            import seaborn as sns
            sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0)
            plt.title("House Effects by Pollster and Party")
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "house_effects_heatmap.png"))
            plt.close()
            
            # 5. Plot poll counts versus model predictions
            plt.figure(figsize=(15, 10))
            for i, party in enumerate(self.dataset.political_families):
                if party in current_polls.columns:
                    plt.subplot(2, 4, i+1)
                    poll_values = current_polls[party] / current_polls['sample_size']
                    plt.scatter(current_polls['date'], poll_values, label="Poll", alpha=0.7)
                    plt.title(f"{party}")
                    plt.xticks(rotation=45)
                    
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, "polls_vs_predictions.png"))
            plt.close()
            
            print(f"Debug plots saved to {debug_dir}")
        
        # Clip countdown values to valid range
        countdown_indices = current_polls["countdown"].astype(int).values.clip(
            0, len(original_model_coords["countdown"]) - 1
        )
        
        # Match election dates to indices
        historical_dates = sorted([pd.to_datetime(date) for date in self.dataset.historical_election_dates])
        latest_historical_date = historical_dates[-1]
        latest_historical_date_str = latest_historical_date.strftime("%Y-%m-%d")
        
        election_indices = np.zeros(len(current_polls), dtype=int)
        elections_list = np.atleast_1d(original_model_coords["elections"])
        
        for i, date in enumerate(current_polls["election_date"]):
            match_found = False
            for j, hist_date in enumerate(elections_list):
                if pd.to_datetime(hist_date) == date:
                    election_indices[i] = j
                    match_found = True
                    break
            
            if not match_found:
                # Find index of most recent election
                most_recent_index = None
                for j, hist_date in enumerate(elections_list):
                    if hist_date == latest_historical_date_str:
                        most_recent_index = j
                        break
                
                if most_recent_index is None:
                    # Fallback to last election in list
                    most_recent_index = len(elections_list) - 1
                
                print(f"IMPORTANT: Election date {date} not found in historical data.")
                print(f"           Using most recent election ({elections_list[most_recent_index]}) as a proxy.")
                election_indices[i] = most_recent_index
        
        # Check if current polls are already in multinomial format (counts) or need conversion
        is_percentage = (current_polls[self.dataset.political_families].sum(axis=1) < 2).any()
        
        if is_percentage:
            print("Current polls appear to be percentages - converting to counts")
            current_polls_mult = self.dataset.cast_as_multinomial(current_polls.copy())
            poll_counts = current_polls_mult[self.dataset.political_families].to_numpy()
        else:
            print("Current polls appear to already be in count format")
            poll_counts = current_polls[self.dataset.political_families].to_numpy()
        
        # Calculate non-competing masks for current polls
        is_here_current = current_polls[self.dataset.political_families].astype(bool).astype(int)
        non_competing_additive = is_here_current.replace(to_replace=0, value=-10).replace(to_replace=1, value=0).to_numpy()
        non_competing_multiplicative = is_here_current.to_numpy()
        
        # Build a fresh prediction model with the correct dimensions
        print("\n=== BUILDING PREDICTION MODEL ===")
        with pm.Model(coords=prediction_coords) as pred_model:
            # Define the exact same variables as in the original model, but with flat priors
            # These will be replaced by the posterior values when we sample

            # Basic components
            party_baseline = pm.Flat("party_baseline", dims="parties_complete")
            # election_party_baseline = pm.Flat("election_party_baseline", dims=("elections", "parties_complete")) # Assuming this is needed if used in original noisy_mu

            # Time components (Using full trajectory now)
            # Ensure party_time_effect_weighted exists in idata.posterior if needed for noisy_mu
            gp_coef_baseline = pm.Flat("gp_coef_baseline", dims=("elections", f"gp_basis_baseline", "parties_complete"))

            # House Effects
            house_effects_sd = pm.Flat("house_effects_sd")
            house_effects = pm.Flat("house_effects", dims=("pollsters", "parties_complete"))
            # Removed house_election_effects

            # Concentration parameter
            concentration_polls = pm.Flat("concentration_polls")

            # Create data containers
            election_idx = pm.Data("election_idx", election_indices, dims="observations")
            pollster_idx = pm.Data("pollster_idx", pollster_indices, dims="observations")
            countdown_idx = pm.Data("countdown_idx", countdown_indices, dims="observations")
            observed_N = pm.Data("observed_N", current_polls["sample_size"].astype(int).to_numpy(), dims="observations")
            non_competing_add = pm.Data(
                "non_competing_polls_additive",
                non_competing_additive,
                dims=("observations", "parties_complete")
            )
            non_competing_mult = pm.Data(
                "non_competing_polls_multiplicative",
                non_competing_multiplicative,
                dims=("observations", "parties_complete")
            )

            # Reconstruct necessary components for noisy_mu using posterior samples
            # Ensure these match the structure in build_model

            # Reconstruct party_time_effect if needed or use the stored version
            # This assumes party_time_effect is deterministic and can be rebuilt if needed
            # Or better, sample it directly if it's in idata.posterior

            # --- Reconstruct latent_mu ---
            # Need election_party_baseline from posterior if it was used
            # Need party_baseline from posterior
            # Need party_time_effect_weighted (or reconstruct it)

            # Placeholder for party_time_effect reconstruction if needed
            # If `party_time_effect` (the deterministic) is in idata, use it directly.
            # Otherwise, reconstruct using gp_coef_baseline, phi_baseline, etc.
            # Example reconstruction (adjust based on actual model structure):
            # Assume gp_baseline is available or reconstructible from idata constants/coords
            # phi_baseline, sqrt_psd_baseline = gp_baseline.prior_linearized(X=prediction_coords["countdown"][:, None])
            # party_time_effect = pt.einsum('cb,ebp->ecp', phi_baseline, gp_coef_baseline * sqrt_psd_baseline[:, None])

            # Use election_party_baseline from posterior
            election_party_baseline = pm.Flat("election_party_baseline", dims=("elections", "parties_complete"))

            # Latent mu calculation (ensure variables exist in idata.posterior)
            # Need to ensure party_time_effect variable used here matches the one derived from posterior samples
            # If 'party_time_effect' itself is sampled/stored, use it. If not, reconstruct.
            # Let's assume 'party_time_effect' *needs* reconstruction for now.
            # We need phi_baseline and sqrt_psd_baseline. We might need to recalculate them
            # based on the original model's gp_baseline parameters if not stored in idata.
            # For simplicity, let's *assume* party_time_effect_weighted is directly available from the posterior for now.
            # If not, this reconstruction step is complex.
            # Check if 'party_time_effect' is in idata.posterior
            if 'party_time_effect' in idata.posterior:
                 party_time_effect_pred = pm.Flat("party_time_effect", dims=("elections", "countdown", "parties_complete"))
            else:
                 # Fallback: This requires complex reconstruction not shown here.
                 # Need gp_baseline parameters from idata.
                 print("WARNING: 'party_time_effect' not found in posterior. Prediction might be inaccurate.")
                 # Ensure gp_basis_coord_name is defined
                 gp_basis_coord_name = "gp_basis_baseline" # Assuming this is the name
                 # Create a placeholder Flat variable to avoid errors, but it won't use posterior info
                 party_time_effect_pred = pm.Flat("party_time_effect_placeholder", dims=("elections", "countdown", gp_basis_coord_name, "parties_complete")) # Adjust dims if needed


            latent_mu = pm.Deterministic(
                 "latent_mu",
                 (
                     party_baseline[None, :]  # From posterior
                     + election_party_baseline[election_idx] # From posterior
                     + party_time_effect_pred[election_idx, countdown_idx] # From posterior (or reconstructed)
                     + non_competing_add # New data
                 ),
                 dims=("observations", "parties_complete")
             )

            # Apply softmax over parties for each observation
            latent_popularity = pm.Deterministic(
                "latent_popularity",
                pt.special.softmax(latent_mu, axis=1),
                dims=("observations", "parties_complete")
            )

            # Compute noisy_mu using the same formula as the original model, including house effects
            noisy_mu = pm.Deterministic(
                "noisy_mu",
                (
                    latent_mu # Already incorporates election-specific effects
                    + house_effects[pollster_idx] # From posterior
                    # Removed house_election_effects
                ) * non_competing_mult, # New data
                dims=("observations", "parties_complete")
            )

            # Apply softmax over parties for each observation
            noisy_popularity = pm.Deterministic(
                "noisy_popularity",
                pt.special.softmax(noisy_mu, axis=1),
                dims=("observations", "parties_complete")
            )

            # Generate counts from Dirichlet-Multinomial
            N_approve = pm.DirichletMultinomial(
                "N_approve",
                a=concentration_polls * noisy_popularity,
                n=observed_N,
                dims=("observations", "parties_complete")
            )

            # Debug: Check which variables could be found in posterior
            needed_vars = [
                "party_baseline",
                "election_party_baseline",
                # "party_time_effect", # Check if this exists, otherwise need coefficients
                "gp_coef_baseline", # Need coefficients if effect not stored
                "house_effects_sd", # Added
                "house_effects", # Added
                # Removed "house_election_effects"
                "concentration_polls"
            ]
            for var in needed_vars:
                if var in idata.posterior:
                    print(f"✓ Found {var} in posterior")
                else:
                    print(f"✗ Missing {var} in posterior")
            
            # Sample posterior predictive
            pred_trace = pm.sample_posterior_predictive(
                idata,
                var_names=[
                    "latent_popularity",
                    "noisy_popularity",
                    "N_approve",
                    # Also sample the components if needed for debugging/analysis
                    "latent_mu",
                    "noisy_mu",
                 ],
                return_inferencedata=True,
                predictions=True
            )
            
            # Debug: check if poll values are being passed to the predictive model
            print("\n=== DEBUG: POLL VALUES PASSED TO MODEL ===")
            poll_data = {}
            poll_data["observed_polls"] = observed_polls if "observed_polls" in pred_model.named_vars else None
            poll_data["observed_N"] = observed_N if "observed_N" in pred_model.named_vars else None
            
            if poll_data["observed_polls"] is not None:
                print("observed_polls shape:", poll_data["observed_polls"].eval().shape)
                print("First few values for parties:")
                poll_values = poll_data["observed_polls"].eval()
                for i, party in enumerate(self.dataset.political_families[:3]):  # First 3 parties
                    print(f"  {party}:", poll_values[:3, i])
                    
            if poll_data["observed_N"] is not None:
                print("observed_N shape:", poll_data["observed_N"].eval().shape)
                print("First few values:", poll_data["observed_N"].eval()[:3])
                
            # Add posterior_predictive group by copying predictions
            if hasattr(pred_trace, 'predictions') and not hasattr(pred_trace, 'posterior_predictive'):
                print("Adding posterior_predictive group by copying from predictions")
                pred_trace.posterior_predictive = pred_trace.predictions.copy()
                
                # Debug: Check coordinates
                print("\n=== DEBUG: CHECKING COORDINATES AFTER SAMPLING ===")
                if hasattr(pred_trace.predictions, 'coords'):
                    print(f"Prediction coordinate keys: {list(pred_trace.predictions.coords.keys())}")
                    if 'observations' in pred_trace.predictions.coords:
                        obs = pred_trace.predictions.coords['observations'].values
                        print(f"Observations type: {type(obs)}, first 3 values: {obs[:3]}")
                        
                        # Ensure the observations are datetime objects
                        if not isinstance(obs[0], (pd.Timestamp, np.datetime64)):
                            print("Converting observation indices to datetimes...")
                            # Replace observation coordinates with real dates
                            new_obs = pd.to_datetime(current_polls['date']).values
                            print(f"New observations: {new_obs[:3]}")
                            # Update both predictions and posterior_predictive
                            pred_trace.predictions = pred_trace.predictions.assign_coords(observations=new_obs)
                            pred_trace.posterior_predictive = pred_trace.posterior_predictive.assign_coords(observations=new_obs)
                    
                    # Debug: Check values to diagnose poor fit
                    print("\n=== DEBUG: CHECKING PREDICTION VALUES ===")
                    latent_pop = None
                    if 'latent_popularity' in pred_trace.posterior_predictive:
                        latent_pop = pred_trace.posterior_predictive.latent_popularity
                        print(f"Latent popularity shape: {latent_pop.shape}")
                        for party in latent_pop.coords['parties_complete'].values[:3]:  # First 3 parties
                            party_latent = latent_pop.sel(parties_complete=party)
                            mean_values = party_latent.mean(("chain", "draw")).values
                            print(f"Party {party} mean values (first 3): {mean_values[:3]}")
                            
                        # Compare with poll values
                        print("\n=== DEBUG: COMPARING TO POLL VALUES ===")
                        # Get poll values directly from current_polls dataframe
                        print("Accessing poll values directly from current_polls")
                        for i, party in enumerate(self.dataset.political_families[:3]):  # First 3 parties
                            if party in current_polls.columns:
                                if is_percentage:
                                    polls_party = current_polls[party].values[:3]
                                else:
                                    # If polls are counts, convert to proportions for comparison
                                    polls_party = current_polls[party].values[:3] / current_polls['sample_size'].values[:3]
                                print(f"Party {party} poll values as proportions (first 3): {polls_party}")
                        
                        # Plot noisy_popularity and latent_popularity comparison
                        os.makedirs(f"{self.model_output_dir}/debug_plots", exist_ok=True)
                        print("\n=== DEBUG: PLOTTING NOISY VS LATENT POPULARITY ===")
                        
                        # Extract variables from posterior predictive
                        noisy_popularity = pred_trace.predictions.noisy_popularity.mean(dim=["chain", "draw"])
                        latent_popularity = pred_trace.predictions.latent_popularity.mean(dim=["chain", "draw"])
                        
                        # Get dates for x-axis
                        dates = pd.to_datetime(noisy_popularity.coords["observations"].values)
                        
                        for i, party in enumerate(self.dataset.political_families):
                            if party not in current_polls.columns:
                                print(f"Skipping {party} - not found in current polls")
                                continue
                                
                            plt.figure(figsize=(12, 8))
                            # Plot noisy_popularity
                            plt.plot(dates, noisy_popularity.sel(parties_complete=party).values, 
                                     'o-', color='red', label='Noisy Popularity (Model)')
                            
                            # Plot latent_popularity
                            plt.plot(dates, latent_popularity.sel(parties_complete=party).values, 
                                     'o-', color='blue', label='Latent Popularity (Model)')
                            
                            # Plot actual polls
                            if is_percentage:
                                poll_values_party = current_polls[party].values
                            else:
                                poll_values_party = current_polls[party].values / current_polls['sample_size'].values
                                
                            plt.scatter(dates, poll_values_party, color='green', 
                                        s=100, alpha=0.7, label='Actual Poll Values', zorder=5)
                            
                            # Add confidence intervals for latent popularity
                            try:
                                latent_party_hdi = az.hdi(pred_trace.predictions.latent_popularity.sel(parties_complete=party))
                                
                                # Directly access the data values from the latent_popularity variable in the Dataset
                                if isinstance(latent_party_hdi, xr.Dataset) and 'latent_popularity' in latent_party_hdi.data_vars:
                                    hdi_data = latent_party_hdi['latent_popularity'].values
                                    # Shape should be (23, 2) with first column being lower bound, second being upper
                                    latent_lower = hdi_data[:, 0]  # Lower bound
                                    latent_upper = hdi_data[:, 1]  # Upper bound
                                else:
                                    # Fallback to direct array access if structure is different
                                    latent_lower = latent_party_hdi.isel(hdi=0).values
                                    latent_upper = latent_party_hdi.isel(hdi=1).values
                                
                                # Convert to numpy arrays and ensure they are numeric
                                latent_lower = np.asarray(latent_lower, dtype=np.float64)
                                latent_upper = np.asarray(latent_upper, dtype=np.float64)
                                plt.fill_between(dates, latent_lower, latent_upper, color='blue', alpha=0.2)
                            except Exception as e:
                                print(f"Error plotting latent HDI for {party}: {e}")
                                print(f"HDI result type: {type(latent_party_hdi)}")
                                if hasattr(latent_party_hdi, 'dims'):
                                    print(f"HDI dims: {latent_party_hdi.dims}")
                                if isinstance(latent_party_hdi, xr.Dataset):
                                    print(f"HDI data vars: {list(latent_party_hdi.data_vars)}")
                            
                            # Add confidence intervals for noisy popularity
                            try:
                                noisy_party_hdi = az.hdi(pred_trace.predictions.noisy_popularity.sel(parties_complete=party))
                                
                                # Directly access the data values from the noisy_popularity variable in the Dataset
                                if isinstance(noisy_party_hdi, xr.Dataset) and 'noisy_popularity' in noisy_party_hdi.data_vars:
                                    hdi_data = noisy_party_hdi['noisy_popularity'].values
                                    # Shape should be (23, 2) with first column being lower bound, second being upper
                                    noisy_lower = hdi_data[:, 0]  # Lower bound
                                    noisy_upper = hdi_data[:, 1]  # Upper bound
                                else:
                                    # Fallback to direct array access if structure is different
                                    noisy_lower = noisy_party_hdi.isel(hdi=0).values
                                    noisy_upper = noisy_party_hdi.isel(hdi=1).values
                                
                                # Convert to numpy arrays and ensure they are numeric
                                noisy_lower = np.asarray(noisy_lower, dtype=np.float64)
                                noisy_upper = np.asarray(noisy_upper, dtype=np.float64)
                                plt.fill_between(dates, noisy_lower, noisy_upper, color='red', alpha=0.2)
                            except Exception as e:
                                print(f"Error plotting noisy HDI for {party}: {e}")
                                print(f"HDI result type: {type(noisy_party_hdi)}")
                                if hasattr(noisy_party_hdi, 'dims'):
                                    print(f"HDI dims: {noisy_party_hdi.dims}")
                                if isinstance(noisy_party_hdi, xr.Dataset):
                                    print(f"HDI data vars: {list(noisy_party_hdi.data_vars)}")
                            
                            plt.title(f'{party} - Latent vs Noisy Popularity Comparison')
                            plt.xlabel('Date')
                            plt.ylabel('Support %')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            
                            # Save figure
                            plt.savefig(f"{self.model_output_dir}/debug_plots/noisy_vs_latent_{party}.png")
                            plt.close()
                            
                        print(f"Noisy vs Latent popularity plots saved to {self.model_output_dir}/debug_plots")

            print("Generating plots...")
        
        # Define prediction dimensions for the returned trace
        prediction_dims = {
            "latent_popularity": ["observations", "parties_complete"],
            "noisy_popularity": ["observations", "parties_complete"],
            "N_approve": ["observations", "parties_complete"],
            "latent_mu": ["observations", "parties_complete"], # Added
            "noisy_mu": ["observations", "parties_complete"] # Added
        }
        
        return pred_trace, prediction_coords, prediction_dims 

    # --- Method for Predicting Latent Trajectory (New) ---
    def predict_latent_trajectory(self, 
                                idata: az.InferenceData, 
                                start_date: pd.Timestamp, 
                                end_date: pd.Timestamp) -> az.InferenceData:
        """Predicts the latent popularity trajectory over a specified date range.

        Uses sample_posterior_predictive with the provided trace.
        
        Args:
            idata: InferenceData object containing the posterior trace.
            start_date: The start date for the prediction period.
            end_date: The end date for the prediction period.

        Returns:
            InferenceData containing the posterior predictive samples for latent_popularity.
        """
        print(f"\n--- Predicting Latent Trajectory from {start_date.date()} to {end_date.date()} ---")
        
        # Check for necessary attributes
        if not hasattr(self, 'gp_baseline') or self.gp_baseline is None:
             # Try to rebuild the model to potentially set gp_baseline if not already set
             print("Warning: gp_baseline not found. Attempting to rebuild model...")
             try:
                 _ = self.build_model() # Rebuild to ensure gp_baseline is set
                 if not hasattr(self, 'gp_baseline') or self.gp_baseline is None:
                      raise RuntimeError("Failed to set gp_baseline even after rebuild.")
             except Exception as e:
                 raise RuntimeError(f"Original GP object (`self.gp_baseline`) not found and couldn't be rebuilt: {e}")
                 
        if not hasattr(self, 'coords') or not self.coords:
             # Try rebuilding coords if missing
             print("Warning: coords not found. Attempting to rebuild coords...")
             try:
                 _, _, _, self.coords, _ = self._build_coords(self.dataset.polls_train)
                 if not hasattr(self, 'coords') or not self.coords:
                     raise RuntimeError("Failed to set coords even after rebuild.")
             except Exception as e:
                 raise RuntimeError(f"Original model coordinates (`self.coords`) not found and couldn't be rebuilt: {e}")

        # Use a known date from coords if election_date attribute isn't reliably set
        if 'elections' not in self.coords or len(self.coords['elections']) == 0:
             raise RuntimeError("Model 'elections' coordinate not found or empty.")
        # Assuming the last date in the 'elections' coordinate is the reference
        reference_end_date_str = self.coords['elections'][-1]
        try:
            reference_end_date = pd.to_datetime(reference_end_date_str)
        except Exception as e:
            raise ValueError(f"Could not parse the last election date '{reference_end_date_str}' from coords: {e}")

        # --- 1. Define Prediction Coordinates for the Date Range ---
        original_coords = self.coords
        gp_basis_coord_name = "gp_basis_baseline" # Make sure this matches the name in build_model
        
        # Ensure countdown coordinate exists and has values
        if 'countdown' not in original_coords or len(original_coords['countdown']) == 0:
            raise ValueError("Original countdown coordinate is missing or empty.")
            
        # Determine the full date range potentially covered by the original countdown
        max_countdown_days = original_coords['countdown'].max()
        min_countdown_days = original_coords['countdown'].min()
        earliest_date_in_model = reference_end_date - pd.Timedelta(days=int(max_countdown_days))
        latest_date_in_model = reference_end_date - pd.Timedelta(days=int(min_countdown_days))

        # Create the sequence of prediction days within the requested range
        pred_days = pd.date_range(start=start_date, end=end_date, freq='D')

        if pred_days.empty:
             print("Warning: No days found in the specified prediction range.")
             return az.InferenceData() # Return empty InferenceData
             
        # Create countdown values relative to the reference end date for the prediction days
        pred_countdown_values = (reference_end_date - pred_days).days.values
        
        # Clip countdown values to the range actually present in the original model's countdown coordinate
        min_coord_countdown = original_coords['countdown'].min()
        max_coord_countdown = original_coords['countdown'].max()
        pred_countdown_values_clipped = np.clip(pred_countdown_values, 
                                               min_coord_countdown, 
                                               max_coord_countdown)
                                        
        # Get the unique countdown values needed for prediction that ARE in the original coords
        pred_countdown_unique_vals = np.unique(pred_countdown_values_clipped)
        valid_pred_countdown_coord_vals = pred_countdown_unique_vals[
            np.isin(pred_countdown_unique_vals, original_coords['countdown'])
        ]
        
        if len(valid_pred_countdown_coord_vals) == 0:
             print(f"Warning: No valid countdown indices (range {min_coord_countdown}-{max_coord_countdown}) found for the prediction date range.")
             print(f"Requested dates: {start_date.date()} to {end_date.date()}")
             print(f"Calculated countdown values: {pred_countdown_values}")
             print(f"Clipped countdown values: {pred_countdown_values_clipped}")
             return az.InferenceData()

        # Filter the prediction days to only those whose countdown value is valid
        valid_days_mask = np.isin(pred_countdown_values_clipped, valid_pred_countdown_coord_vals)
        pred_days_final = pred_days[valid_days_mask]
        pred_countdown_values_final = pred_countdown_values_clipped[valid_days_mask]

        if pred_days_final.empty:
             print("Warning: No observation days remaining after filtering for valid countdown values.")
             return az.InferenceData()

        # Create the 'countdown' coordinate for the prediction model containing only the necessary values
        pred_model_countdown_coord = np.sort(valid_pred_countdown_coord_vals) 
        # Map each final prediction day's countdown value to its index in this new coordinate
        countdown_map_to_pred_coord = {val: i for i, val in enumerate(pred_model_countdown_coord)}
        countdown_indices_for_obs = pd.Series(pred_countdown_values_final).map(countdown_map_to_pred_coord).values.astype(int)

        # Use original coordinate values for other dimensions, but filtered countdown for the model coord
        prediction_coords = {
            "observations": pred_days_final.values, # Use filtered dates
            "parties_complete": self.dataset.political_families,
            "elections": original_coords["elections"], # Keep original elections structure
            "countdown": pred_model_countdown_coord, # Use the minimal set of countdown values
            gp_basis_coord_name: original_coords[gp_basis_coord_name], # Keep original GP basis
        }
        print(f"Prediction coordinates defined for {len(pred_days_final)} days.")
        print(f"Using {len(pred_model_countdown_coord)} unique countdown indices for prediction model.")

        # --- 2. Build Prediction Model Context ---\
        with pm.Model(coords=prediction_coords) as pred_model:
            # Flat variables from posterior (must match names in idata.posterior)
            party_baseline = pm.Flat("party_baseline", dims="parties_complete")
            # Hierarchical variables need priors IF they might extend beyond original coords,
            # but here we are only predicting latent_popularity based on existing parameters.
            # So, we'll use Flat variables for the parts of the hierarchy we need.
            
            # Instead of priors, use Flat for the posterior means/samples of hierarchical params
            # We need election_party_baseline and gp_coef_baseline
            # These will be indexed later using data containers.
            _election_party_baseline = pm.Flat("_election_party_baseline", dims=("elections", "parties_complete"))
            _gp_coef_baseline = pm.Flat("_gp_coef_baseline", dims=("elections", gp_basis_coord_name, "parties_complete"))

            # --- Define Data containers ONLY for indexing ---
            # Map observation days to countdown indices in the *prediction* coord
            countdown_idx_data = pm.Data("countdown_idx_data", countdown_indices_for_obs, dims="observations")

            # Assume we predict latent trajectory *within the context of the last election cycle*
            # This is an approximation.
            last_election_idx_in_original = len(original_coords["elections"]) - 1
            election_indices_pred = np.full(len(pred_days_final), last_election_idx_in_original, dtype=int)
            # Map these indices to the 'elections' coordinate of the prediction model (which is the same as original)
            election_idx_data = pm.Data("election_idx_data", election_indices_pred, dims="observations")

            # --- Reconstruct Deterministic variables --- 
            # Calculate GP basis functions using the prediction model's 'countdown' coordinate
            phi_baseline, sqrt_psd_baseline = self.gp_baseline.prior_linearized(
                X=prediction_coords["countdown"][:, None] # Use the coord from prediction_coords
            )
            
            # Reconstruct party_time_effect using Flat posterior coefs
            # Need to ensure einsum dimensions match Flat variables and phi_baseline
            # phi_baseline shape: (pred_countdown_len, gp_basis)
            # _gp_coef_baseline shape: (elections, gp_basis, parties_complete)
            # sqrt_psd_baseline shape: (gp_basis,)
            # Result needed: (elections, pred_countdown_len, parties_complete)
            party_time_effect = pm.Deterministic(
                "party_time_effect_pred", # Use distinct name to avoid clash if original name exists
                pt.einsum('cb,ebp->ecp', 
                          phi_baseline, 
                          _gp_coef_baseline * sqrt_psd_baseline[:, None]), # Use Flat coefs
                dims=("elections", "countdown", "parties_complete") # Dims match prediction_coords
            )
            
            # Calculate latent_mu using Flat posterior params and data indices
            latent_mu = pm.Deterministic(
                 "latent_mu",
                 (
                     party_baseline[None, :] # Flat posterior baseline
                     + _election_party_baseline[election_idx_data] # Flat posterior election baseline, indexed
                     + party_time_effect[election_idx_data, countdown_idx_data] # Reconstructed time effect, indexed
                     # No non_competing needed for latent trajectory prediction itself
                 ),
                 dims=("observations", "parties_complete")
             )
            latent_popularity = pm.Deterministic(
                "latent_popularity", # This is the variable we want
                pt.special.softmax(latent_mu, axis=1),
                dims=("observations", "parties_complete")
            )
            
        # --- 3. Sample Posterior Predictive ---
        print("Sampling posterior predictive for latent_popularity...")
        
        # We need to manually pass the posterior values for the Flat variables
        posterior_values = {
            "party_baseline": idata.posterior["party_baseline"].values,
            "_election_party_baseline": idata.posterior["election_party_baseline"].values,
            "_gp_coef_baseline": idata.posterior["gp_coef_baseline"].values,
        }

        with pred_model: 
             # Use sample_posterior_predictive with posterior values injected
             pred_trace = pm.sample_posterior_predictive(
                 idata, # Pass the original trace for structure/draw count
                 var_names=["latent_popularity"], # Only predict this
                 return_inferencedata=True,
                 predictions=True,
                 predictions_constant_data=posterior_values # Inject posterior values
             )

        print("Prediction finished.")
        
        # Add posterior_predictive group if needed
        if hasattr(pred_trace, 'predictions') and not hasattr(pred_trace, 'posterior_predictive'):
            pred_trace.posterior_predictive = pred_trace.predictions.copy()
            
        return pred_trace
        
    # --- End New Method ---

    def predict(self, oos_data: pd.DataFrame) -> az.InferenceData:
        """Generate predictions for out-of-sample poll data."""
        if self.trace is None:
            raise ValueError("Model must be fit with run_inference() before predicting")
        if not hasattr(self, 'coords') or not self.coords:
            raise ValueError("Model coordinates not found. Fit the model first.")
        if not hasattr(self, 'model') or self.model is None:
             raise ValueError("Original model object (`self.model`) not found.")

        # Use the *original* model context for setting data
        with self.model:
            # Define prediction coordinates based on the shape of oos_data
            # These coordinates MUST match the dimensions expected by pm.set_data
            # which are based on the *original* model's coordinates.
            coords_pred = {
                 "observations": oos_data.index, # Index of prediction points
                 "parties_complete": self.coords["parties_complete"], # From original model
                 # We don't redefine elections, pollsters, countdown etc. here
                 # We map the oos_data to the *existing* coordinates used by pm.set_data
             }

            # Map prediction data indices to *original* model coordinates
            # Check for pollsters not seen during training
            oos_pollsters = oos_data["pollster"].unique()
            original_pollsters = self.coords["pollsters"]
            new_pollsters = [p for p in oos_pollsters if p not in original_pollsters]
            if new_pollsters:
                 # Option 1: Raise error (simplest)
                 # raise ValueError(f"OOS data contains pollsters not present in the original model fit: {new_pollsters}")
                 # Option 2: Map new pollsters to a default index (e.g., 0) or handle differently
                 print(f"Warning: OOS data contains new pollsters {new_pollsters}. Mapping them to index 0.")
                 pollster_map = {name: i for i, name in enumerate(original_pollsters)}
                 pollster_indices_pred = oos_data["pollster"].map(lambda p: pollster_map.get(p, 0)).values
            else:
                 pollster_indices_pred = pd.Index(original_pollsters).get_indexer(oos_data["pollster"])

            # Clip countdown to ensure it's within the bounds of the original coordinate
            min_countdown = self.coords["countdown"].min()
            max_countdown = self.coords["countdown"].max()
            countdown_indices_pred = oos_data["countdown"].astype(int).values.clip(min_countdown, max_countdown)
            
            # Map election dates
            original_elections = self.coords["elections"]
            try:
                 election_indices_pred = pd.Index(original_elections).get_indexer(oos_data["election_date"].astype(str)) # Ensure consistent type
            except KeyError as e:
                 raise ValueError(f"Error mapping prediction election dates: {e}. Ensure OOS dates are in model coords['elections'].")

            # Check for invalid indices (-1) after mapping
            if np.any(pollster_indices_pred == -1) and not new_pollsters: # Only error if not handled above
                 raise ValueError("Internal error: Failed to map pollsters.")
            if np.any(election_indices_pred == -1):
                 missing_dates = oos_data.loc[election_indices_pred == -1, "election_date"].unique()
                 raise ValueError(f"OOS data contains election dates not present in the original model fit: {missing_dates}")
                 
            # Prepare data containers for pm.set_data (names must match pm.Data in original model)
            is_here_pred = oos_data[self.dataset.political_families].astype(bool).astype(int)
            non_competing_add_pred = is_here_pred.replace(to_replace=0, value=-10).replace(to_replace=1, value=0).to_numpy()
            non_competing_mult_pred = is_here_pred.to_numpy()

            # Prepare the dictionary for pm.set_data
            prediction_data_dict = {
                "election_idx": election_indices_pred,
                "pollster_idx": pollster_indices_pred,
                "countdown_idx": countdown_indices_pred,
                "observed_N": oos_data["sample_size"].to_numpy(),
                "non_competing_polls_additive": non_competing_add_pred,
                "non_competing_polls_multiplicative": non_competing_mult_pred,
                # Add other data containers if they exist in the original model and are needed
                # e.g., "stdz_gdp" - requires mapping oos_data rows to the original GDP data
                # Placeholder: Assume stdz_gdp is not needed or handled internally by model structure
            }
            
            # Set data using pm.set_data within the original model context
            pm.set_data(prediction_data_dict, model=self.model, coords=coords_pred) # Pass coords here

            # Sample posterior predictive using the original model context and trace
            pred = pm.sample_posterior_predictive(
                self.trace,
                var_names=["noisy_popularity", "latent_popularity", "N_approve"], # Variables to predict
                return_inferencedata=True,
                predictions=True # Ensure predictions group is created
            )

        # Optional: Add observed data to the prediction trace for easier comparison
        try:
            # Align observed data with prediction coordinates
            oos_data_aligned = oos_data.set_index(pd.Index(coords_pred["observations"], name="observations"))
            pred.add_groups({"observed_data_pred": oos_data_aligned.to_xarray()}, coords=coords_pred)
        except Exception as e:
            print(f"Warning: Could not add observed prediction data to trace: {e}")
        
        # Add posterior_predictive group if only predictions group exists
        if hasattr(pred, 'predictions') and not hasattr(pred, 'posterior_predictive'):
             pred.posterior_predictive = pred.predictions.copy()

        return pred

    def predict_history(self, elections_to_predict: List[str]):
        """Calculate predictive accuracy for historical elections."""
        if self.trace is None:
             raise ValueError("Model must be fit with run_inference() before predicting historical accuracy")
        if not hasattr(self, 'coords') or not self.coords:
             raise ValueError("Model coordinates not found. Fit the model first.")
        if not hasattr(self, 'model') or self.model is None:
             raise ValueError("Original model object (`self.model`) not found.")

        # Convert string dates to datetime objects for comparison
        elections_to_predict_dt = pd.to_datetime(elections_to_predict)

        # --- Coordinates for Prediction ---
        # We are predicting 'R', which in the original model was defined over 'elections_observed'.
        # The pm.set_data call needs coordinates that align with the data being set.
        # The primary data being set is 'results_N' and 'non_competing_parties_results'.
        # These need to be aligned with the 'elections_observed' dimension.
        coords_pred = {
            "elections_observed": elections_to_predict, # The specific elections we are predicting results for
            "parties_complete": self.coords["parties_complete"],
            # Include other original coords if needed by pm.sample_posterior_predictive later
            "elections": self.coords["elections"], 
            "countdown": self.coords["countdown"], 
            "gp_basis_baseline": self.coords["gp_basis_baseline"],
            "pollsters": self.coords["pollsters"] # Include if house effects influence results indirectly
        }

        # --- Prepare Data for pm.set_data ---
        # Filter the *original* OOS results dataset to the elections we're predicting
        results_subset = self.dataset.results_oos[
            self.dataset.results_oos['election_date'].isin(elections_to_predict_dt)
        ].copy() # Use copy to avoid SettingWithCopyWarning
        
        # Ensure results_subset is sorted according to elections_to_predict for alignment
        results_subset['election_date_str'] = results_subset['election_date'].astype(str)
        results_subset = results_subset.set_index('election_date_str').reindex(elections_to_predict).reset_index()

        if results_subset['sample_size'].isnull().any():
             missing_dates = results_subset.loc[results_subset['sample_size'].isnull(), 'election_date_str'].tolist()
             raise ValueError(f"Missing sample size data in results_oos for elections: {missing_dates}")

        # Prepare non_competing mask for the prediction subset
        # Use results_subset which is already filtered and ordered
        non_competing_subset_mask = results_subset[self.dataset.political_families].notna() & (results_subset[self.dataset.political_families] > 0)
        non_competing_subset_data = non_competing_subset_mask.astype(int).replace(0, -10).replace(1, 0).values

        # Data containers must match names in the original model's pm.Data calls
        prediction_data_dict = {
             # Data for 'results_N' (name from original model)
             "results_N": results_subset["sample_size"].to_numpy(), 
             
             # Data for 'non_competing_parties_results' (name from original model)
             # This variable affects latent_pop_t0 calculation for results
             "non_competing_parties_results": non_competing_subset_data,
             
             # Do NOT provide 'observed_results' - that corresponds to 'R', which we predict
        }
        
        # --- Set Data and Predict ---
        with self.model: # Use the original model context
            # Set the data aligned with the 'elections_observed' coordinate
            pm.set_data(prediction_data_dict, model=self.model, coords=coords_pred) 

            # Sample posterior predictive for 'R' (election results variable name in original model)
            pred_history = pm.sample_posterior_predictive(
                self.trace,
                var_names=["R"], # Predict the results variable
                return_inferencedata=True,
                predictions=True
            )

        # --- Add Observed Data for Comparison ---
        try:
            # Prepare observed results as xarray aligned with prediction coords
            observed_results_xr = results_subset.set_index('election_date_str')[self.dataset.political_families].to_xarray()
            # Rename the dimension to match the prediction coordinate name
            observed_results_xr = observed_results_xr.rename({'election_date_str': 'elections_observed'})
            pred_history.add_groups({"observed_data_history": observed_results_xr}, coords=coords_pred)
        except Exception as e:
            print(f"Warning: Could not add observed history data to trace: {e}")
        
        # Add posterior_predictive group if only predictions group exists
        if hasattr(pred_history, 'predictions') and not hasattr(pred_history, 'posterior_predictive'):
             pred_history.posterior_predictive = pred_history.predictions.copy()

        return pred_history 