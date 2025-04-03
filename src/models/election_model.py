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


class ElectionModel:
    """A Bayesian election model"""

    def __init__(self, dataset: ElectionDataset):
        self.dataset = dataset
        
        self.gp_config = {
            "baseline_lengthscale": dataset.baseline_timescales,
            "election_lengthscale": dataset.election_timescales,            
            "kernel": "matern52",
            "zerosum": True,
            "variance_limit": 0.8,
        }

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
        
        # Use only historical elections for coords, which are the ones we have actual results for
        COORDS["elections"] = self.dataset.historical_election_dates
        print(f"Setting elections coord to historical elections: {COORDS['elections']}")
        
        # Get election indices for the data
        election_dates = data["election_date"].unique()
        election_id = np.zeros(len(data), dtype=int)
        for i, date in enumerate(data["election_date"]):
            # Find the index of this date in the historical elections
            for j, hist_date in enumerate(self.dataset.historical_election_dates):
                if pd.to_datetime(hist_date) == date:
                    election_id[i] = j
                    break
        
        # For elections_observed, use all historical elections 
        # (since now results_oos includes all historical elections)
        COORDS["elections_observed"] = COORDS["elections"]
        print(f"Setting elections_observed to all historical elections: {COORDS['elections_observed']}")

        return pollster_id, countdown_id, election_id, COORDS

    def _build_data_containers(self, polls: pd.DataFrame = None):
        """Build the data containers for the PyMC model
        
        Parameters
        ----------
        polls : pd.DataFrame, optional
            Poll data to use. If None, use the training data.
        """
        if polls is None:
            polls = self.dataset.polls_train
        is_here = polls[self.dataset.political_families].astype(bool).astype(int)
        
        # Create non_competing_parties_results based on the actual historical election results
        non_competing_parties_results = (
            self.dataset.results_mult[self.dataset.political_families]
            .astype(bool)
            .astype(int)
            .replace(to_replace=0, value=-10)
            .replace(to_replace=1, value=0)
            .to_numpy()
        )
        
        print(f"For inference: Using historical elections data with shape: {non_competing_parties_results.shape}")
        
        # Data containers for inference (historical elections)
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
                self.dataset.results_mult["sample_size"].to_numpy(),  # All historical elections
                dims="elections_observed"
            ),
            observed_results=pm.Data(
                "observed_results",
                self.dataset.results_mult[self.dataset.political_families].to_numpy(),  # All historical elections
                dims=("elections_observed", "parties_complete"),
            ),
            non_competing_parties_results=pm.Data(
                "non_competing_parties_results",
                non_competing_parties_results,
                dims=("elections", "parties_complete"),
            ),
            non_competing_polls_additive=pm.Data(
                "non_competing_polls_additive",
                is_here.replace(to_replace=0, value=-10).replace(to_replace=1, value=0).to_numpy(),
                dims=("observations", "parties_complete"),
            ),
            non_competing_polls_multiplicative=pm.Data(
                "non_competing_polls_multiplicative",
                is_here.to_numpy(),
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
            num_elections = len(self.dataset.historical_election_dates)
            print(f"Setting election dimensions to match historical elections: {num_elections}")

            # Election-specific party baseline. TIGHTER fixed prior.
            # election_party_baseline_sd = pm.HalfNormal("election_party_baseline_sd", sigma=0.3) # Removed hierarchical SD
            election_party_baseline = pm.ZeroSumNormal(
                "election_party_baseline",
                # sigma=election_party_baseline_sd, # Old
                sigma=0.15, # New: Fixed, smaller sigma
                dims=("elections", "parties_complete"),
                shape=(num_elections, len(self.dataset.political_families))
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
            #          HOUSE EFFECTS & POLL BIAS (Removed)
            # --------------------------------------------------------
            # Removed poll_bias
            # Removed house_effects
            # Removed house_election_effects

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

            # noisy_mu is now simpler without house effects/bias
            noisy_mu = pm.Deterministic(
                "noisy_mu",
                (
                    latent_mu
                    # Removed poll_bias
                    # Removed house_effects
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
            R = pm.DirichletMultinomial(
                "R",
                n=data_containers["results_N"],
                a=concentration_results * latent_pop_t0,  # Use all historical elections
                observed=data_containers["observed_results"],
                dims=("elections_observed", "parties_complete"),
            )

        return model

    def sample_all(
        self, *, model: pm.Model = None, var_names: List[str], **sampler_kwargs
    ):
        """
        Sample the model and return the trace.

        Parameters
        ----------
        model : optional
            A model previously created using `self.build_model()`.
            Build a new model if None (default)
        var_names: List[str]
            Variables names passed to `pm.fast_sample_posterior_predictive`
        **sampler_kwargs : dict
            Additional arguments to `pm.sample`
        """
        if model is None:
            model = self.build_model()

        # Set defaults for common parameters if not already specified
        sampler_kwargs.setdefault('nuts_sampler', 'numpyro')
        sampler_kwargs.setdefault('return_inferencedata', True)
        
        # Increase defaults for draws and tune if not specified
        sampler_kwargs.setdefault('draws', 3000)  # Increased from 2000
        sampler_kwargs.setdefault('tune', 3000)   # Increased from 2000
        
        # Add chain initialization strategy if not specified
        if 'init' not in sampler_kwargs:
            # Use jitter+adapt_diag for better initialization that's more robust
            sampler_kwargs['init'] = 'jitter+adapt_diag'
        
        # Set number of chains if not specified
        sampler_kwargs.setdefault('chains', 4)
        
        # Set max tree depth
        sampler_kwargs.setdefault('max_treedepth', 15)  # Increased from default 10
        
        # Recommend sampling parameters for better performance
        sampler_kwargs.setdefault('target_accept', 0.95)  # Maintains the already good target
        
        with model:
            prior_checks = pm.sample_prior_predictive()
            
            # Sample with improved diagnostics
            trace = pm.sample(**sampler_kwargs)
            
            # Store additional convergence diagnostics
            if trace is not None and isinstance(trace, arviz.InferenceData):
                # Check for divergences
                n_divergent = trace.sample_stats.diverging.sum().item()
                if n_divergent > 0:
                    print(f"WARNING: {n_divergent} divergent transitions detected")
                
                # Check if any parameters hit max tree depth frequently
                if 'tree_depth' in trace.sample_stats:
                    max_depths = (trace.sample_stats.tree_depth >= sampler_kwargs.get('max_treedepth', 10)).sum().item()
                    if max_depths > 0:
                        pct_max_depth = max_depths / (trace.posterior.dims['chain'] * trace.posterior.dims['draw'])
                        print(f"WARNING: {max_depths} samples ({pct_max_depth:.1%}) reached maximum tree depth")
            
            post_checks = pm.sample_posterior_predictive(
                trace, var_names=var_names
            )

        return prior_checks, trace, post_checks

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
        latest_election_date: str
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
        
        # Get reference to original model
        original_model = self.model
        
        # Map current polls to model indices
        pollster_indices = np.zeros(len(current_polls), dtype=int)
        
        # Get pollster list directly from the posterior to ensure exact match
        if hasattr(idata, 'posterior') and 'house_effects' in idata.posterior:
            pollsters_list = idata.posterior.coords['pollsters'].values
            print(f"\n=== Using pollster list DIRECTLY FROM POSTERIOR ===")
        else:
            pollsters_list = np.atleast_1d(original_model.coords["pollsters"])
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
            "elections": original_model.coords["elections"],
            "countdown": original_model.coords["countdown"],
        }
        
        # Debug: print house effects for current pollsters
        if hasattr(idata, 'posterior') and 'house_effects' in idata.posterior:
            print("\n=== DEBUG: HOUSE EFFECTS FOR CURRENT POLLSTERS ===")
            import matplotlib.pyplot as plt
            import os
            
            # Set the output directory
            self.model_output_dir = "outputs/latest/nowcast"
            
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
                countdown_vals = original_model.coords["countdown"]
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
            0, len(original_model.coords["countdown"]) - 1
        )
        
        # Match election dates to indices
        historical_dates = sorted([pd.to_datetime(date) for date in self.dataset.historical_election_dates])
        latest_historical_date = historical_dates[-1]
        latest_historical_date_str = latest_historical_date.strftime("%Y-%m-%d")
        
        election_indices = np.zeros(len(current_polls), dtype=int)
        elections_list = np.atleast_1d(original_model.coords["elections"])
        
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
            party_baseline_sd = pm.Flat("party_baseline_sd")
            
            # Time components
            party_time_effect_weighted = pm.Flat("party_time_effect_weighted", dims=("countdown", "parties_complete"))
            
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
            
            # Compute latent_mu using the same formula as the original model
            latent_mu = pm.Deterministic(
                "latent_mu",
                (
                    party_baseline[None, :]
                    + party_time_effect_weighted[countdown_idx]
                    + non_competing_add
                ),
                dims=("observations", "parties_complete")
            )
            
            # Apply softmax over parties for each observation
            latent_popularity = pm.Deterministic(
                "latent_popularity",
                pt.special.softmax(latent_mu, axis=1),
                dims=("observations", "parties_complete")
            )
            
            # Compute noisy_mu using the same formula as the original model
            noisy_mu = pm.Deterministic(
                "noisy_mu",
                (
                    latent_mu
                ) * non_competing_mult,
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
            
            # Sample with explicit specification of which variables to use from posterior
            print("\n=== SAMPLING FROM POSTERIOR PREDICTIVE ===")
            
            # Debug: Check what variables are in the posterior
            print("\n=== DEBUG: CHECKING POSTERIOR VARIABLES ===")
            if hasattr(idata, 'posterior'):
                print(f"Posterior variables available: {list(idata.posterior.data_vars.keys())}")
                
                # Print dimensions of house_effects in posterior
                if 'house_effects' in idata.posterior:
                    he_posterior = idata.posterior['house_effects']
                    print(f"\nhouse_effects in posterior:")
                    print(f"  shape: {he_posterior.shape}")
                    print(f"  dims: {he_posterior.dims}")
                    
                    # Print the actual pollster coordinate values from posterior
                    if 'pollsters' in he_posterior.coords:
                        print(f"  posterior pollsters dimension values: {he_posterior.coords['pollsters'].values}")
                        print(f"  prediction model pollsters: {prediction_coords['pollsters']}")
                        print(f"  identical pollster lists: {np.array_equal(he_posterior.coords['pollsters'].values, prediction_coords['pollsters'])}")
            else:
                print("No posterior group found in idata")
                
            # Debug: Check which variables could be found in posterior
            needed_vars = [
                "party_baseline",
                "party_time_effect_weighted",
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
                var_names=["latent_popularity", "noisy_popularity", "N_approve"],
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
            "N_approve": ["observations", "parties_complete"]
        }
        
        return pred_trace, prediction_coords, prediction_dims 