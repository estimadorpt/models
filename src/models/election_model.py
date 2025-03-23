import os
from typing import Dict, List, Tuple

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
            
        with pm.Model(coords=self.coords) as model:
            data_containers = self._build_data_containers(polls)

            # --------------------------------------------------------
            #                   BASELINE COMPONENTS
            # --------------------------------------------------------
            party_baseline_raw = pm.Normal(
                "party_baseline_raw",
                0.0,
                1.0,
                dims="parties_complete",
            )

            # More flexibility for party baseline standard deviation - further increased
            party_baseline_sd = pm.HalfNormal("party_baseline_sd", sigma=0.5)  # Increased from 0.4
            party_baseline = pm.ZeroSumNormal(
                "party_baseline", sigma=party_baseline_sd, dims="parties_complete"
            )

            # Count actual number of elections in the dataset for proper dimensioning
            num_elections = len(self.dataset.historical_election_dates)
            print(f"Setting election dimensions to match historical elections: {num_elections}")
            
            # Election-specific party baseline.
            # Represents election-specific deviation from party baseline
            election_party_baseline_raw = pm.Normal(
                "election_party_baseline_raw",
                0.0,
                1.0,
                dims=("elections", "parties_complete"),
            )

            # Increased flexibility for election-party effects
            election_party_baseline_sd = pm.HalfNormal("election_party_baseline_sd", sigma=0.3)  # Increased from 0.2
            election_party_baseline = pm.ZeroSumNormal(
                "election_party_baseline",
                sigma=election_party_baseline_sd,
                dims=("elections", "parties_complete"),
                shape=(num_elections, len(self.dataset.political_families))  # Explicitly set shape
            )

            # --------------------------------------------------------
            #               TIME-VARYING COMPONENTS
            # --------------------------------------------------------

            # Loop over baseline timescales and sum their contributions
            baseline_gp_contributions = []
            for i, baseline_lengthscale in enumerate(self.gp_config["baseline_lengthscale"]):
                cov_func_baseline = pm.gp.cov.Matern52(input_dim=1, ls=baseline_lengthscale)
                gp_baseline = pm.gp.HSGP(cov_func=cov_func_baseline, m=[20], c=2.0)
                phi_baseline, sqrt_psd_baseline = gp_baseline.prior_linearized(X=self.coords["countdown"][:, None])

                coord_name = f"gp_basis_baseline_{i}"
                if coord_name not in model.coords:
                    model.add_coords({coord_name: np.arange(gp_baseline.n_basis_vectors)})

                gp_coef_baseline = pm.Normal(
                    f"gp_coef_baseline_{baseline_lengthscale}",
                    mu=0,
                    sigma=1,
                    dims=(coord_name, "parties_complete")
                )

                baseline_contrib = pm.Deterministic(
                    f"party_time_effect_baseline_{baseline_lengthscale}",
                    pt.dot(phi_baseline, gp_coef_baseline * sqrt_psd_baseline[:, None]),
                    dims=("countdown", "parties_complete")
                )
                baseline_gp_contributions.append(baseline_contrib)

            # Sum the contributions from different baseline timescales
            party_time_effect = pm.Deterministic(
                "party_time_effect",
                sum(baseline_gp_contributions),
                dims=("countdown", "parties_complete")
            )

            # Weights for baseline component
            lsd_baseline = pm.Normal("lsd_baseline", mu=-1.8, sigma=0.4)  # Increased from 0.3
            lsd_party_effect = pm.ZeroSumNormal(
                "lsd_party_effect_party_amplitude",
                sigma=0.15,  # Increased from 0.1
                dims="parties_complete",
            )
            party_time_weight = pm.Deterministic(
                "party_time_weight",
                pt.exp(lsd_baseline + lsd_party_effect),
                dims="parties_complete"
            )
            party_time_effect_weighted = pm.Deterministic(
                "party_time_effect_weighted",
                party_time_effect * party_time_weight[None, :],
                dims=("countdown", "parties_complete")
            )

            # Loop over election timescales and sum their contributions
            election_gp_contributions = []
            for i, election_lengthscale in enumerate(self.gp_config["election_lengthscale"]):
                cov_func_election = pm.gp.cov.Matern52(input_dim=1, ls=election_lengthscale)
                gp_election = pm.gp.HSGP(cov_func=cov_func_election, m=[20], c=2.0)
                phi_election, sqrt_psd_election = gp_election.prior_linearized(X=self.coords["countdown"][:, None])

                coord_name = f"gp_basis_election_{i}"
                if coord_name not in model.coords:
                    model.add_coords({coord_name: np.arange(gp_election.n_basis_vectors)})

                gp_coef_election = pm.Normal(
                    f"gp_coef_election_{election_lengthscale}",
                    mu=0,
                    sigma=1,
                    dims=(coord_name, "parties_complete", "elections")
                )

                election_contrib = pm.Deterministic(
                    f"election_party_time_effect_{election_lengthscale}",
                    pt.tensordot(phi_election, gp_coef_election * sqrt_psd_election[:, None, None], axes=(1, 0)),
                    dims=("countdown", "parties_complete", "elections")
                )
                election_gp_contributions.append(election_contrib)

            # Sum the contributions from different election timescales
            election_party_time_effect = pm.Deterministic(
                "election_party_time_effect",
                sum(election_gp_contributions),
                dims=("countdown", "parties_complete", "elections")
            )

            # Weights for election-specific component
            lsd_party_effect_election = pm.ZeroSumNormal(
                "lsd_party_effect_election_party_amplitude",
                sigma=0.15,  # Increased from 0.1
                dims="parties_complete",
            )
            lsd_election_effect = pm.ZeroSumNormal(
                "lsd_election_effect",
                sigma=0.15,  # Increased from 0.1
                dims="elections"
            )
            lsd_election_party_sd = pm.HalfNormal(
                "lsd_election_party_sd", sigma=0.3  # Increased from 0.2
            )
            lsd_election_party_effect = pm.ZeroSumNormal(
                "lsd_election_party_effect",
                sigma=lsd_election_party_sd,
                dims=("parties_complete", "elections"),
                n_zerosum_axes=2
            )

            election_party_time_weight = pm.Deterministic(
                "election_party_time_weight",
                pt.exp(
                    lsd_party_effect_election[:, None]
                    + lsd_election_effect[None, :]
                    + lsd_election_party_effect
                ),
                dims=("parties_complete", "elections")
            )
            election_party_time_effect_weighted = pm.Deterministic(
                "election_party_time_effect_weighted",
                election_party_time_effect * election_party_time_weight[None, :, :],
                dims=("countdown", "parties_complete", "elections")
            )

            # --------------------------------------------------------
            #                        HOUSE EFFECTS & POLL BIAS
            # --------------------------------------------------------

            poll_bias = pm.ZeroSumNormal(
                "poll_bias",
                sigma=0.25,
                dims="parties_complete",
            )

            house_effects = pm.ZeroSumNormal(
                "house_effects",
                sigma=0.25,
                dims=("pollsters", "parties_complete"),
            )

            house_election_effects_sd = pm.HalfNormal(
                "house_election_effects_sd",
                sigma=0.15,  # Increased from 0.1
                dims=("pollsters", "parties_complete"),
            )
            house_election_effects_raw = pm.ZeroSumNormal(
                "house_election_effects_raw",
                dims=("pollsters", "parties_complete", "elections"),
            )
            house_election_effects = pm.Deterministic(
                "house_election_effects",
                house_election_effects_sd[..., None] * house_election_effects_raw,
                dims=("pollsters", "parties_complete", "elections"),
            )

            # --------------------------------------------------------
            #                      POLL RESULTS
            # --------------------------------------------------------

            # Compute latent_mu
            latent_mu = pm.Deterministic(
                "latent_mu",
                (
                    party_baseline[None, :]
                    + election_party_baseline[data_containers["election_idx"]]
                    + party_time_effect_weighted[data_containers["countdown_idx"]]
                    + election_party_time_effect_weighted[
                        data_containers["countdown_idx"], :, data_containers["election_idx"]
                    ]
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

            noisy_mu = pm.Deterministic(
                "noisy_mu",
                (
                    latent_mu
                    + poll_bias[None, :]
                    + house_effects[data_containers["pollster_idx"]]
                    + house_election_effects[
                        data_containers["pollster_idx"], :, data_containers["election_idx"]
                    ]
                ) * data_containers['non_competing_polls_multiplicative'],
                dims=("observations", "parties_complete")
            )

            # Apply softmax over parties for each observation
            noisy_popularity = pm.Deterministic(
                "noisy_popularity",
                pt.special.softmax(noisy_mu, axis=1),
                dims=("observations", "parties_complete"),
            )

            # The concentration parameter of a Dirichlet-Multinomial distribution
            # Parameterize log-normal to match desired mean and standard deviation
            desired_mean_polls = 250  # Your intuitive "sample size" interpretation
            desired_sd_polls = 50     # Uncertainty around this mean
            
            # Parameterize log-normal to match desired mean and standard deviation
            log_mean_polls = np.log(desired_mean_polls) - 0.5 * np.log(1 + (desired_sd_polls/desired_mean_polls)**2)
            log_sd_polls = np.sqrt(np.log(1 + (desired_sd_polls/desired_mean_polls)**2))
            
            # Define the log-normal prior
            log_concentration_polls = pm.Normal("log_concentration_polls", 
                                                mu=log_mean_polls, 
                                                sigma=log_sd_polls)
                                                
            # Transform back to natural scale
            concentration_polls = pm.Deterministic("concentration_polls", 
                                                   pt.exp(log_concentration_polls))

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

            # Compute latent_mu_t0
            latent_mu_t0 = pm.Deterministic(
                "latent_mu_t0",
                (
                    party_baseline[None, :]
                    + election_party_baseline
                    + party_time_effect_weighted[0]
                    + election_party_time_effect_weighted[0].transpose((1, 0))
                    + data_containers['non_competing_parties_results']
                ),
                dims=("elections", "parties_complete")
            )

            # Apply softmax over parties for each observation
            latent_pop_t0 = pm.Deterministic(
                "latent_pop_t0",
                pt.special.softmax(latent_mu_t0, axis=1),
                dims=("elections", "parties_complete"),
            )

            # Parameterize log-normal to match desired mean and standard deviation
            desired_mean_results = 2000  # Your intuitive "sample size" interpretation
            desired_sd_results = 200     # Uncertainty around this mean
            
            # Parameterize log-normal to match desired mean and standard deviation
            log_mean_results = np.log(desired_mean_results) - 0.5 * np.log(1 + (desired_sd_results/desired_mean_results)**2)
            log_sd_results = np.sqrt(np.log(1 + (desired_sd_results/desired_mean_results)**2))
            
            # Define the log-normal prior
            log_concentration_results = pm.Normal("log_concentration_results", 
                                                mu=log_mean_results, 
                                                sigma=log_sd_results)
                                                
            # Transform back to natural scale
            concentration_results = pm.Deterministic("concentration_results", 
                                                   pt.exp(log_concentration_results))

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
        sampler_kwargs.setdefault('target_accept', 0.9)  # Maintains the already good target
        
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
                
                # Check energy fraction - should be close to 0.5 for good mixing
                energy = trace.sample_stats.energy
                energy_frac = np.mean(np.abs(energy - np.mean(energy)) / np.std(energy) > 0.2)
                if energy_frac > 0.1:
                    print(f"WARNING: Energy fraction {energy_frac:.2f} indicates potential issues with energy conservation")
                
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