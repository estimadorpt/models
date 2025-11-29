"""
Bayesian Presidential Election Model for Portugal.

This module implements a Gaussian Process-based Bayesian model for forecasting
Portuguese presidential elections. Unlike the parliamentary model which uses
historical party baselines, this model uses candidate-specific priors and
a single campaign dynamics GP.

Key features:
- Candidate-specific priors (based on party affiliation where applicable)
- Campaign dynamics GP for time-varying support
- House effects per pollster
- Probabilistic undecided voter allocation
- Dirichlet-Multinomial likelihood for poll observations
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.special import softmax

from src.data.presidential_dataset import PresidentialElectionDataset


class PresidentialElectionModel:
    """
    Bayesian model for presidential election forecasting.

    This model uses a single Gaussian Process over campaign time to capture
    dynamics in candidate support, combined with pollster-specific house effects
    and a probabilistic model for undecided voter allocation.

    Model structure:
        latent_support[t, c] = prior_mean[c] + campaign_gp[t, c] + house_effect[pollster, c]
        effective_support[c] = declared_support[c] + undecided * allocation[c]
        poll_probs = softmax(effective_support)
        poll_counts ~ DirichletMultinomial(n, concentration * poll_probs)

    Attributes:
        dataset: PresidentialElectionDataset containing the polling data
        model: PyMC model object (created by build_model)
        trace: Inference data from sampling
    """

    def __init__(
        self,
        dataset: PresidentialElectionDataset,
        campaign_gp_lengthscale: float = 21.0,
        campaign_gp_amplitude_scale: float = 0.15,
        house_effect_sd_scale: float = 0.05,
        gp_kernel: str = 'Matern52',
        hsgp_m: int = 30,
        hsgp_c: float = 1.5,
    ):
        """
        Initialize the presidential election model.

        Args:
            dataset: PresidentialElectionDataset with polling data
            campaign_gp_lengthscale: Prior mean for GP lengthscale (days)
            campaign_gp_amplitude_scale: Prior scale for GP amplitude
            house_effect_sd_scale: Prior scale for house effect SD
            gp_kernel: Kernel type for GP ('Matern52', 'Matern32', or 'ExpQuad')
            hsgp_m: Number of basis functions for HSGP approximation
            hsgp_c: Expansion factor for HSGP
        """
        self.dataset = dataset
        self.campaign_gp_lengthscale = campaign_gp_lengthscale
        self.campaign_gp_amplitude_scale = campaign_gp_amplitude_scale
        self.house_effect_sd_scale = house_effect_sd_scale
        self.gp_kernel = gp_kernel
        self.hsgp_m = hsgp_m
        self.hsgp_c = hsgp_c

        # Model components (set during build)
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.coords: Dict = {}
        self.data_containers: Dict = {}

        # Cached indices
        self.pollster_idx: Optional[np.ndarray] = None
        self.calendar_time_poll_idx: Optional[np.ndarray] = None
        self.calendar_time_numeric: Optional[np.ndarray] = None

    def _build_coords(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Build coordinates and index mappings for the PyMC model.

        Returns:
            Tuple of (pollster_idx, calendar_time_poll_idx, COORDS dict)
        """
        polls = self.dataset.polls_train
        candidates = self.dataset.candidates

        # Create calendar time coordinate from poll dates
        unique_dates = pd.to_datetime(polls['date']).unique()
        # Add election day if not present
        election_date = self.dataset.election_date_dt
        if election_date not in unique_dates:
            unique_dates = np.append(unique_dates, election_date)
        unique_dates = pd.to_datetime(np.sort(unique_dates))

        # Convert to numeric (days from first date)
        min_date = unique_dates.min()
        self.calendar_time_numeric = (unique_dates - min_date).days.values

        # Map poll dates to calendar time indices
        date_to_idx = {date: i for i, date in enumerate(unique_dates)}
        poll_dates = pd.to_datetime(polls['date'])
        self.calendar_time_poll_idx = poll_dates.map(date_to_idx).values.astype(int)

        # Factorize pollsters
        self.pollster_idx, pollster_names = polls['pollster'].factorize(sort=True)

        # Build coordinates dict
        COORDS = {
            'observations': polls.index,
            'candidates': candidates,
            'calendar_time': unique_dates.strftime('%Y-%m-%d'),
            'pollsters': pollster_names,
        }

        print("\n=== Model Coordinates ===")
        for key, value in COORDS.items():
            print(f"  {key}: {len(value)} elements")

        self.coords = COORDS
        return self.pollster_idx, self.calendar_time_poll_idx, COORDS

    def _build_data_containers(self) -> Dict[str, pm.Data]:
        """
        Build PyMC data containers for observed data.

        Returns:
            Dictionary of pm.Data objects
        """
        polls = self.dataset.polls_train
        candidates = self.dataset.candidates
        candidate_cols = self.dataset.get_candidate_columns()

        # Get observed poll counts
        observed_counts = polls[candidate_cols].to_numpy()
        observed_n = polls['sample_size'].to_numpy()

        # Get undecided proportions (from raw data before multinomial conversion)
        undecided_props = self.dataset.undecided_proportions

        data_containers = {
            'pollster_idx': pm.Data('pollster_idx', self.pollster_idx, dims='observations'),
            'calendar_time_poll_idx': pm.Data('calendar_time_poll_idx',
                                              self.calendar_time_poll_idx, dims='observations'),
            'observed_N': pm.Data('observed_N', observed_n, dims='observations'),
            'observed_counts': pm.Data('observed_counts', observed_counts,
                                       dims=('observations', 'candidates')),
            'undecided_proportions': pm.Data('undecided_proportions', undecided_props,
                                             dims='observations'),
        }

        self.data_containers = data_containers
        return data_containers

    def build_model(self) -> pm.Model:
        """
        Build the PyMC presidential election model.

        Returns:
            pm.Model: The constructed PyMC model
        """
        # Build coordinates and data containers
        self.pollster_idx, self.calendar_time_poll_idx, coords = self._build_coords()

        candidates = self.dataset.candidates
        n_candidates = len(candidates)
        prior_means = self.dataset.get_prior_means()
        prior_sds = self.dataset.get_prior_sds()

        with pm.Model(coords=coords) as model:
            data_containers = self._build_data_containers()

            # ============================================================
            #                    1. CANDIDATE PRIORS
            # ============================================================
            # Prior means for each candidate (in logit-like space for softmax)
            # Transform prior_means (which are on 0-1 scale) to log-odds scale
            # Using log(p/(1-p)) but adjusted for multi-candidate setting
            prior_logits = np.log(np.clip(prior_means, 0.01, 0.99))
            prior_logits = prior_logits - prior_logits.mean()  # Center

            candidate_baseline = pm.Normal(
                'candidate_baseline',
                mu=prior_logits,
                sigma=prior_sds * 5,  # Scale up for logit space
                dims='candidates'
            )

            # ============================================================
            #              2. CAMPAIGN DYNAMICS GP
            # ============================================================
            # Single GP capturing time-varying dynamics during campaign
            campaign_gp_lengthscale = pm.LogNormal(
                'campaign_gp_lengthscale',
                mu=np.log(self.campaign_gp_lengthscale),
                sigma=0.5
            )
            campaign_gp_amplitude = pm.HalfNormal(
                'campaign_gp_amplitude',
                sigma=self.campaign_gp_amplitude_scale
            )

            # Build covariance function
            if self.gp_kernel == 'Matern52':
                cov_func = campaign_gp_amplitude**2 * pm.gp.cov.Matern52(
                    input_dim=1, ls=campaign_gp_lengthscale
                )
            elif self.gp_kernel == 'Matern32':
                cov_func = campaign_gp_amplitude**2 * pm.gp.cov.Matern32(
                    input_dim=1, ls=campaign_gp_lengthscale
                )
            else:  # ExpQuad
                cov_func = campaign_gp_amplitude**2 * pm.gp.cov.ExpQuad(
                    input_dim=1, ls=campaign_gp_lengthscale
                )

            # Use HSGP approximation for efficiency
            campaign_gp = pm.gp.HSGP(
                cov_func=cov_func,
                m=[self.hsgp_m],
                c=self.hsgp_c
            )
            phi_campaign, sqrt_psd_campaign = campaign_gp.prior_linearized(
                X=self.calendar_time_numeric[:, None]
            )

            # Add GP basis coordinate
            model.add_coords({'gp_basis': np.arange(campaign_gp.n_basis_vectors)})

            # GP coefficients per candidate (zero-sum constraint)
            campaign_gp_coef_raw = pm.Normal(
                'campaign_gp_coef_raw',
                mu=0, sigma=1,
                dims=('gp_basis', 'candidates')
            )
            # Center coefficients (zero-sum across GP basis for identifiability)
            campaign_gp_coef = pm.Deterministic(
                'campaign_gp_coef',
                campaign_gp_coef_raw - campaign_gp_coef_raw.mean(axis=0, keepdims=True),
                dims=('gp_basis', 'candidates')
            )

            # GP effect over calendar time
            campaign_effect = pm.Deterministic(
                'campaign_effect',
                pt.dot(phi_campaign, campaign_gp_coef * sqrt_psd_campaign[:, None]),
                dims=('calendar_time', 'candidates')
            )

            # ============================================================
            #                   3. HOUSE EFFECTS
            # ============================================================
            house_effects_sd = pm.HalfNormal(
                'house_effects_sd',
                sigma=self.house_effect_sd_scale,
                dims='candidates'
            )
            # Zero-sum across pollsters for each candidate
            house_effects_raw = pm.ZeroSumNormal(
                'house_effects_raw',
                sigma=1.0,
                dims=('pollsters', 'candidates')
            )
            house_effects = pm.Deterministic(
                'house_effects',
                house_effects_raw * house_effects_sd[None, :],
                dims=('pollsters', 'candidates')
            )

            # ============================================================
            #            4. UNDECIDED VOTER ALLOCATION
            # ============================================================
            # Model how undecided voters allocate to candidates
            # Prior: equal allocation (uninformative)
            undecided_allocation = pm.Dirichlet(
                'undecided_allocation',
                a=np.ones(n_candidates),  # Equal prior weights
                dims='candidates'
            )

            # ============================================================
            #           5. LATENT SUPPORT TRAJECTORY
            # ============================================================
            # National latent support over calendar time
            latent_support_calendar = pm.Deterministic(
                'latent_support_calendar',
                candidate_baseline[None, :] + campaign_effect,
                dims=('calendar_time', 'candidates')
            )

            # Convert to probabilities via softmax
            national_probs_calendar = pm.Deterministic(
                'national_probs_calendar',
                pm.math.softmax(latent_support_calendar, axis=1),
                dims=('calendar_time', 'candidates')
            )

            # ============================================================
            #            6. POLL-LEVEL PREDICTIONS
            # ============================================================
            # Index latent support at poll observation times
            latent_at_polls = latent_support_calendar[data_containers['calendar_time_poll_idx'], :]

            # Add house effects
            house_at_polls = house_effects[data_containers['pollster_idx'], :]
            latent_polls = latent_at_polls + house_at_polls

            # Apply softmax to get declared support probabilities
            declared_probs = pm.math.softmax(latent_polls, axis=1)

            # Adjust for undecided allocation
            # effective_prob = (1 - undecided) * declared + undecided * allocation
            undecided_share = data_containers['undecided_proportions'][:, None]
            poll_probs = pm.Deterministic(
                'poll_probs',
                (1 - undecided_share) * declared_probs + undecided_share * undecided_allocation[None, :],
                dims=('observations', 'candidates')
            )

            # ============================================================
            #                    7. LIKELIHOOD
            # ============================================================
            # Concentration parameter for Dirichlet-Multinomial
            concentration = pm.Gamma('concentration', alpha=2, beta=0.01)

            pm.DirichletMultinomial(
                'poll_likelihood',
                n=data_containers['observed_N'],
                a=concentration * poll_probs,
                observed=data_containers['observed_counts'],
                dims=('observations', 'candidates')
            )

            # ============================================================
            #             8. DERIVED QUANTITIES
            # ============================================================
            # Win probability (who has >50% or highest share)
            # Get probabilities at election day
            election_day_idx = len(self.calendar_time_numeric) - 1
            election_day_probs = national_probs_calendar[election_day_idx, :]

            pm.Deterministic(
                'election_day_probs',
                election_day_probs,
                dims='candidates'
            )

            # Check model validity
            print("\n=== Checking Model at Initial Point ===")
            try:
                print(model.point_logps())
                print("Model check passed.")
            except Exception as e:
                print(f"WARNING: Model check failed: {e}")

        self.model = model
        return model

    def sample(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        **kwargs
    ) -> az.InferenceData:
        """
        Sample from the posterior distribution.

        Args:
            draws: Number of posterior samples per chain
            tune: Number of tuning/warmup samples
            chains: Number of MCMC chains
            target_accept: Target acceptance rate for NUTS
            **kwargs: Additional arguments to pm.sample

        Returns:
            az.InferenceData containing posterior samples
        """
        if self.model is None:
            self.build_model()

        with self.model:
            print(f"\nSampling {draws} draws with {tune} tuning steps...")
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                nuts_sampler='numpyro',
                return_inferencedata=True,
                **kwargs
            )

        # Check for sampling issues
        n_divergent = self.trace.sample_stats.diverging.sum().item()
        if n_divergent > 0:
            print(f"WARNING: {n_divergent} divergent transitions detected")

        return self.trace

    def get_forecast(
        self,
        target_date: Optional[str] = None,
        include_uncertainty: bool = True
    ) -> pd.DataFrame:
        """
        Get forecasted support for each candidate.

        Args:
            target_date: Date for forecast (default: election day)
            include_uncertainty: Include credible intervals

        Returns:
            DataFrame with mean, median, and credible intervals for each candidate
        """
        if self.trace is None:
            raise ValueError("Must run sample() before getting forecast")

        if target_date is None:
            # Use election day
            probs = self.trace.posterior['election_day_probs']
        else:
            # Find closest date in calendar_time
            target_dt = pd.to_datetime(target_date)
            calendar_dates = pd.to_datetime(self.coords['calendar_time'])
            idx = np.argmin(np.abs(calendar_dates - target_dt))
            probs = self.trace.posterior['national_probs_calendar'].isel(calendar_time=idx)

        candidates = self.dataset.candidates

        results = []
        for i, candidate in enumerate(candidates):
            candidate_probs = probs.isel(candidates=i).values.flatten()
            result = {
                'candidate': candidate,
                'mean': np.mean(candidate_probs),
                'median': np.median(candidate_probs),
            }
            if include_uncertainty:
                result['ci_lower'] = np.percentile(candidate_probs, 2.5)
                result['ci_upper'] = np.percentile(candidate_probs, 97.5)
                result['ci_10'] = np.percentile(candidate_probs, 10)
                result['ci_90'] = np.percentile(candidate_probs, 90)

            results.append(result)

        df = pd.DataFrame(results)
        df = df.sort_values('mean', ascending=False).reset_index(drop=True)
        return df

    def get_win_probabilities(self) -> pd.DataFrame:
        """
        Calculate probability of each candidate winning.

        A candidate wins if they either:
        1. Get >50% in first round, OR
        2. Have the highest vote share (simplified, ignores second round)

        Returns:
            DataFrame with win probability for each candidate
        """
        if self.trace is None:
            raise ValueError("Must run sample() before getting win probabilities")

        probs = self.trace.posterior['election_day_probs'].values
        n_samples = probs.shape[0] * probs.shape[1]
        probs_flat = probs.reshape(n_samples, -1)

        candidates = self.dataset.candidates

        # First round win (>50%)
        first_round_wins = (probs_flat > 0.5).sum(axis=0) / n_samples

        # Leading candidate (highest share)
        leading = (probs_flat == probs_flat.max(axis=1, keepdims=True)).sum(axis=0) / n_samples

        # Probability of second round (no one >50%)
        second_round_prob = (probs_flat.max(axis=1) < 0.5).sum() / n_samples

        results = []
        for i, candidate in enumerate(candidates):
            results.append({
                'candidate': candidate,
                'first_round_win_prob': first_round_wins[i],
                'leading_prob': leading[i],
                'mean_support': probs_flat[:, i].mean(),
            })

        df = pd.DataFrame(results)
        df['second_round_prob'] = second_round_prob
        df = df.sort_values('leading_prob', ascending=False).reset_index(drop=True)
        return df
