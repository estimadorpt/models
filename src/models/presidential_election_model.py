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

House Effect Handling:
    The model supports three approaches for house effect priors:

    1. Parliamentary priors (default): Use house effects from legislative elections
       as informative priors for presidential polling.

    2. Custom fixed priors: Use `custom_house_effect_priors` parameter to provide
       pre-estimated house effects. This is the FiveThirtyEight-style approach
       for campaigns where a single pollster dominates recent polling.

    3. Uninformed priors: Fall back to hierarchical priors centered at zero.

    For campaigns with single-pollster dominance in the late period, use the
    `estimate_fixed_house_effects()` helper function:

        from src.models.presidential_election_model import (
            PresidentialElectionModel,
            estimate_fixed_house_effects,
        )

        # Estimate house effects from early multi-pollster period
        fixed_he = estimate_fixed_house_effects(
            dataset=dataset,
            cutoff_date='2025-12-31',
        )

        # Train model with fixed house effects
        model = PresidentialElectionModel(
            dataset=dataset,
            custom_house_effect_priors=fixed_he,
        )
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
        innovation_sd_scale: float = 0.05,  # Fixed daily RW innovation SD (allows ~10pp movement over campaign)
        concentration_base: float = 60.0,  # DirichletMultinomial concentration (match poll measurement error)
        house_effect_sd_scale: float = 0.04,  # Allow ~4pp house effects
        use_parliamentary_house_priors: bool = True,
        parliamentary_effects_path: Optional[str] = None,
        custom_house_effect_priors: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize the presidential election model.

        Args:
            dataset: PresidentialElectionDataset with polling data
            innovation_sd_scale: Fixed daily random walk innovation SD in log-odds
                                 0.025/day → sqrt(50)*0.025 ≈ 0.18 logits ≈ 4-5pp over campaign
                                 Based on observed GM movement and Economist methodology
            concentration_base: Base DirichletMultinomial concentration (before undecided scaling)
                               Higher = tighter fit to polls = narrower CIs at poll dates
                               80 → concentration ~60 with 35% undecided
            house_effect_sd_scale: Prior scale for house effect SD
            use_parliamentary_house_priors: Whether to use parliamentary house effects as priors
            parliamentary_effects_path: Path to parliamentary house effects JSON
            custom_house_effect_priors: Optional dict with 'means' and 'sds' arrays for
                                        custom house effect priors. Shape must be
                                        (n_pollsters, n_candidates). If provided, takes
                                        priority over parliamentary priors. Use this for
                                        fixed house effects estimated from early data
                                        (FiveThirtyEight-style approach).
        """
        self.dataset = dataset
        self.innovation_sd_scale = innovation_sd_scale
        self.concentration_base = concentration_base
        self.house_effect_sd_scale = house_effect_sd_scale
        self.use_parliamentary_house_priors = use_parliamentary_house_priors
        self.parliamentary_effects_path = parliamentary_effects_path
        self.custom_house_effect_priors = custom_house_effect_priors

        # Model components (set during build)
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.coords: Dict = {}
        self.data_containers: Dict = {}

        # Cached indices
        self.pollster_idx: Optional[np.ndarray] = None
        self.calendar_time_poll_idx: Optional[np.ndarray] = None
        self.calendar_time_numeric: Optional[np.ndarray] = None

        # House effect priors (loaded if using parliamentary priors)
        self.house_effect_prior_means: Optional[np.ndarray] = None
        self.house_effect_prior_sds: Optional[np.ndarray] = None

    def _build_coords(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Build coordinates and index mappings for the PyMC model.

        Returns:
            Tuple of (pollster_idx, calendar_time_poll_idx, COORDS dict)
        """
        polls = self.dataset.polls_train
        candidates = self.dataset.candidates

        # Create DENSE calendar time grid from first poll to election day
        poll_dates_raw = pd.to_datetime(polls['date'])
        min_date = poll_dates_raw.min()
        election_date = self.dataset.election_date_dt
        
        # Create daily grid from first poll to election day
        calendar_dates = pd.date_range(start=min_date, end=election_date, freq='D')
        
        # Convert to numeric (days from first date)
        self.calendar_time_numeric = (calendar_dates - min_date).days.astype(float).values

        # Map poll dates to calendar time indices (find closest date in grid)
        date_to_idx = {date: i for i, date in enumerate(calendar_dates)}
        self.calendar_time_poll_idx = poll_dates_raw.map(
            lambda d: date_to_idx.get(d, np.argmin(np.abs(calendar_dates - d)))
        ).values.astype(int)

        # Factorize pollsters
        self.pollster_idx, pollster_names = polls['pollster'].factorize(sort=True)

        # Build coordinates dict with dense calendar grid
        COORDS = {
            'observations': polls.index,
            'candidates': candidates,
            'calendar_time': calendar_dates.strftime('%Y-%m-%d'),
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

        data_containers = {
            'pollster_idx': pm.Data('pollster_idx', self.pollster_idx, dims='observations'),
            'calendar_time_poll_idx': pm.Data('calendar_time_poll_idx',
                                              self.calendar_time_poll_idx, dims='observations'),
            'observed_N': pm.Data('observed_N', observed_n, dims='observations'),
            'observed_counts': pm.Data('observed_counts', observed_counts,
                                       dims=('observations', 'candidates')),
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

        # ============================================================
        # HOUSE EFFECT PRIORS: Priority order
        # 1. Custom priors (if provided) - for fixed early HE approach
        # 2. Parliamentary priors (if requested)
        # 3. Uninformed priors (fallback)
        # ============================================================
        n_pollsters = len(coords['pollsters'])

        if self.custom_house_effect_priors is not None:
            # Priority 1: Custom house effect priors (e.g., from early data estimation)
            print("\n=== Using Custom House Effect Priors ===")
            self.house_effect_prior_means = self.custom_house_effect_priors['means']
            self.house_effect_prior_sds = self.custom_house_effect_priors['sds']

            # Validate shapes
            expected_shape = (n_pollsters, n_candidates)
            if self.house_effect_prior_means.shape != expected_shape:
                raise ValueError(
                    f"Custom house effect means shape {self.house_effect_prior_means.shape} "
                    f"does not match expected {expected_shape}. "
                    f"Pollsters: {list(coords['pollsters'])}, Candidates: {candidates}"
                )

            print(f"Custom HE prior shape: {self.house_effect_prior_means.shape}")
            print(f"Custom HE prior SD range: [{self.house_effect_prior_sds.min():.4f}, "
                  f"{self.house_effect_prior_sds.max():.4f}]")
            # Force use of informative priors path
            self.use_parliamentary_house_priors = True

        elif self.use_parliamentary_house_priors:
            # Priority 2: Parliamentary house effects
            from src.data.presidential_loaders import (
                load_parliamentary_house_effects,
                build_house_effect_prior_matrix
            )

            print("\n=== Loading Parliamentary House Effects ===")
            parliamentary_effects = load_parliamentary_house_effects(self.parliamentary_effects_path)

            if parliamentary_effects:
                pollster_names = coords['pollsters']
                self.house_effect_prior_means, self.house_effect_prior_sds = \
                    build_house_effect_prior_matrix(
                        pollster_names,
                        candidates,
                        parliamentary_effects
                    )
                print(f"Loaded house effects for {len(parliamentary_effects)} pollsters")
                print(f"Prior matrix shape: {self.house_effect_prior_means.shape}")
            else:
                print("Warning: No parliamentary effects loaded, falling back to uninformed priors")
                self.use_parliamentary_house_priors = False

        with pm.Model(coords=coords) as model:
            data_containers = self._build_data_containers()

            # ============================================================
            #                    1. CANDIDATE PRIORS
            # ============================================================
            # Prior means for each candidate (in logit-like space for softmax)
            # Transform prior_means (which are on 0-1 scale) to log-odds scale
            prior_logits = np.log(np.clip(prior_means, 0.01, 0.99))
            prior_logits = prior_logits - prior_logits.mean()  # Center

            # Transform prior SDs to logit scale using delta method
            # For logit(p), derivative is 1/(p(1-p))
            # Delta method: SD_logit ≈ SD_p / (p * (1-p))
            prior_logit_sds = np.array([
                s / (p * (1 - p)) for p, s in zip(prior_means, prior_sds)
            ])
            prior_logit_sds = np.maximum(prior_logit_sds, 0.2)  # Floor to prevent numerical issues

            candidate_baseline = pm.Normal(
                'candidate_baseline',
                mu=prior_logits,
                sigma=prior_logit_sds,
                dims='candidates'
            )

            # ============================================================
            #              2. CAMPAIGN DYNAMICS (RANDOM WALK)
            # ============================================================
            # Random walk in log-odds space - tight at polls, grows between
            # Innovation SD is FIXED (not learned) following Economist/538 methodology:
            # - With only 7 polls, cannot reliably estimate volatility
            # - Based on observed GM movement: ~5pp over 50 days
            # - Daily volatility: 5pp / sqrt(50) ≈ 0.7pp/day ≈ 0.025 log-odds/day

            n_days = len(self.calendar_time_numeric)

            # Fixed innovation SD based on observed campaign dynamics
            # 0.025/day → sqrt(50)*0.025 ≈ 0.18 log-odds ≈ 4-5pp over campaign
            # This allows real trends while keeping CIs tight (~8-10pp)
            innovation_sd = self.innovation_sd_scale  # Fixed, not sampled

            # ZeroSumNormal for innovations: automatically enforces sum-to-zero
            # Non-centered parameterization for better MCMC mixing
            # This is the cleanest approach for identifiable compositional dynamics

            # Sample standardized innovations with built-in zero-sum constraint
            innovations_raw = pm.ZeroSumNormal(
                'campaign_innovations_raw',
                sigma=1.0,  # Standardized, we scale below
                shape=(n_days, n_candidates),
                dims=('calendar_time', 'candidates')
            )

            # Scale innovations: first row is init (tighter), rest are daily innovations
            init_scale = 0.05
            scales = pt.concatenate([
                pt.full((1,), init_scale),
                pt.full((n_days - 1,), innovation_sd)
            ])
            scaled_innovations = innovations_raw * scales[:, None]

            # Cumulative sum to get random walks (zero-sum preserved at each time point)
            campaign_effect = pm.Deterministic(
                'campaign_effect',
                pt.cumsum(scaled_innovations, axis=0),
                dims=('calendar_time', 'candidates')
            )

            # ============================================================
            #          3. HOUSE EFFECTS (WITH PARLIAMENTARY PRIORS)
            # ============================================================
            if self.use_parliamentary_house_priors and self.house_effect_prior_means is not None:
                # Use parliamentary house effects as informative priors
                print("Using parliamentary house effects as informative priors")

                # Informative priors centered on parliamentary effects
                house_effects_raw = pm.Normal(
                    'house_effects_raw',
                    mu=self.house_effect_prior_means,
                    sigma=self.house_effect_prior_sds,
                    dims=('pollsters', 'candidates')
                )

                # Apply zero-sum constraint for identifiability using pytensor
                pollster_means = pt.mean(house_effects_raw, axis=0)
                house_effects = pm.Deterministic(
                    'house_effects',
                    house_effects_raw - pollster_means[None, :],
                    dims=('pollsters', 'candidates')
                )
            else:
                # Fallback to uninformed priors (original implementation)
                print("Using uninformed house effect priors")

                house_effects_sd = pm.HalfNormal(
                    'house_effects_sd',
                    sigma=self.house_effect_sd_scale,
                    dims='candidates'
                )
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
            #            4. UNDECIDED VOTER ALLOCATION (REMOVED)
            # ============================================================
            # NOTE: We previously modeled undecided allocation but removed it because:
            # - We have no data on how undecideds actually break
            # - It was an unidentified parameter absorbing all residuals
            # - It was masking house effects and prior misspecification
            #
            # The model now forecasts DECLARED voting intention only.
            # Undecided voters are a separate source of uncertainty not modeled here.

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

            # Apply softmax to get poll probabilities (declared voting intention)
            poll_probs = pm.Deterministic(
                'poll_probs',
                pm.math.softmax(latent_polls, axis=1),
                dims=('observations', 'candidates')
            )

            # ============================================================
            #                    7. LIKELIHOOD
            # ============================================================
            # Concentration parameter for Dirichlet-Multinomial
            # Higher concentration = tighter fit to polls = narrower CIs at poll dates
            # Lower concentration = looser fit = wider CIs but more smoothing
            #
            # Concentration is structural - should match poll measurement error
            # With n=600, p=0.2: poll SD ≈ 1.64pp, concentration=60 gives posterior SD ≈ 1.7pp
            concentration = self.concentration_base

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


def create_shrunk_house_effect_priors(
    dataset: PresidentialElectionDataset,
    prior_sd: float = 0.02,
) -> Dict[str, np.ndarray]:
    """
    Create zero-centered house effect priors with specified SD.

    This is the simplest approach for handling single-pollster dominance:
    shrink house effects toward zero rather than estimating them. When
    only one pollster is providing recent data, we cannot distinguish
    pollster bias from genuine movement, so we assume minimal bias.

    Usage:
        # Create shrunk priors (house effects ~0 with SD=2%)
        shrunk_priors = create_shrunk_house_effect_priors(
            dataset=dataset,
            prior_sd=0.02,  # Allow up to ~4pp house effect (2 SD)
        )

        # Train model with shrunk house effects
        model = PresidentialElectionModel(
            dataset=dataset,
            custom_house_effect_priors=shrunk_priors,
        )

    Args:
        dataset: PresidentialElectionDataset with polling data
        prior_sd: SD for house effect priors. Default 0.02 (2%) allows
                  modest house effects while shrinking toward zero.

    Returns:
        Dict with 'means' and 'sds' arrays suitable for custom_house_effect_priors.
    """
    # Get pollster count from model coords
    temp_model = PresidentialElectionModel(dataset=dataset)
    temp_model._build_coords()
    n_pollsters = len(temp_model.coords['pollsters'])
    n_candidates = len(dataset.candidates)

    # Zero means, uniform SD
    means = np.zeros((n_pollsters, n_candidates))
    sds = np.full((n_pollsters, n_candidates), prior_sd)

    print(f"\n=== Creating Shrunk House Effect Priors ===")
    print(f"Shape: {means.shape}")
    print(f"Prior SD: {prior_sd*100:.1f}pp (shrinking toward zero)")

    return {'means': means, 'sds': sds}


def estimate_fixed_house_effects(
    dataset: PresidentialElectionDataset,
    cutoff_date: str,
    fixed_sd: float = 0.005,
    draws: int = 500,
    tune: int = 500,
    **model_kwargs
) -> Dict[str, np.ndarray]:
    """
    Estimate house effects from early data to use as fixed priors.

    NOTE: This approach may not work as expected when pollsters don't overlap
    in time. If early pollsters don't poll at the same dates, the model will
    estimate house effects from temporal differences rather than true bias.

    For single-pollster dominance, consider using `create_shrunk_house_effect_priors()`
    instead, which simply shrinks house effects toward zero.

    Usage:
        # Step 1: Estimate house effects from early overlapping period
        fixed_he = estimate_fixed_house_effects(
            dataset,
            cutoff_date='2025-12-31',  # Before single-pollster period
        )

        # Step 2: Train full model with fixed house effects
        model = PresidentialElectionModel(
            dataset=dataset,
            custom_house_effect_priors=fixed_he,
        )
        model.build_model()
        model.sample()

    Args:
        dataset: PresidentialElectionDataset with full polling data
        cutoff_date: Date string (YYYY-MM-DD) to use as cutoff. Only polls
                     before this date are used to estimate house effects.
        fixed_sd: Floor SD for fixed priors (default 0.005).
        draws: Posterior samples for house effect estimation
        tune: Tuning samples for house effect estimation
        **model_kwargs: Additional arguments passed to PresidentialElectionModel

    Returns:
        Dict with 'means' and 'sds' arrays suitable for custom_house_effect_priors.
        Shape is (n_pollsters, n_candidates) where pollsters are those in the
        FULL dataset (not just the early subset).
    """
    import warnings

    # Create early-only dataset
    cutoff_dt = pd.to_datetime(cutoff_date)
    all_polls = dataset.polls_train.copy()
    early_polls = all_polls[pd.to_datetime(all_polls['date']) < cutoff_dt]

    if len(early_polls) < 3:
        raise ValueError(
            f"Only {len(early_polls)} polls before {cutoff_date}. "
            "Need at least 3 polls to estimate house effects."
        )

    print(f"\n{'='*60}")
    print(f"ESTIMATING HOUSE EFFECTS FROM EARLY DATA")
    print(f"{'='*60}")
    print(f"Cutoff date: {cutoff_date}")
    print(f"Early polls: {len(early_polls)} (of {len(all_polls)} total)")
    print(f"Pollsters in early data: {early_polls['pollster'].nunique()}")

    # Create a temporary dataset with only early polls
    # We need to copy the original dataset and replace the polls
    import copy
    early_dataset = copy.copy(dataset)
    early_dataset.polls_train = early_polls.reset_index(drop=True)
    early_dataset.polls_raw = early_polls.reset_index(drop=True)

    # Train model on early data
    early_model = PresidentialElectionModel(
        dataset=early_dataset,
        use_parliamentary_house_priors=model_kwargs.get('use_parliamentary_house_priors', True),
        house_effect_sd_scale=model_kwargs.get('house_effect_sd_scale', 0.04),
        innovation_sd_scale=model_kwargs.get('innovation_sd_scale', 0.05),
    )
    early_model.build_model()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        early_model.sample(draws=draws, tune=tune, chains=4)

    # Extract house effect posteriors from early model
    early_he = early_model.trace.posterior['house_effects']
    early_he_means = early_he.mean(dim=['chain', 'draw']).values
    early_he_sds = early_he.std(dim=['chain', 'draw']).values
    early_pollsters = list(early_model.coords['pollsters'])
    early_candidates = early_dataset.candidates

    print(f"\nHouse effects estimated from early data:")
    for i, pollster in enumerate(early_pollsters):
        effects = early_he_means[i, :]
        print(f"  {pollster}: mean effect range [{effects.min():.3f}, {effects.max():.3f}]")

    # Map to full dataset pollsters
    # Get pollsters from full dataset by building coords
    full_model_temp = PresidentialElectionModel(dataset=dataset)
    full_model_temp._build_coords()
    full_pollsters = list(full_model_temp.coords['pollsters'])
    n_candidates = len(dataset.candidates)

    print(f"\nMapping to full dataset pollsters ({len(full_pollsters)} total):")

    # Build prior matrices for full pollster set
    # Use posterior SDs from early model (with floor) to allow updating based on new data
    full_means = np.zeros((len(full_pollsters), n_candidates))
    full_sds = np.full((len(full_pollsters), n_candidates), fixed_sd)

    for i, pollster in enumerate(full_pollsters):
        if pollster in early_pollsters:
            early_idx = early_pollsters.index(pollster)
            # Map candidates (handle case where order might differ)
            for j, candidate in enumerate(dataset.candidates):
                if candidate in early_candidates:
                    early_c_idx = early_candidates.index(candidate)
                    full_means[i, j] = early_he_means[early_idx, early_c_idx]
                    # Use posterior SD from early model, with floor
                    # This allows the model to update based on new data
                    full_sds[i, j] = max(early_he_sds[early_idx, early_c_idx], fixed_sd)
            print(f"  {pollster}: using estimated effect (SD range [{full_sds[i,:].min():.3f}, {full_sds[i,:].max():.3f}])")
        else:
            # Pollster not in early data - use zero prior with moderate uncertainty
            full_means[i, :] = 0.0
            full_sds[i, :] = 0.02  # ~2pp uncertainty for new pollsters
            print(f"  {pollster}: not in early data, using zero prior")

    print(f"\nFixed house effect priors ready:")
    print(f"  Shape: {full_means.shape}")
    print(f"  Fixed SD: {fixed_sd} (very tight)")

    return {
        'means': full_means,
        'sds': full_sds,
    }
