#!/usr/bin/env python
"""
Fixed House Effect Simulation: FiveThirtyEight-Style Approach

This script implements the expert-recommended methodology for handling house effects
when a single pollster dominates recent data.

PROBLEM:
- January 2026 polls are ALL from Pitagórica
- The model estimates house effects from the entire poll history
- Early disagreements (Oct-Nov) create a large Pitagórica house effect for Cotrim
- This house effect is then subtracted from ALL January polls, resisting genuine movement

SOLUTION (from FiveThirtyEight methodology):
- Estimate house effects only from periods with multiple pollsters (Oct-Dec)
- Fix those house effects when processing single-pollster periods (January)
- This separates "learning pollster bias" from "tracking the campaign"

SCENARIOS:
1. current: Baseline - estimate house effects from all data (broken behavior)
2. fixed_early: FiveThirtyEight-style - estimate HE from Oct-Dec, fix for full data
3. disabled: House effects shrunk to near-zero (maximum data-driven)

EXPECTED RESULTS:
- current: Cotrim ~16% (house effect absorbs January signal)
- fixed_early: Cotrim ~18-19% (January polls drive latent support, not HE)
- disabled: Cotrim ~17% (raw poll average)

Usage:
    pixi run python scripts/simulate_fixed_house_effects.py \
        --output-dir outputs/fixed_he_sim

References:
- FiveThirtyEight: https://fivethirtyeight.com/methodology/how-our-polling-averages-work/
- Gelman blog: https://statmodeling.stat.columbia.edu/2020/10/19/estimated-house-effects/
- Italian research: https://projecteuclid.org/journals/annals-of-applied-statistics/volume-16/issue-1/
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.presidential_dataset import PresidentialElectionDataset
from src.data.presidential_loaders import (
    load_parliamentary_house_effects,
    build_house_effect_prior_matrix,
)


# =============================================================================
# MODIFIED MODEL CLASS: Accepts Custom House Effect Priors
# =============================================================================

class PresidentialElectionModelWithCustomHE:
    """
    Presidential Election Model with support for CUSTOM house effect priors.

    This is a modified version of PresidentialElectionModel that allows:
    1. Standard behavior (parliamentary house effects)
    2. Custom house effect priors (from previous model run)
    3. Disabled house effects (shrunk to near-zero)

    Key addition:
        custom_house_effect_priors: Optional dict with 'means' and 'sds' arrays
                                    These OVERRIDE parliamentary priors if provided
    """

    def __init__(
        self,
        dataset: PresidentialElectionDataset,
        innovation_sd_scale: float = 0.05,
        concentration_base: float = 60.0,
        house_effect_sd_scale: float = 0.04,
        use_parliamentary_house_priors: bool = True,
        parliamentary_effects_path: Optional[str] = None,
        # NEW: Custom house effect priors
        custom_house_effect_priors: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize the model.

        Args:
            dataset: PresidentialElectionDataset with polling data
            innovation_sd_scale: Daily random walk innovation SD
            concentration_base: DirichletMultinomial concentration
            house_effect_sd_scale: Prior scale for house effect SD (used if no priors)
            use_parliamentary_house_priors: Whether to use parliamentary house effects
            parliamentary_effects_path: Path to parliamentary house effects JSON
            custom_house_effect_priors: Dict with 'means' and 'sds' arrays
                                        If provided, OVERRIDES parliamentary priors
        """
        self.dataset = dataset
        self.innovation_sd_scale = innovation_sd_scale
        self.concentration_base = concentration_base
        self.house_effect_sd_scale = house_effect_sd_scale
        self.use_parliamentary_house_priors = use_parliamentary_house_priors
        self.parliamentary_effects_path = parliamentary_effects_path
        self.custom_house_effect_priors = custom_house_effect_priors

        # Model components
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.coords: Dict = {}
        self.data_containers: Dict = {}

        # Cached indices
        self.pollster_idx: Optional[np.ndarray] = None
        self.calendar_time_poll_idx: Optional[np.ndarray] = None
        self.calendar_time_numeric: Optional[np.ndarray] = None

        # House effect priors
        self.house_effect_prior_means: Optional[np.ndarray] = None
        self.house_effect_prior_sds: Optional[np.ndarray] = None

    def _build_coords(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Build coordinates and index mappings for the PyMC model."""
        polls = self.dataset.polls_train
        candidates = self.dataset.candidates

        poll_dates_raw = pd.to_datetime(polls['date'])
        min_date = poll_dates_raw.min()
        election_date = self.dataset.election_date_dt

        calendar_dates = pd.date_range(start=min_date, end=election_date, freq='D')
        self.calendar_time_numeric = (calendar_dates - min_date).days.astype(float).values

        date_to_idx = {date: i for i, date in enumerate(calendar_dates)}
        self.calendar_time_poll_idx = poll_dates_raw.map(
            lambda d: date_to_idx.get(d, np.argmin(np.abs(calendar_dates - d)))
        ).values.astype(int)

        self.pollster_idx, pollster_names = polls['pollster'].factorize(sort=True)

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
        """Build PyMC data containers for observed data."""
        polls = self.dataset.polls_train
        candidate_cols = self.dataset.get_candidate_columns()

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

        House effect prior logic:
        1. If custom_house_effect_priors is provided → USE THOSE (highest priority)
        2. Elif use_parliamentary_house_priors=True → load from parliamentary data
        3. Else → use uninformed priors with house_effect_sd_scale
        """
        self.pollster_idx, self.calendar_time_poll_idx, coords = self._build_coords()

        candidates = self.dataset.candidates
        n_candidates = len(candidates)
        prior_means = self.dataset.get_prior_means()
        prior_sds = self.dataset.get_prior_sds()
        n_pollsters = len(coords['pollsters'])

        # ============================================================
        # HOUSE EFFECT PRIOR LOGIC (Modified for custom priors)
        # ============================================================
        use_informative_he = False

        # Priority 1: Custom house effect priors (from previous model run)
        if self.custom_house_effect_priors is not None:
            print("\n=== Using CUSTOM House Effect Priors ===")
            self.house_effect_prior_means = self.custom_house_effect_priors['means']
            self.house_effect_prior_sds = self.custom_house_effect_priors['sds']

            # Verify shapes
            expected_shape = (n_pollsters, n_candidates)
            if self.house_effect_prior_means.shape != expected_shape:
                raise ValueError(f"Custom HE means shape {self.house_effect_prior_means.shape} "
                               f"!= expected {expected_shape}")

            print(f"Custom HE priors shape: {self.house_effect_prior_means.shape}")
            print(f"Custom HE prior SDs range: [{self.house_effect_prior_sds.min():.4f}, "
                  f"{self.house_effect_prior_sds.max():.4f}]")
            use_informative_he = True

        # Priority 2: Parliamentary house effects
        elif self.use_parliamentary_house_priors:
            print("\n=== Loading Parliamentary House Effects ===")
            parliamentary_effects = load_parliamentary_house_effects(
                self.parliamentary_effects_path
            )

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
                use_informative_he = True
            else:
                print("Warning: No parliamentary effects loaded")

        # Build model
        with pm.Model(coords=coords) as model:
            data_containers = self._build_data_containers()

            # ============================================================
            #                    1. CANDIDATE PRIORS
            # ============================================================
            prior_logits = np.log(np.clip(prior_means, 0.01, 0.99))
            prior_logits = prior_logits - prior_logits.mean()

            prior_logit_sds = np.array([
                s / (p * (1 - p)) for p, s in zip(prior_means, prior_sds)
            ])
            prior_logit_sds = np.maximum(prior_logit_sds, 0.2)

            candidate_baseline = pm.Normal(
                'candidate_baseline',
                mu=prior_logits,
                sigma=prior_logit_sds,
                dims='candidates'
            )

            # ============================================================
            #              2. CAMPAIGN DYNAMICS (RANDOM WALK)
            # ============================================================
            n_days = len(self.calendar_time_numeric)
            innovation_sd = self.innovation_sd_scale

            innovations_raw = pm.ZeroSumNormal(
                'campaign_innovations_raw',
                sigma=1.0,
                shape=(n_days, n_candidates),
                dims=('calendar_time', 'candidates')
            )

            init_scale = 0.05
            scales = pt.concatenate([
                pt.full((1,), init_scale),
                pt.full((n_days - 1,), innovation_sd)
            ])
            scaled_innovations = innovations_raw * scales[:, None]

            campaign_effect = pm.Deterministic(
                'campaign_effect',
                pt.cumsum(scaled_innovations, axis=0),
                dims=('calendar_time', 'candidates')
            )

            # ============================================================
            #          3. HOUSE EFFECTS
            # ============================================================
            if use_informative_he and self.house_effect_prior_means is not None:
                print("Using INFORMATIVE house effect priors")

                house_effects_raw = pm.Normal(
                    'house_effects_raw',
                    mu=self.house_effect_prior_means,
                    sigma=self.house_effect_prior_sds,
                    dims=('pollsters', 'candidates')
                )

                pollster_means = pt.mean(house_effects_raw, axis=0)
                house_effects = pm.Deterministic(
                    'house_effects',
                    house_effects_raw - pollster_means[None, :],
                    dims=('pollsters', 'candidates')
                )
            else:
                print("Using UNINFORMED house effect priors")

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
            #           4. LATENT SUPPORT TRAJECTORY
            # ============================================================
            latent_support_calendar = pm.Deterministic(
                'latent_support_calendar',
                candidate_baseline[None, :] + campaign_effect,
                dims=('calendar_time', 'candidates')
            )

            national_probs_calendar = pm.Deterministic(
                'national_probs_calendar',
                pm.math.softmax(latent_support_calendar, axis=1),
                dims=('calendar_time', 'candidates')
            )

            # ============================================================
            #            5. POLL-LEVEL PREDICTIONS
            # ============================================================
            latent_at_polls = latent_support_calendar[data_containers['calendar_time_poll_idx'], :]
            house_at_polls = house_effects[data_containers['pollster_idx'], :]
            latent_polls = latent_at_polls + house_at_polls

            poll_probs = pm.Deterministic(
                'poll_probs',
                pm.math.softmax(latent_polls, axis=1),
                dims=('observations', 'candidates')
            )

            # ============================================================
            #                    6. LIKELIHOOD
            # ============================================================
            concentration = self.concentration_base

            pm.DirichletMultinomial(
                'poll_likelihood',
                n=data_containers['observed_N'],
                a=concentration * poll_probs,
                observed=data_containers['observed_counts'],
                dims=('observations', 'candidates')
            )

            # ============================================================
            #             7. DERIVED QUANTITIES
            # ============================================================
            election_day_idx = len(self.calendar_time_numeric) - 1
            election_day_probs = national_probs_calendar[election_day_idx, :]

            pm.Deterministic(
                'election_day_probs',
                election_day_probs,
                dims='candidates'
            )

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
        """Sample from the posterior distribution."""
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

        n_divergent = self.trace.sample_stats.diverging.sum().item()
        if n_divergent > 0:
            print(f"WARNING: {n_divergent} divergent transitions detected")

        return self.trace

    def get_forecast(self, target_date: Optional[str] = None) -> pd.DataFrame:
        """Get forecasted support for each candidate."""
        if self.trace is None:
            raise ValueError("Must run sample() before getting forecast")

        if target_date is None:
            probs = self.trace.posterior['election_day_probs']
        else:
            target_dt = pd.to_datetime(target_date)
            calendar_dates = pd.to_datetime(self.coords['calendar_time'])
            idx = np.argmin(np.abs(calendar_dates - target_dt))
            probs = self.trace.posterior['national_probs_calendar'].isel(calendar_time=idx)

        candidates = self.dataset.candidates

        results = []
        for i, candidate in enumerate(candidates):
            candidate_probs = probs.isel(candidates=i).values.flatten()
            results.append({
                'candidate': candidate,
                'mean': np.mean(candidate_probs),
                'median': np.median(candidate_probs),
                'ci_lower': np.percentile(candidate_probs, 2.5),
                'ci_upper': np.percentile(candidate_probs, 97.5),
                'ci_10': np.percentile(candidate_probs, 10),
                'ci_90': np.percentile(candidate_probs, 90),
            })

        df = pd.DataFrame(results)
        df = df.sort_values('mean', ascending=False).reset_index(drop=True)
        return df

    def get_house_effects(self) -> pd.DataFrame:
        """Extract house effect estimates for all pollster-candidate pairs."""
        if self.trace is None:
            raise ValueError("Must run sample() before getting house effects")

        house_effects = self.trace.posterior['house_effects']
        pollsters = list(self.coords['pollsters'])
        candidates = self.dataset.candidates

        results = []
        for p_idx, pollster in enumerate(pollsters):
            for c_idx, candidate in enumerate(candidates):
                effects = house_effects.isel(pollsters=p_idx, candidates=c_idx)
                mean_val = float(effects.mean())
                ci_lower = float(effects.quantile(0.05))
                ci_upper = float(effects.quantile(0.95))

                # Significance: CI doesn't include zero
                significant = (ci_lower > 0) or (ci_upper < 0)

                results.append({
                    'pollster': pollster,
                    'candidate': candidate,
                    'mean': mean_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'significant': significant,
                })

        return pd.DataFrame(results)


# =============================================================================
# SCENARIO IMPLEMENTATIONS
# =============================================================================

def run_current_scenario(
    polls_file: str,
    election_date: str,
    draws: int,
    tune: int,
) -> Dict[str, Any]:
    """
    Scenario 1: CURRENT (Baseline)

    Train model on all data with current settings.
    House effects estimated from entire poll history.

    This is our "broken" behavior where early disagreements
    affect late estimates.
    """
    print("\n" + "="*70)
    print("SCENARIO: CURRENT (Baseline)")
    print("House effects estimated from ALL data (Oct 2025 - Jan 2026)")
    print("="*70)

    dataset = PresidentialElectionDataset(
        election_date=election_date,
        polls_file=polls_file,
    )

    model = PresidentialElectionModelWithCustomHE(
        dataset=dataset,
        use_parliamentary_house_priors=True,
    )
    model.build_model()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.sample(draws=draws, tune=tune, chains=4)

    forecast_df = model.get_forecast()
    house_effects_df = model.get_house_effects()

    return {
        'scenario': 'current',
        'description': 'House effects from all data',
        'forecast': forecast_df,
        'house_effects': house_effects_df,
        'trace': model.trace,
        'coords': model.coords,
    }


def run_fixed_early_scenario(
    polls_file: str,
    election_date: str,
    draws: int,
    tune: int,
    early_cutoff: str = '2025-12-31',
) -> Dict[str, Any]:
    """
    Scenario 2: FIXED_EARLY (FiveThirtyEight-style)

    Two-stage estimation:
    1. Train on Oct-Dec polls only (where multiple pollsters overlap)
       → Extract house effect posterior means
    2. Use those as FIXED priors (very tight SDs) for full data run
       → January polls inform latent support, not house effects

    This simulates: "Estimate pollster bias from overlapping period,
    then treat recent single-pollster data as genuine movement."
    """
    print("\n" + "="*70)
    print("SCENARIO: FIXED_EARLY (FiveThirtyEight-style)")
    print(f"Stage 1: Estimate house effects from Oct-Dec only (cutoff: {early_cutoff})")
    print("Stage 2: Fix house effects and train on all data")
    print("="*70)

    # ============================================================
    # STAGE 1: Train on early data to estimate house effects
    # ============================================================
    print("\n--- STAGE 1: Training on Oct-Dec polls only ---")

    early_dataset = PresidentialElectionDataset(
        election_date=election_date,
        polls_file=polls_file,
        cutoff_date=early_cutoff,  # Only Oct-Dec polls
    )

    print(f"Early dataset: {len(early_dataset.polls_train)} polls")
    print(f"Pollsters in early data: {list(early_dataset.unique_pollsters)}")

    early_model = PresidentialElectionModelWithCustomHE(
        dataset=early_dataset,
        use_parliamentary_house_priors=True,
    )
    early_model.build_model()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        early_model.sample(draws=draws, tune=tune, chains=4)

    # Extract house effect posteriors
    early_he = early_model.trace.posterior['house_effects']
    early_he_means = early_he.mean(dim=['chain', 'draw']).values
    early_he_sds = early_he.std(dim=['chain', 'draw']).values

    print(f"\nExtracted house effects from early data:")
    print(f"  Shape: {early_he_means.shape}")
    print(f"  Mean range: [{early_he_means.min():.4f}, {early_he_means.max():.4f}]")

    # Find Pitagórica-Cotrim effect
    early_pollsters = list(early_model.coords['pollsters'])
    early_candidates = early_dataset.candidates

    pita_idx = None
    for i, p in enumerate(early_pollsters):
        if 'pitag' in p.lower():
            pita_idx = i
            break

    cotrim_idx = early_candidates.index('Cotrim Figueiredo') if 'Cotrim Figueiredo' in early_candidates else None

    if pita_idx is not None and cotrim_idx is not None:
        print(f"  Early Pitagórica-Cotrim effect: {early_he_means[pita_idx, cotrim_idx]:.4f}")

    # ============================================================
    # STAGE 2: Train on full data with FIXED house effects
    # ============================================================
    print("\n--- STAGE 2: Training on all data with FIXED house effects ---")

    full_dataset = PresidentialElectionDataset(
        election_date=election_date,
        polls_file=polls_file,
    )

    print(f"Full dataset: {len(full_dataset.polls_train)} polls")
    full_pollsters = list(full_dataset.unique_pollsters)

    # Build custom HE prior matrix
    # Need to map early pollsters to full pollsters (they should be same or subset)
    n_full_pollsters = len(full_pollsters)
    n_candidates = len(full_dataset.candidates)

    custom_he_means = np.zeros((n_full_pollsters, n_candidates))
    custom_he_sds = np.ones((n_full_pollsters, n_candidates)) * 0.005  # Very tight priors

    # Map early pollster effects to full pollster matrix
    for i, full_pollster in enumerate(full_pollsters):
        if full_pollster in early_pollsters:
            early_idx = early_pollsters.index(full_pollster)
            # Map candidates (should be same)
            for j, candidate in enumerate(full_dataset.candidates):
                if candidate in early_candidates:
                    early_c_idx = early_candidates.index(candidate)
                    custom_he_means[i, j] = early_he_means[early_idx, early_c_idx]
                    # Use posterior SD from early model, but with floor
                    custom_he_sds[i, j] = max(early_he_sds[early_idx, early_c_idx], 0.005)
        else:
            # New pollster not in early data: use zero with moderate uncertainty
            print(f"  Pollster '{full_pollster}' not in early data, using zero prior")
            custom_he_sds[i, :] = 0.02  # More uncertainty for new pollsters

    print(f"\nCustom HE prior matrix:")
    print(f"  Shape: {custom_he_means.shape}")
    print(f"  SD range: [{custom_he_sds.min():.4f}, {custom_he_sds.max():.4f}]")

    custom_he_priors = {
        'means': custom_he_means,
        'sds': custom_he_sds,
    }

    full_model = PresidentialElectionModelWithCustomHE(
        dataset=full_dataset,
        custom_house_effect_priors=custom_he_priors,  # Use extracted priors
    )
    full_model.build_model()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_model.sample(draws=draws, tune=tune, chains=4)

    forecast_df = full_model.get_forecast()
    house_effects_df = full_model.get_house_effects()

    # Also get early forecast for comparison
    early_forecast_df = early_model.get_forecast()

    return {
        'scenario': 'fixed_early',
        'description': 'House effects from Oct-Dec only, fixed for full data',
        'forecast': forecast_df,
        'house_effects': house_effects_df,
        'early_forecast': early_forecast_df,
        'early_he_means': early_he_means,
        'trace': full_model.trace,
        'coords': full_model.coords,
    }


def run_disabled_scenario(
    polls_file: str,
    election_date: str,
    draws: int,
    tune: int,
) -> Dict[str, Any]:
    """
    Scenario 3: DISABLED

    House effects shrunk to near-zero.
    Maximum data-driven approach.

    This represents: "What if we trusted the polls directly
    without adjusting for pollster bias?"
    """
    print("\n" + "="*70)
    print("SCENARIO: DISABLED")
    print("House effects shrunk to near-zero (SD=0.01)")
    print("="*70)

    dataset = PresidentialElectionDataset(
        election_date=election_date,
        polls_file=polls_file,
    )

    model = PresidentialElectionModelWithCustomHE(
        dataset=dataset,
        use_parliamentary_house_priors=False,
        house_effect_sd_scale=0.01,  # Very small
    )
    model.build_model()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.sample(draws=draws, tune=tune, chains=4)

    forecast_df = model.get_forecast()
    house_effects_df = model.get_house_effects()

    return {
        'scenario': 'disabled',
        'description': 'House effects disabled (SD=0.01)',
        'forecast': forecast_df,
        'house_effects': house_effects_df,
        'trace': model.trace,
        'coords': model.coords,
    }


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def print_comparison_table(all_results: List[Dict[str, Any]]):
    """Print comprehensive comparison table."""
    print("\n" + "="*90)
    print("FIXED HOUSE EFFECT SIMULATION: FORECAST COMPARISON")
    print("="*90)

    # Recent poll averages (from last 3 Pitagórica polls)
    poll_avg = {
        'Cotrim Figueiredo': 19.1,
        'Marques Mendes': 15.5,
        'António José Seguro': 20.5,
        'André Ventura': 20.3,
        'Gouveia e Melo': 17.2,
    }

    print(f"\nTarget (last 3 polls avg):")
    for candidate, avg in poll_avg.items():
        print(f"  {candidate}: {avg:.1f}%")

    # Main comparison table
    print(f"\n{'Scenario':<15} {'Cotrim':>10} {'Gap':>8} {'Mendes':>10} {'Gap':>8}")
    print("-"*55)

    for r in all_results:
        forecast = r['forecast']

        cotrim_row = forecast[forecast['candidate'] == 'Cotrim Figueiredo'].iloc[0]
        mendes_row = forecast[forecast['candidate'] == 'Marques Mendes'].iloc[0]

        cotrim_gap = cotrim_row['mean'] * 100 - poll_avg['Cotrim Figueiredo']
        mendes_gap = mendes_row['mean'] * 100 - poll_avg['Marques Mendes']

        print(f"{r['scenario']:<15} "
              f"{cotrim_row['mean']*100:>9.1f}% {cotrim_gap:>+7.1f}pp "
              f"{mendes_row['mean']*100:>9.1f}% {mendes_gap:>+7.1f}pp")

    # House effects comparison
    print("\n" + "="*90)
    print("PITAGÓRICA HOUSE EFFECT FOR COTRIM (Key Diagnostic)")
    print("="*90)

    print(f"\n{'Scenario':<15} {'Effect':>10} {'90% CI':>22} {'Significant?':>14}")
    print("-"*65)

    for r in all_results:
        he_df = r['house_effects']
        pita_cotrim = he_df[
            (he_df['pollster'].str.contains('itag', case=False)) &
            (he_df['candidate'] == 'Cotrim Figueiredo')
        ]

        if len(pita_cotrim) > 0:
            row = pita_cotrim.iloc[0]
            ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            sig = "Yes" if row['significant'] else "No"
            print(f"{r['scenario']:<15} {row['mean']:>9.3f} {ci:>22} {sig:>14}")
        else:
            print(f"{r['scenario']:<15} {'N/A':>10}")


def print_full_forecasts(all_results: List[Dict[str, Any]]):
    """Print full forecasts for all scenarios."""
    for r in all_results:
        print(f"\n{'='*60}")
        print(f"FULL FORECAST: {r['scenario'].upper()}")
        print(f"Description: {r['description']}")
        print(f"{'='*60}")

        forecast = r['forecast']
        print(f"\n{'Candidate':<25} {'Mean':>10} {'80% CI':>22}")
        print("-"*60)

        for _, row in forecast.iterrows():
            ci = f"[{row['ci_10']*100:.1f}%, {row['ci_90']*100:.1f}%]"
            print(f"{row['candidate']:<25} {row['mean']*100:>9.1f}% {ci:>22}")


def plot_comparison(all_results: List[Dict[str, Any]], output_path: str):
    """Create visualization comparing scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    scenarios = [r['scenario'] for r in all_results]
    x = np.arange(len(scenarios))
    bar_width = 0.35

    # Poll averages
    poll_avg = {
        'Cotrim Figueiredo': 19.1,
        'Marques Mendes': 15.5,
    }

    # Extract data
    cotrim_data = []
    mendes_data = []

    for r in all_results:
        forecast = r['forecast']
        cotrim_row = forecast[forecast['candidate'] == 'Cotrim Figueiredo'].iloc[0]
        mendes_row = forecast[forecast['candidate'] == 'Marques Mendes'].iloc[0]

        cotrim_data.append({
            'mean': cotrim_row['mean'] * 100,
            'ci_10': cotrim_row['ci_10'] * 100,
            'ci_90': cotrim_row['ci_90'] * 100,
        })
        mendes_data.append({
            'mean': mendes_row['mean'] * 100,
            'ci_10': mendes_row['ci_10'] * 100,
            'ci_90': mendes_row['ci_90'] * 100,
        })

    # Plot 1: Cotrim
    ax = axes[0]
    means = [d['mean'] for d in cotrim_data]
    lows = [d['ci_10'] for d in cotrim_data]
    highs = [d['ci_90'] for d in cotrim_data]

    bars = ax.bar(x, means, color='#00CED1', alpha=0.7, label='Model Forecast')
    ax.errorbar(x, means,
                yerr=[np.array(means) - np.array(lows),
                      np.array(highs) - np.array(means)],
                fmt='none', color='black', capsize=5)
    ax.axhline(poll_avg['Cotrim Figueiredo'], color='red', linestyle='--',
               linewidth=2, label=f"Poll Avg ({poll_avg['Cotrim Figueiredo']:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.set_ylabel('Support (%)')
    ax.set_title('Cotrim Figueiredo')
    ax.legend(loc='lower right')
    ax.set_ylim(10, 25)

    # Plot 2: Mendes
    ax = axes[1]
    means = [d['mean'] for d in mendes_data]
    lows = [d['ci_10'] for d in mendes_data]
    highs = [d['ci_90'] for d in mendes_data]

    ax.bar(x, means, color='#FF8C00', alpha=0.7, label='Model Forecast')
    ax.errorbar(x, means,
                yerr=[np.array(means) - np.array(lows),
                      np.array(highs) - np.array(means)],
                fmt='none', color='black', capsize=5)
    ax.axhline(poll_avg['Marques Mendes'], color='red', linestyle='--',
               linewidth=2, label=f"Poll Avg ({poll_avg['Marques Mendes']:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.set_ylabel('Support (%)')
    ax.set_title('Marques Mendes')
    ax.legend(loc='upper right')
    ax.set_ylim(10, 25)

    # Plot 3: Pitagórica House Effect for Cotrim
    ax = axes[2]
    he_means = []
    he_lows = []
    he_highs = []

    for r in all_results:
        he_df = r['house_effects']
        pita_cotrim = he_df[
            (he_df['pollster'].str.contains('itag', case=False)) &
            (he_df['candidate'] == 'Cotrim Figueiredo')
        ]
        if len(pita_cotrim) > 0:
            row = pita_cotrim.iloc[0]
            he_means.append(row['mean'])
            he_lows.append(row['ci_lower'])
            he_highs.append(row['ci_upper'])
        else:
            he_means.append(0)
            he_lows.append(0)
            he_highs.append(0)

    colors = ['#FF6B6B' if m > 0.02 else '#4ECDC4' for m in he_means]
    ax.bar(x, he_means, color=colors, alpha=0.7)
    ax.errorbar(x, he_means,
                yerr=[np.array(he_means) - np.array(he_lows),
                      np.array(he_highs) - np.array(he_means)],
                fmt='none', color='black', capsize=5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.set_ylabel('House Effect (log-odds)')
    ax.set_title('Pitagórica→Cotrim Effect')
    ax.set_ylim(-0.05, 0.15)

    plt.suptitle('Fixed House Effect Simulation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")
    plt.close()


def save_results(all_results: List[Dict[str, Any]], output_dir: str):
    """Save all results to files."""
    output_path = Path(output_dir)

    # Save comparison CSV
    comparison_data = []
    for r in all_results:
        forecast = r['forecast']

        row = {
            'scenario': r['scenario'],
            'description': r['description'],
        }

        for _, f_row in forecast.iterrows():
            candidate = f_row['candidate'].replace(' ', '_')
            row[f'{candidate}_mean'] = f_row['mean']
            row[f'{candidate}_ci_10'] = f_row['ci_10']
            row[f'{candidate}_ci_90'] = f_row['ci_90']

        # Add Pitagórica-Cotrim house effect
        he_df = r['house_effects']
        pita_cotrim = he_df[
            (he_df['pollster'].str.contains('itag', case=False)) &
            (he_df['candidate'] == 'Cotrim Figueiredo')
        ]
        if len(pita_cotrim) > 0:
            row['pita_cotrim_he'] = pita_cotrim.iloc[0]['mean']
            row['pita_cotrim_he_significant'] = pita_cotrim.iloc[0]['significant']

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    csv_path = output_path / 'fixed_he_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV to {csv_path}")

    # Save full results as JSON (excluding traces which are large)
    json_results = []
    for r in all_results:
        json_results.append({
            'scenario': r['scenario'],
            'description': r['description'],
            'forecast': r['forecast'].to_dict(orient='records'),
            'house_effects': r['house_effects'].to_dict(orient='records'),
        })

    json_path = output_path / 'fixed_he_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Saved full results to {json_path}")


def print_interpretation(all_results: List[Dict[str, Any]]):
    """Print interpretation of results."""
    print("\n" + "="*90)
    print("INTERPRETATION")
    print("="*90)

    poll_avg_cotrim = 19.1
    poll_avg_mendes = 15.5

    # Find best scenario
    best_scenario = None
    best_error = float('inf')

    for r in all_results:
        forecast = r['forecast']
        cotrim_row = forecast[forecast['candidate'] == 'Cotrim Figueiredo'].iloc[0]
        mendes_row = forecast[forecast['candidate'] == 'Marques Mendes'].iloc[0]

        cotrim_err = abs(cotrim_row['mean'] * 100 - poll_avg_cotrim)
        mendes_err = abs(mendes_row['mean'] * 100 - poll_avg_mendes)
        total_err = cotrim_err + mendes_err

        if total_err < best_error:
            best_error = total_err
            best_scenario = r

    print(f"\n1. BEST SCENARIO: {best_scenario['scenario']} (total error: {best_error:.1f}pp)")

    forecast = best_scenario['forecast']
    cotrim_row = forecast[forecast['candidate'] == 'Cotrim Figueiredo'].iloc[0]
    mendes_row = forecast[forecast['candidate'] == 'Marques Mendes'].iloc[0]

    print(f"   Cotrim: {cotrim_row['mean']*100:.1f}% (target: {poll_avg_cotrim:.1f}%, gap: {cotrim_row['mean']*100 - poll_avg_cotrim:+.1f}pp)")
    print(f"   Mendes: {mendes_row['mean']*100:.1f}% (target: {poll_avg_mendes:.1f}%, gap: {mendes_row['mean']*100 - poll_avg_mendes:+.1f}pp)")

    # Analyze house effects
    he_df = best_scenario['house_effects']
    pita_cotrim = he_df[
        (he_df['pollster'].str.contains('itag', case=False)) &
        (he_df['candidate'] == 'Cotrim Figueiredo')
    ]

    if len(pita_cotrim) > 0:
        row = pita_cotrim.iloc[0]
        print(f"\n2. PITAGÓRICA-COTRIM HOUSE EFFECT:")
        print(f"   Effect: {row['mean']:.4f} (90% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}])")
        print(f"   Significant: {'Yes' if row['significant'] else 'No'}")

        if row['significant'] and row['mean'] > 0.05:
            print(f"\n   WARNING: Large significant house effect remains.")
            print(f"   This suggests the model still thinks Pitagórica is biased,")
            print(f"   even with fixed early house effects.")
        elif not row['significant'] or row['mean'] < 0.03:
            print(f"\n   SUCCESS: House effect is small/non-significant.")
            print(f"   The model is treating January Pitagórica polls as genuine movement.")

    # Compare fixed_early to current
    print("\n3. SCENARIO COMPARISON:")

    for r in all_results:
        scenario = r['scenario']
        forecast = r['forecast']
        cotrim_row = forecast[forecast['candidate'] == 'Cotrim Figueiredo'].iloc[0]

        he_df = r['house_effects']
        pita_cotrim = he_df[
            (he_df['pollster'].str.contains('itag', case=False)) &
            (he_df['candidate'] == 'Cotrim Figueiredo')
        ]

        he_val = pita_cotrim.iloc[0]['mean'] if len(pita_cotrim) > 0 else 0

        print(f"   {scenario:<15} Cotrim: {cotrim_row['mean']*100:>5.1f}%  HE: {he_val:>+.3f}")

    # Recommendation
    print("\n4. RECOMMENDATION:")

    current_result = next((r for r in all_results if r['scenario'] == 'current'), None)
    fixed_early_result = next((r for r in all_results if r['scenario'] == 'fixed_early'), None)

    if fixed_early_result and current_result:
        fixed_forecast = fixed_early_result['forecast']
        current_forecast = current_result['forecast']

        fixed_cotrim = fixed_forecast[fixed_forecast['candidate'] == 'Cotrim Figueiredo'].iloc[0]['mean'] * 100
        current_cotrim = current_forecast[current_forecast['candidate'] == 'Cotrim Figueiredo'].iloc[0]['mean'] * 100

        improvement = fixed_cotrim - current_cotrim

        if improvement > 1.0:
            print(f"   The 'fixed_early' approach improves Cotrim estimate by {improvement:.1f}pp.")
            print(f"   RECOMMEND: Use 'fixed_early' configuration for final model run.")
        else:
            print(f"   The 'fixed_early' approach shows minimal improvement ({improvement:.1f}pp).")
            print(f"   Consider: 'disabled' may be more appropriate for final week of campaign.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fixed house effect simulation using FiveThirtyEight methodology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/fixed_he_sim",
        help="Output directory for results",
    )
    parser.add_argument(
        "--polls-file",
        default="presidenciais_polls_2026.parquet",
        help="Polls parquet file in data/",
    )
    parser.add_argument(
        "--election-date",
        default="2026-01-18",
        help="Election date",
    )
    parser.add_argument(
        "--early-cutoff",
        default="2025-12-31",
        help="Cutoff date for early (multi-pollster) period",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=500,
        help="Number of posterior samples per chain",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=500,
        help="Number of tuning steps",
    )
    parser.add_argument(
        "--scenarios",
        nargs='+',
        choices=['current', 'fixed_early', 'disabled'],
        default=None,
        help="Which scenarios to run (default: all)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("FIXED HOUSE EFFECT SIMULATION")
    print("FiveThirtyEight-style approach to single-pollster dominance")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Polls file: {args.polls_file}")
    print(f"Election date: {args.election_date}")
    print(f"Early cutoff: {args.early_cutoff}")
    print(f"Draws: {args.draws}, Tune: {args.tune}")

    scenarios_to_run = args.scenarios or ['current', 'fixed_early', 'disabled']
    print(f"Scenarios to run: {scenarios_to_run}")

    # Run scenarios
    all_results = []

    for scenario in scenarios_to_run:
        if scenario == 'current':
            result = run_current_scenario(
                polls_file=args.polls_file,
                election_date=args.election_date,
                draws=args.draws,
                tune=args.tune,
            )
        elif scenario == 'fixed_early':
            result = run_fixed_early_scenario(
                polls_file=args.polls_file,
                election_date=args.election_date,
                draws=args.draws,
                tune=args.tune,
                early_cutoff=args.early_cutoff,
            )
        elif scenario == 'disabled':
            result = run_disabled_scenario(
                polls_file=args.polls_file,
                election_date=args.election_date,
                draws=args.draws,
                tune=args.tune,
            )

        all_results.append(result)

    # Analysis
    print_comparison_table(all_results)
    print_full_forecasts(all_results)

    # Visualization
    plot_path = output_dir / 'fixed_he_comparison.png'
    plot_comparison(all_results, str(plot_path))

    # Save results
    save_results(all_results, str(output_dir))

    # Interpretation
    print_interpretation(all_results)

    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"Results saved to {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
