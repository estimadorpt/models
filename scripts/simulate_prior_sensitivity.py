#!/usr/bin/env python
"""
Prior Sensitivity Analysis for Presidential Model

This script tests different prior configurations to determine what prior values
would allow the model to respond appropriately to poll data.

The current priors for Cotrim Figueiredo (8% mean, 3% SD) are too restrictive
given that polls show him at 18-21%. This analysis tests various prior
configurations to find the minimum change needed.

DIAGNOSTIC ONLY - This script does NOT permanently modify the model.
It temporarily overrides priors for each scenario.

Usage:
    pixi run python scripts/simulate_prior_sensitivity.py \
        --output-dir outputs/prior_sensitivity_sim
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import warnings
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.presidential_dataset import PresidentialElectionDataset
from src.data.presidential_loaders import (
    load_presidential_polls,
    DEFAULT_CANDIDATES_2026,
    CANDIDATE_PARTY_PRIORS,
)
from src.models.presidential_election_model import PresidentialElectionModel


# =============================================================================
# PRIOR SCENARIOS
# =============================================================================

# Original priors for reference
ORIGINAL_PRIORS = copy.deepcopy(CANDIDATE_PARTY_PRIORS)

# Define prior scenarios to test
PRIOR_SCENARIOS = {
    'current': {
        # Use original priors unchanged
        'description': 'Current priors (8% Cotrim, 20% Mendes)',
        'overrides': {},
    },
    'weak_sd': {
        # Keep means but widen SDs
        'description': 'Wider SDs only (8% Cotrim SD→8%, 20% Mendes SD→8%)',
        'overrides': {
            'Cotrim Figueiredo': {'prior_mean': 0.08, 'prior_sd': 0.08},
            'Marques Mendes': {'prior_mean': 0.20, 'prior_sd': 0.08},
        },
    },
    'poll_avg': {
        # Use overall poll averages as priors
        'description': 'Poll average priors (12% Cotrim, 17% Mendes)',
        'overrides': {
            'Cotrim Figueiredo': {'prior_mean': 0.12, 'prior_sd': 0.05},
            'Marques Mendes': {'prior_mean': 0.17, 'prior_sd': 0.05},
        },
    },
    'recent_polls': {
        # Use mid-campaign averages
        'description': 'Recent poll priors (15% Cotrim, 16% Mendes)',
        'overrides': {
            'Cotrim Figueiredo': {'prior_mean': 0.15, 'prior_sd': 0.05},
            'Marques Mendes': {'prior_mean': 0.16, 'prior_sd': 0.05},
        },
    },
    'flat_prior': {
        # Very weak priors - let data dominate
        'description': 'Flat priors (15% both, SD=10%)',
        'overrides': {
            'Cotrim Figueiredo': {'prior_mean': 0.15, 'prior_sd': 0.10},
            'Marques Mendes': {'prior_mean': 0.15, 'prior_sd': 0.10},
            'Gouveia e Melo': {'prior_mean': 0.15, 'prior_sd': 0.10},
            'António José Seguro': {'prior_mean': 0.15, 'prior_sd': 0.10},
            'André Ventura': {'prior_mean': 0.15, 'prior_sd': 0.10},
        },
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def apply_prior_overrides(
    base_priors: Dict[str, Dict],
    overrides: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Apply prior overrides to base prior configuration.
    """
    result = copy.deepcopy(base_priors)
    for candidate, new_values in overrides.items():
        if candidate in result:
            result[candidate].update(new_values)
        else:
            result[candidate] = new_values
    return result


def train_with_priors(
    scenario_name: str,
    prior_overrides: Dict[str, Dict],
    polls_file: str,
    election_date: str,
    draws: int = 500,
    tune: int = 500,
) -> Dict[str, Any]:
    """
    Train a model with specific prior overrides and extract key results.
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    # Create dataset
    dataset = PresidentialElectionDataset(
        election_date=election_date,
        polls_file=polls_file,
    )

    # Apply prior overrides to the dataset
    modified_priors = apply_prior_overrides(ORIGINAL_PRIORS, prior_overrides)
    dataset.candidate_priors = modified_priors

    # Print prior values for Cotrim and Mendes
    cotrim_prior = modified_priors.get('Cotrim Figueiredo', {})
    mendes_prior = modified_priors.get('Marques Mendes', {})
    print(f"Cotrim prior: {cotrim_prior.get('prior_mean', 0)*100:.1f}% (SD {cotrim_prior.get('prior_sd', 0)*100:.1f}%)")
    print(f"Mendes prior: {mendes_prior.get('prior_mean', 0)*100:.1f}% (SD {mendes_prior.get('prior_sd', 0)*100:.1f}%)")

    # Build and train model
    model = PresidentialElectionModel(dataset=dataset)
    model.build_model()

    print(f"\nSampling {draws} draws with {tune} tuning steps...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.sample(draws=draws, tune=tune, chains=4)

    # Extract results
    forecast_df = model.get_forecast()

    # Extract house effects for Pitagórica -> Cotrim
    house_effects = model.trace.posterior['house_effects']
    pollsters = list(model.coords['pollsters'])
    candidates = model.dataset.candidates

    # Find Pitagórica index
    pita_idx = None
    for i, p in enumerate(pollsters):
        if 'pitag' in p.lower():
            pita_idx = i
            break

    cotrim_idx = candidates.index('Cotrim Figueiredo') if 'Cotrim Figueiredo' in candidates else None
    mendes_idx = candidates.index('Marques Mendes') if 'Marques Mendes' in candidates else None

    pita_cotrim_effect = None
    pita_mendes_effect = None
    if pita_idx is not None and cotrim_idx is not None:
        effects = house_effects.isel(pollsters=pita_idx, candidates=cotrim_idx)
        pita_cotrim_effect = {
            'mean': float(effects.mean()),
            'ci_lower': float(effects.quantile(0.05)),
            'ci_upper': float(effects.quantile(0.95)),
        }
    if pita_idx is not None and mendes_idx is not None:
        effects = house_effects.isel(pollsters=pita_idx, candidates=mendes_idx)
        pita_mendes_effect = {
            'mean': float(effects.mean()),
            'ci_lower': float(effects.quantile(0.05)),
            'ci_upper': float(effects.quantile(0.95)),
        }

    # Get Cotrim and Mendes forecasts
    cotrim_row = forecast_df[forecast_df['candidate'] == 'Cotrim Figueiredo'].iloc[0]
    mendes_row = forecast_df[forecast_df['candidate'] == 'Marques Mendes'].iloc[0]

    results = {
        'scenario': scenario_name,
        'cotrim_prior_mean': cotrim_prior.get('prior_mean', 0),
        'cotrim_prior_sd': cotrim_prior.get('prior_sd', 0),
        'mendes_prior_mean': mendes_prior.get('prior_mean', 0),
        'mendes_prior_sd': mendes_prior.get('prior_sd', 0),
        'cotrim_mean': float(cotrim_row['mean']),
        'cotrim_ci_10': float(cotrim_row['ci_10']),
        'cotrim_ci_90': float(cotrim_row['ci_90']),
        'mendes_mean': float(mendes_row['mean']),
        'mendes_ci_10': float(mendes_row['ci_10']),
        'mendes_ci_90': float(mendes_row['ci_90']),
        'pita_cotrim_effect': pita_cotrim_effect,
        'pita_mendes_effect': pita_mendes_effect,
    }

    return results


def print_comparison_table(all_results: List[Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "="*90)
    print("PRIOR SENSITIVITY: FORECAST COMPARISON")
    print("="*90)

    print(f"\n{'Scenario':<15} {'Cotrim Prior':>14} {'Cotrim Fcst':>12} {'Mendes Prior':>14} {'Mendes Fcst':>12}")
    print("-"*90)

    for r in all_results:
        cotrim_prior = f"{r['cotrim_prior_mean']*100:.0f}% (±{r['cotrim_prior_sd']*100:.0f}%)"
        mendes_prior = f"{r['mendes_prior_mean']*100:.0f}% (±{r['mendes_prior_sd']*100:.0f}%)"
        print(f"{r['scenario']:<15} {cotrim_prior:>14} {r['cotrim_mean']*100:>11.1f}% {mendes_prior:>14} {r['mendes_mean']*100:>11.1f}%")

    print("\n" + "="*90)
    print("PITAGÓRICA HOUSE EFFECT FOR COTRIM")
    print("="*90)

    print(f"\n{'Scenario':<15} {'Effect':>10} {'90% CI':>20} {'Significant?':>12}")
    print("-"*60)

    for r in all_results:
        if r['pita_cotrim_effect']:
            eff = r['pita_cotrim_effect']
            ci = f"[{eff['ci_lower']:.3f}, {eff['ci_upper']:.3f}]"
            sig = "Yes" if (eff['ci_lower'] > 0 or eff['ci_upper'] < 0) else "No"
            print(f"{r['scenario']:<15} {eff['mean']:>9.3f} {ci:>20} {sig:>12}")
        else:
            print(f"{r['scenario']:<15} {'N/A':>10}")

    print()


def plot_comparison(all_results: List[Dict[str, Any]], output_path: str):
    """Create visualization comparing scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scenarios = [r['scenario'] for r in all_results]
    x = np.arange(len(scenarios))

    # Cotrim plot
    ax = axes[0]
    cotrim_means = [r['cotrim_mean'] * 100 for r in all_results]
    cotrim_lows = [r['cotrim_ci_10'] * 100 for r in all_results]
    cotrim_highs = [r['cotrim_ci_90'] * 100 for r in all_results]
    cotrim_priors = [r['cotrim_prior_mean'] * 100 for r in all_results]

    bars = ax.bar(x, cotrim_means, color='#00CED1', alpha=0.7, label='Model Forecast')
    ax.errorbar(x, cotrim_means,
                yerr=[np.array(cotrim_means) - np.array(cotrim_lows),
                      np.array(cotrim_highs) - np.array(cotrim_means)],
                fmt='none', color='black', capsize=5)

    # Add prior markers
    ax.scatter(x, cotrim_priors, color='blue', marker='o', s=100, zorder=5, label='Prior Mean')

    # Add poll average line
    ax.axhline(19.1, color='red', linestyle='--', linewidth=2, label='Recent Poll Avg (19.1%)')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('Support (%)')
    ax.set_title('Cotrim Figueiredo: Prior Sensitivity')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 30)

    # Mendes plot
    ax = axes[1]
    mendes_means = [r['mendes_mean'] * 100 for r in all_results]
    mendes_lows = [r['mendes_ci_10'] * 100 for r in all_results]
    mendes_highs = [r['mendes_ci_90'] * 100 for r in all_results]
    mendes_priors = [r['mendes_prior_mean'] * 100 for r in all_results]

    ax.bar(x, mendes_means, color='#FF8C00', alpha=0.7, label='Model Forecast')
    ax.errorbar(x, mendes_means,
                yerr=[np.array(mendes_means) - np.array(mendes_lows),
                      np.array(mendes_highs) - np.array(mendes_means)],
                fmt='none', color='black', capsize=5)

    # Add prior markers
    ax.scatter(x, mendes_priors, color='blue', marker='o', s=100, zorder=5, label='Prior Mean')

    # Add poll average line
    ax.axhline(15.5, color='red', linestyle='--', linewidth=2, label='Recent Poll Avg (15.5%)')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('Support (%)')
    ax.set_title('Marques Mendes: Prior Sensitivity')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")
    plt.close()


def save_results(all_results: List[Dict[str, Any]], output_dir: str):
    """Save results to files."""
    # Save comparison CSV
    comparison_data = []
    for r in all_results:
        row = {
            'scenario': r['scenario'],
            'cotrim_prior_mean': r['cotrim_prior_mean'],
            'cotrim_prior_sd': r['cotrim_prior_sd'],
            'cotrim_forecast': r['cotrim_mean'],
            'cotrim_ci_10': r['cotrim_ci_10'],
            'cotrim_ci_90': r['cotrim_ci_90'],
            'mendes_prior_mean': r['mendes_prior_mean'],
            'mendes_prior_sd': r['mendes_prior_sd'],
            'mendes_forecast': r['mendes_mean'],
            'mendes_ci_10': r['mendes_ci_10'],
            'mendes_ci_90': r['mendes_ci_90'],
        }
        if r['pita_cotrim_effect']:
            row['pita_cotrim_effect'] = r['pita_cotrim_effect']['mean']
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    csv_path = Path(output_dir) / 'prior_sensitivity.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV to {csv_path}")

    # Save full results as JSON
    json_path = Path(output_dir) / 'prior_sensitivity_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved full results to {json_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prior sensitivity analysis for presidential model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/prior_sensitivity_sim",
        help="Output directory for results (default: outputs/prior_sensitivity_sim)",
    )
    parser.add_argument(
        "--polls-file",
        default="presidenciais_polls_2026.parquet",
        help="Polls parquet file in data/ (default: presidenciais_polls_2026.parquet)",
    )
    parser.add_argument(
        "--election-date",
        default="2026-01-18",
        help="Election date (default: 2026-01-18)",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=500,
        help="Number of posterior samples per chain (default: 500)",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=500,
        help="Number of tuning steps (default: 500)",
    )
    parser.add_argument(
        "--scenarios",
        nargs='+',
        choices=list(PRIOR_SCENARIOS.keys()),
        default=None,
        help="Which scenarios to run (default: all)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("PRIOR SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Polls file: {args.polls_file}")
    print(f"Election date: {args.election_date}")
    print(f"Draws: {args.draws}, Tune: {args.tune}")

    # Determine which scenarios to run
    scenarios_to_run = args.scenarios or list(PRIOR_SCENARIOS.keys())
    print(f"Scenarios to run: {scenarios_to_run}")

    # Print scenario descriptions
    print("\nScenario descriptions:")
    for name in scenarios_to_run:
        desc = PRIOR_SCENARIOS[name]['description']
        print(f"  {name}: {desc}")

    # Run each scenario
    all_results = []
    for scenario_name in scenarios_to_run:
        scenario = PRIOR_SCENARIOS[scenario_name]

        results = train_with_priors(
            scenario_name=scenario_name,
            prior_overrides=scenario['overrides'],
            polls_file=args.polls_file,
            election_date=args.election_date,
            draws=args.draws,
            tune=args.tune,
        )
        all_results.append(results)

    # Print comparison table
    print_comparison_table(all_results)

    # Create visualization
    plot_path = output_dir / 'prior_sensitivity.png'
    plot_comparison(all_results, str(plot_path))

    # Save results
    save_results(all_results, str(output_dir))

    # Interpretation summary
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    baseline = all_results[0]  # current

    # Find which scenario gets closest to poll average
    poll_avg_cotrim = 19.1
    poll_avg_mendes = 15.5

    print(f"\nTarget (recent poll avg): Cotrim {poll_avg_cotrim:.1f}%, Mendes {poll_avg_mendes:.1f}%")
    print("\nScenario results:")

    best_scenario = None
    best_error = float('inf')

    for r in all_results:
        cotrim_err = abs(r['cotrim_mean'] * 100 - poll_avg_cotrim)
        mendes_err = abs(r['mendes_mean'] * 100 - poll_avg_mendes)
        total_err = cotrim_err + mendes_err

        print(f"  {r['scenario']:<15}: Cotrim {r['cotrim_mean']*100:.1f}% (err {cotrim_err:.1f}pp), "
              f"Mendes {r['mendes_mean']*100:.1f}% (err {mendes_err:.1f}pp)")

        if total_err < best_error:
            best_error = total_err
            best_scenario = r['scenario']

    print(f"\n✓ Best scenario: {best_scenario} (total error: {best_error:.1f}pp)")

    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)

    if best_scenario == 'current':
        print("\nCurrent priors are adequate. Model is responding to data.")
    elif best_scenario == 'weak_sd':
        print("\nRecommendation: Widen prior SDs to 8%")
        print("This allows more data influence without changing prior means.")
    elif best_scenario == 'poll_avg':
        print("\nRecommendation: Update prior means to early poll averages")
        print("Cotrim: 8% → 12%, Mendes: 20% → 17%")
    elif best_scenario == 'recent_polls':
        print("\nRecommendation: Update prior means to mid-campaign values")
        print("Cotrim: 8% → 15%, Mendes: 20% → 16%")
    elif best_scenario == 'flat_prior':
        print("\nRecommendation: Use flat priors (15% mean, 10% SD)")
        print("This indicates original priors were severely miscalibrated.")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
