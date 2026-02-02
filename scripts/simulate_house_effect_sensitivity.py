#!/usr/bin/env python
"""
House Effect Sensitivity Analysis for Presidential Model

This script tests different house effect configurations to find one that
allows the model to respond appropriately to poll data.

The problem: The model learns house effects from the entire poll history.
When one pollster (Pitagórica) showed higher Cotrim early, the model
learned "Pitagórica is biased +10pp for Cotrim". Now it subtracts 10pp
from all Pitagórica polls, even though the early difference might have
been Pitagórica detecting movement earlier, not bias.

DIAGNOSTIC ONLY - This script does NOT permanently modify the model.

Usage:
    pixi run python scripts/simulate_house_effect_sensitivity.py \
        --output-dir outputs/house_effect_sensitivity
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.presidential_dataset import PresidentialElectionDataset
from src.models.presidential_election_model import PresidentialElectionModel


# =============================================================================
# HOUSE EFFECT SCENARIOS
# =============================================================================

HOUSE_EFFECT_SCENARIOS = {
    'current': {
        'description': 'Current settings (parliamentary priors, SD=0.04)',
        'use_parliamentary_house_priors': True,
        'house_effect_sd_scale': 0.04,
    },
    'shrunk': {
        'description': 'Shrunk house effects (parliamentary priors, SD=0.02)',
        'use_parliamentary_house_priors': True,
        'house_effect_sd_scale': 0.02,
    },
    'disabled': {
        'description': 'Disabled house effects (no priors, SD=0.01)',
        'use_parliamentary_house_priors': False,
        'house_effect_sd_scale': 0.01,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def train_with_house_effect_config(
    scenario_name: str,
    use_parliamentary_house_priors: bool,
    house_effect_sd_scale: float,
    polls_file: str,
    election_date: str,
    draws: int = 500,
    tune: int = 500,
) -> Dict[str, Any]:
    """
    Train a model with specific house effect configuration.
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"use_parliamentary_house_priors: {use_parliamentary_house_priors}")
    print(f"house_effect_sd_scale: {house_effect_sd_scale}")
    print(f"{'='*60}")

    # Create dataset
    dataset = PresidentialElectionDataset(
        election_date=election_date,
        polls_file=polls_file,
    )

    # Build model with specific house effect config
    model = PresidentialElectionModel(
        dataset=dataset,
        use_parliamentary_house_priors=use_parliamentary_house_priors,
        house_effect_sd_scale=house_effect_sd_scale,
    )
    model.build_model()

    print(f"\nSampling {draws} draws with {tune} tuning steps...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.sample(draws=draws, tune=tune, chains=4)

    # Extract results
    forecast_df = model.get_forecast()

    # Extract house effects for Pitagórica
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

    # Get all candidate forecasts
    all_forecasts = {}
    for _, row in forecast_df.iterrows():
        all_forecasts[row['candidate']] = {
            'mean': float(row['mean']),
            'ci_10': float(row['ci_10']),
            'ci_90': float(row['ci_90']),
        }

    # Get Cotrim and Mendes specifically
    cotrim_row = forecast_df[forecast_df['candidate'] == 'Cotrim Figueiredo'].iloc[0]
    mendes_row = forecast_df[forecast_df['candidate'] == 'Marques Mendes'].iloc[0]

    results = {
        'scenario': scenario_name,
        'use_parliamentary_house_priors': use_parliamentary_house_priors,
        'house_effect_sd_scale': house_effect_sd_scale,
        'cotrim_mean': float(cotrim_row['mean']),
        'cotrim_ci_10': float(cotrim_row['ci_10']),
        'cotrim_ci_90': float(cotrim_row['ci_90']),
        'mendes_mean': float(mendes_row['mean']),
        'mendes_ci_10': float(mendes_row['ci_10']),
        'mendes_ci_90': float(mendes_row['ci_90']),
        'pita_cotrim_effect': pita_cotrim_effect,
        'pita_mendes_effect': pita_mendes_effect,
        'all_forecasts': all_forecasts,
    }

    return results


def print_comparison_table(all_results: List[Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "="*90)
    print("HOUSE EFFECT SENSITIVITY: FORECAST COMPARISON")
    print("="*90)

    # Target values from recent polls
    poll_avg_cotrim = 19.1
    poll_avg_mendes = 15.5

    print(f"\nTarget (recent poll avg): Cotrim {poll_avg_cotrim:.1f}%, Mendes {poll_avg_mendes:.1f}%")

    print(f"\n{'Scenario':<12} {'HE SD':>8} {'Cotrim':>10} {'Gap':>8} {'Mendes':>10} {'Gap':>8}")
    print("-"*60)

    for r in all_results:
        cotrim_gap = r['cotrim_mean'] * 100 - poll_avg_cotrim
        mendes_gap = r['mendes_mean'] * 100 - poll_avg_mendes
        print(f"{r['scenario']:<12} {r['house_effect_sd_scale']:>8.2f} "
              f"{r['cotrim_mean']*100:>9.1f}% {cotrim_gap:>+7.1f}pp "
              f"{r['mendes_mean']*100:>9.1f}% {mendes_gap:>+7.1f}pp")

    print("\n" + "="*90)
    print("PITAGÓRICA HOUSE EFFECT FOR COTRIM")
    print("="*90)

    print(f"\n{'Scenario':<12} {'Effect':>10} {'90% CI':>22} {'Significant?':>14}")
    print("-"*65)

    for r in all_results:
        if r['pita_cotrim_effect']:
            eff = r['pita_cotrim_effect']
            ci = f"[{eff['ci_lower']:.3f}, {eff['ci_upper']:.3f}]"
            sig = "Yes" if (eff['ci_lower'] > 0 or eff['ci_upper'] < 0) else "No"
            print(f"{r['scenario']:<12} {eff['mean']:>9.3f} {ci:>22} {sig:>14}")
        else:
            print(f"{r['scenario']:<12} {'N/A':>10}")

    print()


def print_full_forecast(results: Dict[str, Any]):
    """Print full forecast for all candidates."""
    print(f"\n{'='*60}")
    print(f"FULL FORECAST: {results['scenario']}")
    print(f"{'='*60}")

    forecasts = results['all_forecasts']
    sorted_candidates = sorted(forecasts.items(), key=lambda x: -x[1]['mean'])

    print(f"\n{'Candidate':<25} {'Mean':>10} {'80% CI':>20}")
    print("-"*55)

    for candidate, forecast in sorted_candidates:
        ci = f"[{forecast['ci_10']*100:.1f}%, {forecast['ci_90']*100:.1f}%]"
        print(f"{candidate:<25} {forecast['mean']*100:>9.1f}% {ci:>20}")


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

    bars = ax.bar(x, cotrim_means, color='#00CED1', alpha=0.7, label='Model Forecast')
    ax.errorbar(x, cotrim_means,
                yerr=[np.array(cotrim_means) - np.array(cotrim_lows),
                      np.array(cotrim_highs) - np.array(cotrim_means)],
                fmt='none', color='black', capsize=5)

    # Add poll average line
    ax.axhline(19.1, color='red', linestyle='--', linewidth=2, label='Recent Poll Avg (19.1%)')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Support (%)')
    ax.set_title('Cotrim Figueiredo: House Effect Sensitivity')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 30)

    # Mendes plot
    ax = axes[1]
    mendes_means = [r['mendes_mean'] * 100 for r in all_results]
    mendes_lows = [r['mendes_ci_10'] * 100 for r in all_results]
    mendes_highs = [r['mendes_ci_90'] * 100 for r in all_results]

    ax.bar(x, mendes_means, color='#FF8C00', alpha=0.7, label='Model Forecast')
    ax.errorbar(x, mendes_means,
                yerr=[np.array(mendes_means) - np.array(mendes_lows),
                      np.array(mendes_highs) - np.array(mendes_means)],
                fmt='none', color='black', capsize=5)

    # Add poll average line
    ax.axhline(15.5, color='red', linestyle='--', linewidth=2, label='Recent Poll Avg (15.5%)')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Support (%)')
    ax.set_title('Marques Mendes: House Effect Sensitivity')
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
            'use_parliamentary_priors': r['use_parliamentary_house_priors'],
            'house_effect_sd_scale': r['house_effect_sd_scale'],
            'cotrim_forecast': r['cotrim_mean'],
            'cotrim_ci_10': r['cotrim_ci_10'],
            'cotrim_ci_90': r['cotrim_ci_90'],
            'mendes_forecast': r['mendes_mean'],
            'mendes_ci_10': r['mendes_ci_10'],
            'mendes_ci_90': r['mendes_ci_90'],
        }
        if r['pita_cotrim_effect']:
            row['pita_cotrim_effect'] = r['pita_cotrim_effect']['mean']
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    csv_path = Path(output_dir) / 'house_effect_sensitivity.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV to {csv_path}")

    # Save full results as JSON
    json_path = Path(output_dir) / 'house_effect_sensitivity_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved full results to {json_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="House effect sensitivity analysis for presidential model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/house_effect_sensitivity",
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
        choices=list(HOUSE_EFFECT_SCENARIOS.keys()),
        default=None,
        help="Which scenarios to run (default: all)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("HOUSE EFFECT SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Polls file: {args.polls_file}")
    print(f"Election date: {args.election_date}")
    print(f"Draws: {args.draws}, Tune: {args.tune}")

    # Determine which scenarios to run
    scenarios_to_run = args.scenarios or list(HOUSE_EFFECT_SCENARIOS.keys())
    print(f"Scenarios to run: {scenarios_to_run}")

    # Print scenario descriptions
    print("\nScenario descriptions:")
    for name in scenarios_to_run:
        desc = HOUSE_EFFECT_SCENARIOS[name]['description']
        print(f"  {name}: {desc}")

    # Run each scenario
    all_results = []
    for scenario_name in scenarios_to_run:
        scenario = HOUSE_EFFECT_SCENARIOS[scenario_name]

        results = train_with_house_effect_config(
            scenario_name=scenario_name,
            use_parliamentary_house_priors=scenario['use_parliamentary_house_priors'],
            house_effect_sd_scale=scenario['house_effect_sd_scale'],
            polls_file=args.polls_file,
            election_date=args.election_date,
            draws=args.draws,
            tune=args.tune,
        )
        all_results.append(results)

    # Print comparison table
    print_comparison_table(all_results)

    # Print full forecast for disabled scenario
    disabled_result = next((r for r in all_results if r['scenario'] == 'disabled'), None)
    if disabled_result:
        print_full_forecast(disabled_result)

    # Create visualization
    plot_path = output_dir / 'house_effect_sensitivity.png'
    plot_comparison(all_results, str(plot_path))

    # Save results
    save_results(all_results, str(output_dir))

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    poll_avg_cotrim = 19.1
    poll_avg_mendes = 15.5

    best_scenario = None
    best_error = float('inf')

    for r in all_results:
        cotrim_err = abs(r['cotrim_mean'] * 100 - poll_avg_cotrim)
        mendes_err = abs(r['mendes_mean'] * 100 - poll_avg_mendes)
        total_err = cotrim_err + mendes_err

        if total_err < best_error:
            best_error = total_err
            best_scenario = r

    print(f"\nBest scenario: {best_scenario['scenario']} (total error: {best_error:.1f}pp)")
    print(f"  Cotrim: {best_scenario['cotrim_mean']*100:.1f}% (target: {poll_avg_cotrim:.1f}%)")
    print(f"  Mendes: {best_scenario['mendes_mean']*100:.1f}% (target: {poll_avg_mendes:.1f}%)")

    if best_scenario['pita_cotrim_effect']:
        eff = best_scenario['pita_cotrim_effect']
        print(f"  Pitagórica Cotrim effect: {eff['mean']:.3f}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
