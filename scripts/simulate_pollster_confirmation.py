#!/usr/bin/env python
"""
Simulate Pollster Confirmation for Presidential Model Diagnostics

This script tests whether the presidential model will self-correct when
other pollsters (not just Pitagórica) confirm similar numbers for candidates.

The model currently shows:
- Cotrim Figueiredo at ~15% (but polls show 18-21%)
- Marques Mendes at ~18% (but polls show 14-16%)

This discrepancy may be because all recent polls are from Pitagórica,
and the model attributes the difference to house effects rather than
genuine candidate movement.

DIAGNOSTIC ONLY - This script does NOT modify the model.
It creates synthetic poll data to test model behavior.

Usage:
    pixi run python scripts/simulate_pollster_confirmation.py \
        --output-dir outputs/pollster_confirmation_sim
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.presidential_dataset import PresidentialElectionDataset
from src.data.presidential_loaders import load_presidential_polls, DEFAULT_CANDIDATES_2026
from src.models.presidential_election_model import PresidentialElectionModel


# =============================================================================
# SYNTHETIC POLL DEFINITIONS
# =============================================================================

# Use Pitagórica's Jan 10 numbers as the "confirmed" values
# These are the values other pollsters will "confirm"
CONFIRMED_VALUES = {
    'Gouveia e Melo': 17.0,
    'Marques Mendes': 14.5,
    'António José Seguro': 21.4,
    'André Ventura': 19.7,
    'Cotrim Figueiredo': 21.1,
    'Catarina Martins': 2.6,
    'António Filipe': 2.0,
    'Jorge Pinto': 0.9,
    'Others': 0.7,
}

# Map short candidate names to poll data column format
CANDIDATE_TO_POLL_NAME = {
    'Gouveia e Melo': 'Henrique Gouveia e Melo',
    'Marques Mendes': 'Luís Marques Mendes',
    'António José Seguro': 'António José Seguro',
    'André Ventura': 'André Ventura',
    'Cotrim Figueiredo': 'João Cotrim Figueiredo',
    'Catarina Martins': 'Catarina Martins',
    'António Filipe': 'António Filipe',
    'Jorge Pinto': 'Jorge Pinto',
    'Others': 'Others',
}

# Define synthetic confirmation polls
SYNTHETIC_POLLS = [
    {
        'poll_id': 'synthetic_aximage_jan12',
        'pollster': 'Aximage',
        'fieldwork_start': '2026-01-10',
        'fieldwork_end': '2026-01-12',
        'sample_size': 800,
        'candidates': CONFIRMED_VALUES.copy(),
    },
    {
        'poll_id': 'synthetic_intercampus_jan12',
        'pollster': 'Intercampus',
        'fieldwork_start': '2026-01-10',
        'fieldwork_end': '2026-01-12',
        'sample_size': 800,
        # Slight variation to seem realistic
        'candidates': {
            'Gouveia e Melo': 17.5,
            'Marques Mendes': 15.0,
            'António José Seguro': 20.5,
            'André Ventura': 20.0,
            'Cotrim Figueiredo': 19.5,
            'Catarina Martins': 3.0,
            'António Filipe': 2.5,
            'Jorge Pinto': 1.2,
            'Others': 0.8,
        },
    },
    {
        'poll_id': 'synthetic_ics_jan13',
        'pollster': 'ICS/ISCTE',
        'fieldwork_start': '2026-01-11',
        'fieldwork_end': '2026-01-13',
        'sample_size': 900,
        'candidates': {
            'Gouveia e Melo': 16.5,
            'Marques Mendes': 14.0,
            'António José Seguro': 21.0,
            'André Ventura': 19.5,
            'Cotrim Figueiredo': 20.5,
            'Catarina Martins': 3.2,
            'António Filipe': 2.8,
            'Jorge Pinto': 1.5,
            'Others': 1.0,
        },
    },
]

SCENARIOS = {
    'baseline': [],  # No synthetic polls
    'plus_1_aximage': [SYNTHETIC_POLLS[0]],
    'plus_2_pollsters': [SYNTHETIC_POLLS[0], SYNTHETIC_POLLS[1]],
    'plus_3_pollsters': [SYNTHETIC_POLLS[0], SYNTHETIC_POLLS[1], SYNTHETIC_POLLS[2]],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_synthetic_poll_rows(synthetic_poll: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a synthetic poll definition to DataFrame rows in long format.

    The poll data is stored in long format with one row per candidate per poll.
    """
    rows = []
    for short_name, pct in synthetic_poll['candidates'].items():
        full_name = CANDIDATE_TO_POLL_NAME.get(short_name, short_name)
        rows.append({
            'poll_id': synthetic_poll['poll_id'],
            'source': 'synthetic',
            'source_id': None,
            'pollster': synthetic_poll['pollster'],
            'fieldwork_start': pd.to_datetime(synthetic_poll['fieldwork_start']),
            'fieldwork_end': pd.to_datetime(synthetic_poll['fieldwork_end']),
            'sample_size': synthetic_poll['sample_size'],
            'methodology': 'Synthetic',
            'election_type': 'Presidenciais',
            'geographic_scope': 'Nacional',
            'poll_type': 'first_round',
            'year': 2026,
            'candidate_name': full_name,
            'vote_intention_pct': pct,
            'party_affiliation': None,
        })
    return pd.DataFrame(rows)


def load_polls_with_synthetic(
    base_polls_file: str,
    synthetic_polls: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Load base polls and add synthetic polls.

    Returns the merged DataFrame in long format (ready for parquet-like processing).
    """
    # Load base polls as parquet (long format)
    base_path = Path(__file__).parent.parent / 'data' / base_polls_file
    base_df = pd.read_parquet(base_path)

    if not synthetic_polls:
        return base_df

    # Create synthetic poll rows
    synthetic_dfs = [create_synthetic_poll_rows(sp) for sp in synthetic_polls]
    synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)

    # Merge
    combined_df = pd.concat([base_df, synthetic_df], ignore_index=True)

    return combined_df


def train_scenario(
    scenario_name: str,
    synthetic_polls: List[Dict[str, Any]],
    base_polls_file: str,
    election_date: str,
    draws: int = 500,
    tune: int = 500,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a model for a given scenario and extract key results.
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"Synthetic polls: {len(synthetic_polls)}")
    print(f"{'='*60}")

    # Load polls with synthetic additions
    combined_polls = load_polls_with_synthetic(base_polls_file, synthetic_polls)

    # Save combined polls to temp file in the DATA directory (where loader expects it)
    data_dir = Path(__file__).parent.parent / 'data'
    temp_polls_path = data_dir / f'temp_polls_{scenario_name}.parquet'
    combined_polls.to_parquet(temp_polls_path)

    try:
        # Create dataset from temp file (now in correct location)
        dataset = PresidentialElectionDataset(
            election_date=election_date,
            polls_file=temp_polls_path.name,  # Just filename, loader adds data/ prefix
        )

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
            'n_synthetic_polls': len(synthetic_polls),
            'n_total_polls': len(dataset.polls_train),
            'cotrim_mean': float(cotrim_row['mean']),
            'cotrim_ci_10': float(cotrim_row['ci_10']),
            'cotrim_ci_90': float(cotrim_row['ci_90']),
            'mendes_mean': float(mendes_row['mean']),
            'mendes_ci_10': float(mendes_row['ci_10']),
            'mendes_ci_90': float(mendes_row['ci_90']),
            'pita_cotrim_effect': pita_cotrim_effect,
            'pita_mendes_effect': pita_mendes_effect,
            'pollsters': pollsters,
        }

        return results

    finally:
        # Clean up temp file
        if temp_polls_path.exists():
            temp_polls_path.unlink()


def print_comparison_table(all_results: List[Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("FORECAST COMPARISON TABLE")
    print("="*80)

    print(f"\n{'Scenario':<20} {'Cotrim Mean':>12} {'Cotrim 80%CI':>18} {'Mendes Mean':>12} {'Mendes 80%CI':>18}")
    print("-"*80)

    for r in all_results:
        cotrim_ci = f"[{r['cotrim_ci_10']*100:.1f}%, {r['cotrim_ci_90']*100:.1f}%]"
        mendes_ci = f"[{r['mendes_ci_10']*100:.1f}%, {r['mendes_ci_90']*100:.1f}%]"
        print(f"{r['scenario']:<20} {r['cotrim_mean']*100:>11.1f}% {cotrim_ci:>18} {r['mendes_mean']*100:>11.1f}% {mendes_ci:>18}")

    print("\n" + "="*80)
    print("PITAGÓRICA HOUSE EFFECT COMPARISON")
    print("="*80)

    print(f"\n{'Scenario':<20} {'Cotrim Effect':>15} {'Cotrim 90%CI':>20} {'Significant?':>12}")
    print("-"*70)

    for r in all_results:
        if r['pita_cotrim_effect']:
            eff = r['pita_cotrim_effect']
            ci = f"[{eff['ci_lower']:.3f}, {eff['ci_upper']:.3f}]"
            sig = "Yes" if (eff['ci_lower'] > 0 or eff['ci_upper'] < 0) else "No"
            print(f"{r['scenario']:<20} {eff['mean']:>14.3f} {ci:>20} {sig:>12}")
        else:
            print(f"{r['scenario']:<20} {'N/A':>15}")

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

    ax.bar(x, cotrim_means, color='#00CED1', alpha=0.7, label='Model Estimate')
    ax.errorbar(x, cotrim_means,
                yerr=[np.array(cotrim_means) - np.array(cotrim_lows),
                      np.array(cotrim_highs) - np.array(cotrim_means)],
                fmt='none', color='black', capsize=5)

    # Add poll average line
    ax.axhline(19.1, color='red', linestyle='--', linewidth=2, label='Recent Poll Avg (19.1%)')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('Support (%)')
    ax.set_title('Cotrim Figueiredo: Model Response to Confirmations')
    ax.legend()
    ax.set_ylim(0, 30)

    # Mendes plot
    ax = axes[1]
    mendes_means = [r['mendes_mean'] * 100 for r in all_results]
    mendes_lows = [r['mendes_ci_10'] * 100 for r in all_results]
    mendes_highs = [r['mendes_ci_90'] * 100 for r in all_results]

    ax.bar(x, mendes_means, color='#FF8C00', alpha=0.7, label='Model Estimate')
    ax.errorbar(x, mendes_means,
                yerr=[np.array(mendes_means) - np.array(mendes_lows),
                      np.array(mendes_highs) - np.array(mendes_means)],
                fmt='none', color='black', capsize=5)

    # Add poll average line
    ax.axhline(15.5, color='red', linestyle='--', linewidth=2, label='Recent Poll Avg (15.5%)')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('Support (%)')
    ax.set_title('Marques Mendes: Model Response to Confirmations')
    ax.legend()
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
            'n_synthetic_polls': r['n_synthetic_polls'],
            'n_total_polls': r['n_total_polls'],
            'cotrim_mean': r['cotrim_mean'],
            'cotrim_ci_10': r['cotrim_ci_10'],
            'cotrim_ci_90': r['cotrim_ci_90'],
            'mendes_mean': r['mendes_mean'],
            'mendes_ci_10': r['mendes_ci_10'],
            'mendes_ci_90': r['mendes_ci_90'],
        }
        if r['pita_cotrim_effect']:
            row['pita_cotrim_effect_mean'] = r['pita_cotrim_effect']['mean']
            row['pita_cotrim_effect_ci_lower'] = r['pita_cotrim_effect']['ci_lower']
            row['pita_cotrim_effect_ci_upper'] = r['pita_cotrim_effect']['ci_upper']
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    csv_path = Path(output_dir) / 'confirmation_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV to {csv_path}")

    # Save full results as JSON
    json_path = Path(output_dir) / 'confirmation_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved full results to {json_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulate pollster confirmation for presidential model diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/pollster_confirmation_sim",
        help="Output directory for results (default: outputs/pollster_confirmation_sim)",
    )
    parser.add_argument(
        "--polls-file",
        default="presidenciais_polls_2026.parquet",
        help="Base polls parquet file in data/ (default: presidenciais_polls_2026.parquet)",
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
        choices=list(SCENARIOS.keys()),
        default=None,
        help="Which scenarios to run (default: all)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("POLLSTER CONFIRMATION SIMULATION")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Base polls file: {args.polls_file}")
    print(f"Election date: {args.election_date}")
    print(f"Draws: {args.draws}, Tune: {args.tune}")

    # Determine which scenarios to run
    scenarios_to_run = args.scenarios or list(SCENARIOS.keys())
    print(f"Scenarios to run: {scenarios_to_run}")

    # Run each scenario
    all_results = []
    for scenario_name in scenarios_to_run:
        synthetic_polls = SCENARIOS[scenario_name]

        results = train_scenario(
            scenario_name=scenario_name,
            synthetic_polls=synthetic_polls,
            base_polls_file=args.polls_file,
            election_date=args.election_date,
            draws=args.draws,
            tune=args.tune,
            output_dir=str(output_dir),
        )
        all_results.append(results)

    # Print comparison table
    print_comparison_table(all_results)

    # Create visualization
    plot_path = output_dir / 'confirmation_comparison.png'
    plot_comparison(all_results, str(plot_path))

    # Save results
    save_results(all_results, str(output_dir))

    # Interpretation summary
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    baseline = all_results[0]
    final = all_results[-1]

    cotrim_change = (final['cotrim_mean'] - baseline['cotrim_mean']) * 100
    mendes_change = (final['mendes_mean'] - baseline['mendes_mean']) * 100

    print(f"\nWith {final['n_synthetic_polls']} confirming polls:")
    print(f"  Cotrim moved: {baseline['cotrim_mean']*100:.1f}% → {final['cotrim_mean']*100:.1f}% ({cotrim_change:+.1f}pp)")
    print(f"  Mendes moved: {baseline['mendes_mean']*100:.1f}% → {final['mendes_mean']*100:.1f}% ({mendes_change:+.1f}pp)")

    if cotrim_change >= 3:
        print("\n✓ Model responds well to pollster confirmation.")
        print("  Wait for real polls from other pollsters to validate Pitagórica's numbers.")
    elif cotrim_change >= 1:
        print("\n△ Model responds partially to confirmation.")
        print("  May need 4+ confirming polls or consider increasing innovation_sd.")
    else:
        print("\n✗ Model does not respond sufficiently to confirmation.")
        print("  Consider adjusting priors or innovation_sd parameter.")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
