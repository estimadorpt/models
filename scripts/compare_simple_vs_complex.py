#!/usr/bin/env python
"""
Quick comparison: Simple parameter tuning vs Complex fixed_early approach.

Question: Is fixed_early better than just shrinking house effects
and allowing higher innovation?

Scenarios:
1. current: baseline (innovation=0.05, parliamentary HE priors)
2. fixed_early: complex two-stage approach
3. simple_tuned: shrunk HE (SD=0.02) + higher innovation (0.08)
4. simple_aggressive: disabled HE (SD=0.01) + higher innovation (0.10)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.presidential_dataset import PresidentialElectionDataset
from src.models.presidential_election_model import PresidentialElectionModel


def run_scenario(name: str, description: str, **model_kwargs):
    """Run a single scenario and return results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{description}")
    print(f"{'='*60}")

    dataset = PresidentialElectionDataset(
        election_date='2026-01-18',
        polls_file='presidenciais_polls_2026.parquet',
    )

    model = PresidentialElectionModel(dataset=dataset, **model_kwargs)
    model.build_model()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.sample(draws=500, tune=500, chains=4)

    forecast = model.get_forecast()

    # Get house effects
    he = model.trace.posterior['house_effects']
    pollsters = list(model.coords['pollsters'])
    candidates = dataset.candidates

    pita_idx = next((i for i, p in enumerate(pollsters) if 'pitag' in p.lower()), None)
    cotrim_idx = candidates.index('Cotrim Figueiredo') if 'Cotrim Figueiredo' in candidates else None

    pita_cotrim_he = None
    if pita_idx is not None and cotrim_idx is not None:
        effects = he.isel(pollsters=pita_idx, candidates=cotrim_idx)
        pita_cotrim_he = float(effects.mean())

    cotrim_row = forecast[forecast['candidate'] == 'Cotrim Figueiredo'].iloc[0]
    mendes_row = forecast[forecast['candidate'] == 'Marques Mendes'].iloc[0]

    return {
        'scenario': name,
        'description': description,
        'cotrim': cotrim_row['mean'] * 100,
        'cotrim_ci': (cotrim_row['ci_10'] * 100, cotrim_row['ci_90'] * 100),
        'mendes': mendes_row['mean'] * 100,
        'mendes_ci': (mendes_row['ci_10'] * 100, mendes_row['ci_90'] * 100),
        'pita_cotrim_he': pita_cotrim_he,
    }


def main():
    print("="*70)
    print("SIMPLE vs COMPLEX: Parameter Tuning vs Fixed Early HE")
    print("="*70)

    results = []

    # 1. Current baseline
    results.append(run_scenario(
        name='current',
        description='innovation=0.05, parliamentary HE (baseline)',
        innovation_sd_scale=0.05,
        use_parliamentary_house_priors=True,
        house_effect_sd_scale=0.04,
    ))

    # 2. Simple tuned: shrunk HE + higher innovation
    results.append(run_scenario(
        name='simple_tuned',
        description='innovation=0.08, shrunk HE (SD=0.02)',
        innovation_sd_scale=0.08,
        use_parliamentary_house_priors=True,
        house_effect_sd_scale=0.02,
    ))

    # 3. Simple aggressive: disabled HE + high innovation
    results.append(run_scenario(
        name='simple_aggressive',
        description='innovation=0.10, disabled HE (SD=0.01)',
        innovation_sd_scale=0.10,
        use_parliamentary_house_priors=False,
        house_effect_sd_scale=0.01,
    ))

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON: Simple Parameter Tuning")
    print("="*80)

    poll_avg_cotrim = 19.1
    poll_avg_mendes = 15.5

    print(f"\nTarget (last 3 polls): Cotrim {poll_avg_cotrim}%, Mendes {poll_avg_mendes}%")
    print(f"\n{'Scenario':<20} {'Cotrim':>10} {'Gap':>8} {'Mendes':>10} {'Gap':>8} {'Pita HE':>10}")
    print("-"*70)

    for r in results:
        cotrim_gap = r['cotrim'] - poll_avg_cotrim
        mendes_gap = r['mendes'] - poll_avg_mendes
        he_str = f"{r['pita_cotrim_he']:+.3f}" if r['pita_cotrim_he'] else "N/A"
        print(f"{r['scenario']:<20} {r['cotrim']:>9.1f}% {cotrim_gap:>+7.1f}pp "
              f"{r['mendes']:>9.1f}% {mendes_gap:>+7.1f}pp {he_str:>10}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    # Find best
    best = min(results, key=lambda r: abs(r['cotrim'] - poll_avg_cotrim) + abs(r['mendes'] - poll_avg_mendes))
    print(f"\nBest scenario: {best['scenario']}")
    print(f"  Cotrim: {best['cotrim']:.1f}% (gap: {best['cotrim'] - poll_avg_cotrim:+.1f}pp)")
    print(f"  Mendes: {best['mendes']:.1f}% (gap: {best['mendes'] - poll_avg_mendes:+.1f}pp)")

    # Compare simple_tuned vs fixed_early (from previous run)
    print("\n" + "-"*60)
    print("SIMPLE vs COMPLEX COMPARISON:")
    print("-"*60)
    print("fixed_early (complex):  Cotrim 17.6%, gap -1.5pp")

    simple_tuned = next((r for r in results if r['scenario'] == 'simple_tuned'), None)
    if simple_tuned:
        print(f"simple_tuned:           Cotrim {simple_tuned['cotrim']:.1f}%, gap {simple_tuned['cotrim'] - poll_avg_cotrim:+.1f}pp")

        if abs(simple_tuned['cotrim'] - 17.6) < 1.0:
            print("\n→ Simple tuning achieves SIMILAR results to complex approach!")
            print("→ RECOMMEND: Use simple_tuned for production (easier to maintain)")
        elif simple_tuned['cotrim'] > 17.6:
            print(f"\n→ Simple tuning is BETTER (+{simple_tuned['cotrim'] - 17.6:.1f}pp)")
            print("→ RECOMMEND: Use simple_tuned for production")
        else:
            print(f"\n→ Complex approach is better by {17.6 - simple_tuned['cotrim']:.1f}pp")
            print("→ Consider: Is the complexity worth the improvement?")


if __name__ == "__main__":
    main()
