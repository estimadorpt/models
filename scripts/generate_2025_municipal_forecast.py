"""Generate full 2025 municipal election forecast with comprehensive outputs."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import arviz as az

from src.data.municipal_coupling import (
    build_municipal_coupling_dataset,
    TARGET_PARTIES,
    MUNICIPAL_ELECTION_DATES,
)
from src.data.municipal_common import canonicalize_party_tokens, TOKEN_TO_PARTY
from src.models.municipal_coupling_model import MunicipalCouplingModel

# Configuration
TRACE_PATH = "outputs/latest/trace.zarr"
HISTORICAL_YEARS = [2009, 2013, 2017, 2021]
FORECAST_YEAR = 2025
OUTPUT_DIR = Path("outputs/municipal_2025_forecast")
DRAWS = 1000
TUNE = 1000
SEED = 42
TARGET_ACCEPT = 0.97
POLL_PATHS = [
    "data/erc_polls_2009_2010_2011_2012_2013_2014_geo.parquet",
    "data/erc_polls_2015_2016_2017_2018_2019_geo.parquet",
    "data/erc_polls_2020_2021_2022_2023_2024_geo.parquet",
    "data/erc_polls_2025_geo.parquet",
]

def load_coalition_mappings_2025(data_dir: str = "data") -> pd.DataFrame:
    """Load 2025 coalition information for aggregation."""
    coalition_path = Path(data_dir) / "municipal_coalitions_2025.parquet"
    if coalition_path.exists():
        return pd.read_parquet(coalition_path)
    return pd.DataFrame()


def compute_winner_probabilities(samples: np.ndarray, parties: list) -> pd.DataFrame:
    """Compute winner probabilities from posterior samples.

    Args:
        samples: Shape (n_samples, n_municipalities, n_parties)
        parties: List of party names

    Returns:
        DataFrame with winner probabilities for each municipality
    """
    n_samples, n_municipalities, n_parties = samples.shape

    # Find winner for each sample
    winners = samples.argmax(axis=2)  # Shape: (n_samples, n_municipalities)

    # Compute probability each party wins
    winner_probs = np.zeros((n_municipalities, n_parties))
    for party_idx in range(n_parties):
        winner_probs[:, party_idx] = (winners == party_idx).mean(axis=0)

    prob_df = pd.DataFrame(winner_probs, columns=[f"{party}_win_prob" for party in parties])

    # Add most likely winner
    prob_df['predicted_winner'] = [parties[idx] for idx in winner_probs.argmax(axis=1)]
    prob_df['winner_probability'] = winner_probs.max(axis=1)

    return prob_df


def compute_credible_intervals(samples: np.ndarray, parties: list, prob: float = 0.9) -> pd.DataFrame:
    """Compute credible intervals for vote shares.

    Args:
        samples: Shape (n_samples, n_municipalities, n_parties)
        parties: List of party names
        prob: Credible interval probability

    Returns:
        DataFrame with mean, lower, and upper bounds for each municipality-party
    """
    n_samples, n_municipalities, n_parties = samples.shape

    alpha = (1 - prob) / 2
    lower_quantile = alpha
    upper_quantile = 1 - alpha

    results = []
    for party_idx, party in enumerate(parties):
        party_samples = samples[:, :, party_idx]  # Shape: (n_samples, n_municipalities)

        mean_shares = party_samples.mean(axis=0)
        lower_bounds = np.quantile(party_samples, lower_quantile, axis=0)
        upper_bounds = np.quantile(party_samples, upper_quantile, axis=0)

        party_df = pd.DataFrame({
            f'{party}_mean': mean_shares,
            f'{party}_lower': lower_bounds,
            f'{party}_upper': upper_bounds,
        })
        results.append(party_df)

    return pd.concat(results, axis=1)


def export_individual_samples(
    samples: np.ndarray,
    municipality_codes: list,
    parties: list,
    output_path: Path,
    max_samples: int = 1000
):
    """Export individual posterior samples to CSV.

    Args:
        samples: Shape (n_samples, n_municipalities, n_parties)
        municipality_codes: List of municipality codes
        parties: List of party names
        output_path: Path to save CSV
        max_samples: Maximum number of samples to export
    """
    n_samples = min(samples.shape[0], max_samples)

    # Reshape: (n_samples, n_municipalities, n_parties) -> rows of (sample_idx, municipality, party, value)
    records = []
    for sample_idx in range(n_samples):
        for muni_idx, muni_code in enumerate(municipality_codes):
            for party_idx, party in enumerate(parties):
                records.append({
                    'sample_idx': sample_idx,
                    'municipality_code': muni_code,
                    'party': party,
                    'vote_share': samples[sample_idx, muni_idx, party_idx]
                })

    samples_df = pd.DataFrame.from_records(records)
    samples_df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(records):,} sample records to {output_path}")


def aggregate_by_coalitions(
    predictions_df: pd.DataFrame,
    coalition_mappings: pd.DataFrame,
    parties: list
) -> pd.DataFrame:
    """Aggregate party predictions by actual coalitions running in 2025.

    Args:
        predictions_df: DataFrame with municipality-level party predictions
        coalition_mappings: DataFrame with coalition information
        parties: List of party names

    Returns:
        DataFrame with coalition-aggregated predictions
    """
    if coalition_mappings.empty:
        return pd.DataFrame()

    # Group by municipality and aggregate coalitions
    coalition_records = []

    for municipality_code in predictions_df['municipality_code'].unique():
        muni_preds = predictions_df[predictions_df['municipality_code'] == municipality_code].iloc[0]
        muni_coalitions = coalition_mappings[coalition_mappings['municipality_code'] == municipality_code]

        if muni_coalitions.empty:
            continue

        for _, coalition_row in muni_coalitions.iterrows():
            list_name = coalition_row['party']
            raw_components = str(coalition_row.get('coalition_parties', '') or '').split(';')
            raw_components = [c.strip() for c in raw_components if c.strip()]

            tokens: set[str] = set()
            if raw_components:
                for comp in raw_components:
                    tokens.update(canonicalize_party_tokens(comp))
            else:
                tokens.update(canonicalize_party_tokens(list_name))

            mapped_tokens = {TOKEN_TO_PARTY.get(tok, tok) for tok in tokens if tok}
            if not mapped_tokens:
                mapped_tokens = {"OTHER"}

            coalition_share = 0.0
            for party in mapped_tokens:
                target_col = f'{party}_mean' if f'{party}_mean' in muni_preds else party
                if target_col in muni_preds:
                    coalition_share += float(muni_preds[target_col])

            coalition_records.append({
                'municipality_code': municipality_code,
                'municipality_name': muni_preds.get('municipality_name', ''),
                'coalition_list': list_name,
                'predicted_share': coalition_share,
            })

    if not coalition_records:
        return pd.DataFrame()

    coalition_df = pd.DataFrame.from_records(coalition_records)
    return coalition_df.sort_values(['municipality_code', 'predicted_share'], ascending=[True, False])


def main():
    print("""
    ========================================================================
    🗳️  Municipal Elections 2025 - Full Forecast
    ========================================================================

    Generating comprehensive predictions for Sunday's election including:
    - Individual posterior samples for all municipalities
    - Summary statistics and credible intervals
    - Winner probabilities
    - Coalition-aggregated results
    """)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build dataset with all years including 2025
    print("\n[1/6] Loading data...")
    all_years = HISTORICAL_YEARS + [FORECAST_YEAR]
    dataset = build_municipal_coupling_dataset(
        election_years=all_years,
        trace_path=TRACE_PATH,
        train_years=HISTORICAL_YEARS,
        poll_paths=POLL_PATHS,
    )
    print(f"✓ Loaded data for {len(dataset.results['municipality_code'].unique())} municipalities")

    # Step 2: Train model on historical data
    print(f"\n[2/6] Training model on {HISTORICAL_YEARS}...")
    print("  Using separate concentration parameters for polls (looser) and results (tighter)")
    model = MunicipalCouplingModel(
        dataset=dataset,
        train_years=HISTORICAL_YEARS,
        test_year=FORECAST_YEAR,
        concentration_polls_prior=(2.0, 0.01),  # Gamma(2, 0.01) -> mean~200, loose for polls
        concentration_results_prior=(100.0, 0.1),  # Gamma(100, 0.1) -> mean~1000, tight for results
        use_separate_poll_concentration=True,
    )

    idata = model.fit(
        draws=DRAWS,
        tune=TUNE,
        target_accept=TARGET_ACCEPT,
        random_seed=SEED,
    )
    print("✓ Model training complete")

    # Step 3: Generate predictions for 2025
    print(f"\n[3/6] Generating predictions for {FORECAST_YEAR}...")
    samples = model.predict_vote_shares(idata, FORECAST_YEAR)
    print(f"✓ Generated {samples.shape[0]} posterior samples for {samples.shape[1]} municipalities")

    # Step 4: Compute summary statistics
    print("\n[4/6] Computing summary statistics...")

    # Mean predictions
    mean_shares = samples.mean(axis=0)
    mean_df = pd.DataFrame(mean_shares, columns=TARGET_PARTIES)
    mean_df.insert(0, 'municipality_code', model.municipality_codes)

    # Merge with metadata
    mean_df = mean_df.merge(dataset.metadata, on='municipality_code', how='left')

    # Credible intervals
    ci_df = compute_credible_intervals(samples, TARGET_PARTIES, prob=0.9)
    ci_df.insert(0, 'municipality_code', model.municipality_codes)

    # Winner probabilities
    winner_df = compute_winner_probabilities(samples, TARGET_PARTIES)
    winner_df.insert(0, 'municipality_code', model.municipality_codes)

    # Combine all summary stats
    summary_df = mean_df.merge(ci_df, on='municipality_code', how='left')
    summary_df = summary_df.merge(winner_df, on='municipality_code', how='left')

    print(f"✓ Computed statistics for {len(summary_df)} municipalities")

    # Step 5: Export individual samples
    print("\n[5/6] Exporting individual posterior samples...")
    export_individual_samples(
        samples=samples,
        municipality_codes=model.municipality_codes,
        parties=TARGET_PARTIES,
        output_path=OUTPUT_DIR / "posterior_samples.csv",
        max_samples=1000
    )

    # Step 6: Generate coalition-aggregated results
    print("\n[6/6] Aggregating by 2025 coalitions...")
    coalition_mappings = load_coalition_mappings_2025()
    if not coalition_mappings.empty:
        coalition_df = aggregate_by_coalitions(summary_df, coalition_mappings, TARGET_PARTIES)
        if not coalition_df.empty:
            coalition_df.to_csv(OUTPUT_DIR / "coalition_predictions.csv", index=False)
            print(f"✓ Exported coalition predictions for {len(coalition_df)} lists")
    else:
        print("⚠ No 2025 coalition data found, skipping coalition aggregation")

    # Export main outputs
    print("\n📊 Exporting final outputs...")
    summary_df.to_csv(OUTPUT_DIR / "predictions_summary.csv", index=False)
    print(f"✓ predictions_summary.csv - Full summary with means, CIs, winner probs")

    # Export simple format for website (similar to parliamentary)
    simple_columns = [
        'municipality_code',
        'municipality_name',
        'district_name',
        'predicted_winner',
        'winner_probability',
    ] + list(TARGET_PARTIES)
    simple_df = summary_df[simple_columns].copy()
    simple_df.to_csv(OUTPUT_DIR / "predictions_simple.csv", index=False)
    print(f"✓ predictions_simple.csv - Simplified format for website")

    # Save model artifacts
    az.to_netcdf(idata, OUTPUT_DIR / "posterior.nc")
    print(f"✓ posterior.nc - Full posterior inference data")

    # Generate summary statistics
    national_avg = mean_shares.mean(axis=0)
    print(f"\n📈 National average predictions (unweighted by population):")
    for party, avg in zip(TARGET_PARTIES, national_avg):
        print(f"   {party:12s}: {avg:6.2%}")

    # Count predicted winners
    predicted_winners = summary_df['predicted_winner'].value_counts()
    print(f"\n🏆 Predicted municipality winners:")
    for party, count in predicted_winners.items():
        pct = count / len(summary_df) * 100
        print(f"   {party:12s}: {count:3d} ({pct:5.1f}%)")

    # Export metadata
    metadata = {
        'forecast_year': FORECAST_YEAR,
        'training_years': HISTORICAL_YEARS,
        'n_municipalities': len(model.municipality_codes),
        'n_samples': samples.shape[0],
        'draws': DRAWS,
        'tune': TUNE,
        'seed': SEED,
        'parties': TARGET_PARTIES,
    }
    pd.Series(metadata).to_json(OUTPUT_DIR / "forecast_metadata.json")

    print(f"\n✅ Forecast complete! All outputs saved to {OUTPUT_DIR}")
    print("\n📁 Output files:")
    print(f"   - predictions_summary.csv: Complete predictions with CIs and winner probs")
    print(f"   - predictions_simple.csv: Simplified format for website")
    print(f"   - posterior_samples.csv: Individual posterior samples (1000 samples)")
    print(f"   - coalition_predictions.csv: Coalition-aggregated predictions")
    print(f"   - posterior.nc: Full posterior inference data")
    print(f"   - forecast_metadata.json: Forecast configuration and metadata")


if __name__ == "__main__":
    main()
