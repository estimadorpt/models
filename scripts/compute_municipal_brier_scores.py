"""Compute Brier scores for municipal coupling model outputs."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compute_brier_score(predicted_probs: np.ndarray, observed_shares: np.ndarray) -> float:
    """
    Compute Brier score for probabilistic predictions.

    Args:
        predicted_probs: Array of predicted probabilities, shape (n_obs, n_categories)
        observed_shares: Array of observed vote shares, shape (n_obs, n_categories)

    Returns:
        Brier score (lower is better)
    """
    # Brier score is mean squared error between predicted probs and observed outcomes
    squared_errors = (predicted_probs - observed_shares) ** 2
    brier_score = squared_errors.mean()

    return brier_score


def compute_brier_from_csvs(
    observed_path: Path,
    predictions_path: Path,
    parties: list
) -> dict:
    """
    Compute Brier score and related metrics from saved CSV files.

    Args:
        observed_path: Path to observed.csv file
        predictions_path: Path to predictions.csv file
        parties: List of party names

    Returns:
        Dictionary with Brier score and other metrics
    """
    # Load observed data
    print(f"  Loading observed data from {observed_path.name}")
    observed_df = pd.read_csv(observed_path)

    # Load predictions (mean probabilities)
    print(f"  Loading predictions from {predictions_path.name}")
    predictions_df = pd.read_csv(predictions_path)

    # Check they have same municipalities
    if not observed_df["municipality_code"].equals(predictions_df["municipality_code"]):
        print("  Warning: Municipality codes don't match exactly, aligning...")
        # Merge on municipality_code to ensure alignment
        merged = observed_df[["municipality_code"] + parties].merge(
            predictions_df[["municipality_code"] + parties],
            on="municipality_code",
            suffixes=("_obs", "_pred")
        )
        observed_shares = merged[[f"{p}_obs" for p in parties]].to_numpy()
        predicted_probs = merged[[f"{p}_pred" for p in parties]].to_numpy()
    else:
        # Extract vote shares
        observed_shares = observed_df[parties].to_numpy()
        predicted_probs = predictions_df[parties].to_numpy()

    print(f"  Computing Brier score for {len(observed_shares)} municipalities")

    # Compute overall Brier score
    brier = compute_brier_score(predicted_probs, observed_shares)

    # Compute per-party Brier scores
    party_brier_scores = {}
    for i, party in enumerate(parties):
        party_pred = predicted_probs[:, i]
        party_obs = observed_shares[:, i]
        party_brier = ((party_pred - party_obs) ** 2).mean()
        party_brier_scores[party] = party_brier

    # Compute calibration metrics
    # For each party, bin predictions into deciles and check calibration
    n_bins = 10
    all_calibration_errors = []

    for i, party in enumerate(parties):
        pred = predicted_probs[:, i]
        obs = observed_shares[:, i]

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(pred, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute calibration error for each bin
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 5:  # Only use bins with sufficient data
                mean_pred = pred[mask].mean()
                mean_obs = obs[mask].mean()
                all_calibration_errors.append(abs(mean_pred - mean_obs))

    mean_calibration_error = np.mean(all_calibration_errors) if all_calibration_errors else np.nan

    results = {
        "brier_score": brier,
        "mean_calibration_error": mean_calibration_error,
        "party_brier_scores": party_brier_scores,
        "n_municipalities": len(observed_shares)
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute Brier scores for municipal model outputs")
    parser.add_argument("output_dirs", nargs="+", help="Output directories to evaluate")
    parser.add_argument("--parties", default="PS,PSD,CDS-PP,BE,CDU,CH,IL,LOCAL_INC,OTHER",
                       help="Comma-separated list of party names")

    args = parser.parse_args()
    parties = args.parties.split(",")

    all_results = {}

    for output_dir in args.output_dirs:
        output_path = Path(output_dir)

        if not output_path.exists():
            print(f"\n❌ Skipping {output_dir}: directory not found")
            continue

        observed_path = output_path / "observed.csv"
        predictions_path = output_path / "predictions.csv"

        if not all([observed_path.exists(), predictions_path.exists()]):
            print(f"\n❌ Skipping {output_dir}: missing required CSV files")
            continue

        print(f"\n{'='*70}")
        print(f"📊 Evaluating: {output_path.name}")
        print('='*70)

        try:
            results = compute_brier_from_csvs(
                observed_path,
                predictions_path,
                parties
            )

            all_results[str(output_path)] = results

            print(f"\n✅ Results:")
            print(f"  Brier Score:                {results['brier_score']:.6f}")
            print(f"  Mean Calibration Error:     {results['mean_calibration_error']:.6f}")
            print(f"  Number of municipalities:   {results['n_municipalities']}")

            print(f"\n  Per-party Brier scores:")
            sorted_parties = sorted(results['party_brier_scores'].items(), key=lambda x: x[1])
            for party, score in sorted_parties:
                print(f"    {party:12s}: {score:.6f}")

        except Exception as e:
            print(f"\n❌ Error processing {output_dir}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n\n{'='*70}")
        print("🔍 COMPARISON SUMMARY")
        print('='*70)

        comparison_df = pd.DataFrame({
            Path(name).name: {
                "Brier Score": results["brier_score"],
                "Calibration Error": results["mean_calibration_error"],
                "N Municipalities": results["n_municipalities"]
            }
            for name, results in all_results.items()
        }).T

        print("\n" + comparison_df.to_string())

        # Rank by Brier score
        comparison_df_sorted = comparison_df.sort_values("Brier Score")
        print(f"\n🏆 Ranking by Brier Score (lower is better):")
        for i, (name, row) in enumerate(comparison_df_sorted.iterrows(), 1):
            improvement = ""
            if i == 1 and len(comparison_df_sorted) > 1:
                second_best = comparison_df_sorted.iloc[1]["Brier Score"]
                diff = second_best - row["Brier Score"]
                pct = (diff / second_best) * 100
                improvement = f" ({diff:.6f} better, {pct:.2f}% improvement)"
            print(f"  {i}. {name}: {row['Brier Score']:.6f}{improvement}")

    elif len(all_results) == 1:
        print(f"\n✅ Single model evaluated. Run with multiple directories to compare.")


if __name__ == "__main__":
    main()
