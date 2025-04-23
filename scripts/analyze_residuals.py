"""
Analyzes residuals (Observed Share - Predicted Share) for district results
from a saved model run.

Loads InferenceData and the corresponding ElectionDataset to compare observed
results against the model's posterior predictive mean.
"""

import os
import argparse
import arviz as az
import numpy as np
import pandas as pd
import json

# Assuming the script is run from the root 'models' directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.data.dataset import ElectionDataset
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

def analyze_district_residuals(idata: az.InferenceData, dataset: ElectionDataset, parties_of_interest: list = ['PAN', 'L', 'IL', 'PS']):
    """
    Calculates and analyzes district-level residuals.

    Args:
        idata: The loaded InferenceData object.
        dataset: The corresponding initialized ElectionDataset.
        parties_of_interest: List of party names to focus on.
    """
    if "posterior" not in idata:
        print("Error: Posterior group not found in InferenceData.")
        return
    if "p_results_district" not in idata.posterior:
        print("Error: Predicted district probabilities ('p_results_district') not found in posterior.")
        return
    if not hasattr(dataset, 'results_mult_district') or dataset.results_mult_district is None or dataset.results_mult_district.empty:
        print("Error: District results (results_mult_district) not found or empty in the dataset.")
        return

    posterior = idata.posterior
    observed_results_df = dataset.results_mult_district.copy()

    # --- Get Predicted Probabilities (Posterior Mean) --- 
    # Shape: (elections_observed_district, parties_complete)
    pred_probs_mean = posterior["p_results_district"].mean(dim=["chain", "draw"]).values

    # --- Get Observed Shares --- 
    # Ensure observed results align with the model's internal ordering/indices
    # The `elections_observed_district` dimension in the trace corresponds
    # to the rows of `results_mult_district` used during model fitting.
    obs_counts = observed_results_df[dataset.political_families].to_numpy()
    obs_n = observed_results_df["sample_size"].to_numpy()

    if pred_probs_mean.shape[0] != obs_counts.shape[0]:
        print(f"Error: Shape mismatch between predicted probabilities ({pred_probs_mean.shape[0]} rows) and observed counts ({obs_counts.shape[0]} rows).")
        return
    if pred_probs_mean.shape[1] != len(dataset.political_families):
         print(f"Error: Shape mismatch between predicted probabilities ({pred_probs_mean.shape[1]} parties) and configured political families ({len(dataset.political_families)}).")
         return

    # Calculate observed shares, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        obs_shares = obs_counts / obs_n[:, np.newaxis]
    obs_shares = np.nan_to_num(obs_shares) # Replace NaNs resulting from division by zero with 0

    # --- Calculate Residuals --- 
    residuals = obs_shares - pred_probs_mean

    # Convert residuals to a DataFrame for easier analysis, adding back district/year info
    residuals_df = pd.DataFrame(residuals, columns=dataset.political_families)
    # Add identifying columns from the original observed_results_df
    # Ensure index alignment if necessary (usually okay if row order wasn't changed)
    residuals_df['Circulo'] = observed_results_df['Circulo'].values
    residuals_df['election_date'] = observed_results_df['election_date'].values
    residuals_df['year'] = observed_results_df['election_date'].dt.year

    print("\n--- Analyzing District Residuals (Observed Share - Predicted Share) ---")

    for party in parties_of_interest:
        if party not in residuals_df.columns:
            print(f"\nParty '{party}' not found in results. Skipping.")
            continue

        print(f"\n--- Residuals for {party} ---")
        party_residuals = residuals_df[party]
        summary = party_residuals.describe(percentiles=[.05, .25, .5, .75, .95])
        print(summary.to_string())

        # Optional: Analyze residuals by year or district if needed
        # print("\n  Mean Residual by Year:")
        # print(residuals_df.groupby('year')[party].mean().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze district residuals from a saved model run.")
    parser.add_argument(
        "--run-dir",
        default="outputs/dynamic_gp_run_20250420_012946",
        help="Path to the model run directory containing trace.zarr and model_config.json",
    )
    parser.add_argument(
        "--parties",
        nargs='+',
        default=["PAN", "L", "IL", "PS", "AD", "BE", "CDU", "CH"], # Analyze all by default
        help="List of party acronyms to analyze (e.g., PAN L IL PS AD)",
    )

    args = parser.parse_args()

    trace_path = os.path.join(args.run_dir, "trace.zarr")
    config_path = os.path.join(args.run_dir, "model_config.json")

    if not os.path.exists(args.run_dir):
        print(f"Error: Model run directory not found at {args.run_dir}")
        sys.exit(1)
    if not os.path.exists(trace_path):
        print(f"Error: Trace file not found at {trace_path}")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"Error: Model config file not found at {config_path}")
        sys.exit(1)

    # Load config to get election date for dataset
    print(f"Loading model config from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        election_date = config.get('election_date')
        if not election_date:
             raise ValueError("Election date not found in config.")
        # Get timescales from config if available, otherwise use defaults
        # (These might not be strictly necessary if only loading results,
        # but good practice to match the run)
        baseline_timescales = config.get('baseline_timescales', [365*4])
        election_timescales = config.get('election_timescales', [90])
    except Exception as e:
        print(f"Error loading or parsing model config: {e}")
        sys.exit(1)

    # Load dataset corresponding to the model run
    print(f"Loading dataset with election date context: {election_date}")
    try:
        # Ensure timescales are lists
        if not isinstance(baseline_timescales, list): baseline_timescales = [baseline_timescales]
        if not isinstance(election_timescales, list): election_timescales = [election_timescales]
        dataset = ElectionDataset(election_date=election_date,
                                baseline_timescales=baseline_timescales,
                                election_timescales=election_timescales)
    except Exception as e:
        print(f"Error initializing ElectionDataset: {e}")
        sys.exit(1)

    # Load trace
    print(f"Loading trace from: {trace_path}")
    try:
        idata = az.from_zarr(trace_path)
    except Exception as e:
        print(f"Error loading trace file: {e}")
        sys.exit(1)

    # Run the analysis
    analyze_district_residuals(idata, dataset, args.parties)

    print("\nAnalysis complete.") 