"""
Analyzes intermediate swing calculation variables saved in a model trace.

Loads an InferenceData object and calculates summary statistics for the debug variables:
- debug_national_dev_results
- debug_beta_term
- debug_dynamic_adjustment_term
- debug_base_offset_term
- debug_total_district_adjustment

Focuses on the statistics for specific parties like PAN to understand magnitudes.
"""

import os
import argparse
import arviz as az
import numpy as np
import pandas as pd

# Assuming the script is run from the root 'models' directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # We might need ElectionDataset later if we want to map indices back to names
    # from src.data.dataset import ElectionDataset
    pass
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

def analyze_swing_terms(idata: az.InferenceData, parties_of_interest: list = ['PAN', 'L', 'IL', 'PS']):
    """
    Analyzes the debug swing term variables in the InferenceData.

    Args:
        idata: The loaded InferenceData object.
        parties_of_interest: List of party names to focus on.
    """
    if "posterior" not in idata:
        print("Error: Posterior group not found in InferenceData.")
        return

    posterior = idata.posterior
    debug_vars = [
        "debug_national_dev_results",
        "debug_beta_term",
        "debug_dynamic_adjustment_term",
        "debug_base_offset_term",
        "debug_total_district_adjustment"
    ]

    # Get coordinate values for mapping
    try:
        party_coords = posterior["parties_complete"].values.tolist()
        obs_district_coords = posterior["elections_observed_district"].values # These are likely just indices 0-99
    except KeyError as e:
        print(f"Error: Missing coordinate in posterior data: {e}")
        return

    print("\n--- Analyzing Swing Term Variables ---")

    for var_name in debug_vars:
        if var_name not in posterior:
            print(f"\nVariable '{var_name}' not found in posterior. Skipping.")
            continue

        print(f"\n--- Analysis for {var_name} ---")
        data_da = posterior[var_name]
        print(f"  Shape: {data_da.shape} (chain, draw, *dims)")
        print(f"  Dims: {data_da.dims}")

        # --- Overall Summary --- 
        # Flatten across chains, draws, and non-party dimensions for a general idea
        if data_da.ndim > 2: # Check if it has dimensions beyond chain/draw
             # Identify non-party dimensions to flatten over
             non_party_dims = [d for d in data_da.dims if d not in ('chain', 'draw', 'parties_complete')] # Handle potential variations in dim order
             # Flatten all samples and non-party dimensions
             flattened_data = data_da.stack(sample=('chain', 'draw', *non_party_dims)).values.flatten()
             # Need to handle potential NaNs if calculations resulted in them
             flattened_data = flattened_data[~np.isnan(flattened_data)]
             if flattened_data.size > 0:
                  overall_summary = pd.Series(flattened_data).describe(percentiles=[.05, .25, .5, .75, .95])
                  print("\n  Overall Distribution (flattened across samples, districts, parties):")
                  print(overall_summary.to_string())
             else:
                  print("\n  Overall Distribution: No valid data points found after flattening.")
        else: # Should not happen for these vars, but handle just in case
             print("\n  Overall Distribution: Cannot flatten variable with 2 or fewer dimensions.")

        # --- Per Party Summary --- 
        print("\n  Distribution by Party (flattened across samples and districts):")
        for party in parties_of_interest:
            if party not in party_coords:
                print(f"    Party '{party}' not found in coordinates. Skipping.")
                continue

            try:
                # Select data for the party
                if "parties_complete" in data_da.dims:
                     party_da = data_da.sel(parties_complete=party)
                     # Flatten across chain, draw, and other non-party dims
                     non_party_dims_party = [d for d in party_da.dims if d not in ('chain', 'draw')] # Find remaining dims
                     flattened_party_data = party_da.stack(sample=('chain', 'draw', *non_party_dims_party)).values.flatten()
                     # Handle potential NaNs
                     flattened_party_data = flattened_party_data[~np.isnan(flattened_party_data)]

                     if flattened_party_data.size > 0:
                          party_summary = pd.Series(flattened_party_data).describe(percentiles=[.05, .25, .5, .75, .95])
                          print(f"    --- {party} ---")
                          print(party_summary.to_string())
                     else:
                          print(f"    --- {party} ---: No valid data points found.")
                else:
                     print(f"    Party dimension 'parties_complete' not found for variable {var_name}. Cannot summarize by party.")
                     break # Stop trying parties for this var

            except Exception as e:
                 print(f"    Error processing party '{party}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze intermediate swing terms from a saved trace.")
    parser.add_argument(
        "--trace-path",
        default="model_outputs/dynamic_gp_poll_bias_beta_3k_3k/trace.zarr",
        help="Path to the saved trace file (trace.zarr)",
    )
    parser.add_argument(
        "--parties",
        nargs='+',
        default=["PAN", "L", "IL", "PS", "AD"], # Default parties to focus on
        help="List of party acronyms to analyze in detail (e.g., PAN L IL PS AD)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.trace_path):
        print(f"Error: Trace file not found at {args.trace_path}")
        sys.exit(1)

    print(f"Loading trace from: {args.trace_path}")
    try:
        idata = az.from_zarr(args.trace_path)
    except Exception as e:
        print(f"Error loading trace file: {e}")
        sys.exit(1)

    # Run the analysis
    analyze_swing_terms(idata, args.parties)

    print("\nAnalysis complete.") 