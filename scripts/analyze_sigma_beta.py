import arviz as az
import numpy as np
import os
import pandas as pd
import argparse

def analyze_sigma_beta(model_run_dir: str):
    """
    Loads the trace from a model run directory, analyzes the posterior distribution
    of sigma_beta for each party, and prints summary statistics.

    Args:
        model_run_dir: Path to the directory containing the model run results
                       (specifically, the trace.zarr file).
    """
    trace_path = os.path.join(model_run_dir, "trace.zarr")

    if not os.path.exists(trace_path):
        print(f"Error: Trace file not found at {trace_path}")
        return
    else:
        try:
            print(f"Loading trace from {trace_path}...")
            idata = az.from_zarr(trace_path)

            if "posterior" in idata and "sigma_beta" in idata.posterior:
                sigma_beta_da = idata.posterior["sigma_beta"]

                # Check dimensions and coordinates
                if "parties_complete" not in sigma_beta_da.dims:
                    print("Error: 'parties_complete' dimension not found for sigma_beta.")
                    return

                # Calculate mean and std dev across chain and draw dimensions
                sigma_beta_mean = sigma_beta_da.mean(dim=["chain", "draw"])
                sigma_beta_std = sigma_beta_da.std(dim=["chain", "draw"])

                # Get party names
                parties = sigma_beta_da.coords["parties_complete"].values

                # Create a pandas DataFrame for easier sorting and display
                summary_df = pd.DataFrame({
                    'Party': parties,
                    'Mean': sigma_beta_mean.values,
                    'Std Dev': sigma_beta_std.values
                })

                # Sort by mean descending (largest sigma_beta first)
                summary_sorted_mean = summary_df.sort_values('Mean', ascending=False).reset_index(drop=True)

                print("\n--- Sigma Beta Posterior Summary (Sorted by Mean Descending) ---")
                print(summary_sorted_mean.to_string(index=False, float_format="%.3f"))

                # Sort by std dev descending (broadest distribution first)
                summary_sorted_std = summary_df.sort_values('Std Dev', ascending=False).reset_index(drop=True)

                print("\n--- Sigma Beta Posterior Summary (Sorted by Std Dev Descending) ---")
                print(summary_sorted_std.to_string(index=False, float_format="%.3f"))

            else:
                print("Error: 'sigma_beta' not found in the posterior group of the trace.")

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze sigma_beta posterior from election model trace.")
    parser.add_argument(
        "model_run_dir",
        help="Path to the directory containing the model run results (e.g., outputs/dynamic_gp_run_YYYYMMDD_HHMMSS)"
    )
    args = parser.parse_args()

    analyze_sigma_beta(args.model_run_dir) 