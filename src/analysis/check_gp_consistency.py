import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import traceback

# Function to load trace (reusing from previous script, could be refactored)
def load_latest_trace(base_output_dir="outputs"):
    """Loads the trace from the 'latest' symbolic link."""
    latest_dir = os.path.join(base_output_dir, "latest")
    if not os.path.exists(latest_dir):
        raise FileNotFoundError(f"Latest model directory link/folder not found at {latest_dir}")
    if os.path.islink(latest_dir):
        actual_run_dir = os.path.realpath(latest_dir)
        print(f"Resolved 'latest' link to: {actual_run_dir}")
    else:
        actual_run_dir = latest_dir
        print(f"Using directory directly: {actual_run_dir}")
    trace_path = os.path.join(actual_run_dir, "trace.zarr")
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file not found at {trace_path}")
    print(f"Loading trace from: {trace_path}")
    return az.from_zarr(trace_path)

def plot_gp_time_effects(trace, output_plot_dir="."):
    """Plots the estimated party_time_effect GP for each party across elections."""

    if "party_time_effect" not in trace.posterior:
        print("Variable 'party_time_effect' not found in trace.")
        return

    posterior = trace.posterior
    # Get coordinates safely
    parties = posterior.coords.get("parties_complete", None)
    elections = posterior.coords.get("elections", None)
    countdown = posterior.coords.get("countdown", None)

    if parties is None or elections is None or countdown is None:
        print("Could not find required coordinates ('parties_complete', 'elections', 'countdown') in the trace.")
        return

    parties = parties.values
    elections = elections.values
    countdown = countdown.values # Should be the numerical countdown days

    # Get election mapping from dataset (assuming standard structure)
    # This requires loading the dataset config or having it accessible
    # As a fallback, we'll just use the election dates from coords
    try:
        # Attempt to map election index (used in coords) to actual dates for plotting
        # This relies on the order being consistent or having metadata
        # Using dates directly if they are the coordinate values
        election_labels = [str(e)[:10] for e in elections] # Convert datetime64 to string if needed
        print(f"Using election labels: {election_labels}")
    except Exception:
        print("Warning: Could not properly format election labels from coordinates.")
        election_labels = [f"Election_{i}" for i in range(len(elections))]


    print("\n--- Plotting GP Time Effects Across Elections ---")

    # Ensure the output directory for these specific plots exists
    gp_plot_path = os.path.join(output_plot_dir, "gp_consistency_plots")
    os.makedirs(gp_plot_path, exist_ok=True)
    print(f"Saving GP consistency plots to: {gp_plot_path}")

    # Extract the relevant dataarray
    gp_effect_da = posterior["party_time_effect"]

    for j, party in enumerate(parties):
        print(f"  Plotting for party: {party}")
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, election_label in enumerate(election_labels):
            try:
                # Select data for the specific party and election index
                # Mean across samples (chain, draw)
                mean_effect = gp_effect_da.sel(parties_complete=party, elections=elections[i]).mean(dim=["chain", "draw"]).values

                # Plot against countdown days
                ax.plot(countdown, mean_effect, label=f"{election_label}", alpha=0.8)

            except Exception as e:
                print(f"    Error plotting election {election_label} for party {party}: {e}")

        ax.set_xlabel("Days Before Election (Countdown)")
        ax.set_ylabel("Mean Estimated Time Effect (Logit Scale)")
        ax.set_title(f"Party Time Effect (GP) for '{party}' Across Elections")
        ax.legend(title="Election")
        ax.grid(True, linestyle='--', alpha=0.5)

        # Save the figure
        plot_filename = os.path.join(gp_plot_path, f"gp_effect_{party}.png")
        try:
            plt.savefig(plot_filename, bbox_inches='tight')
        except Exception as save_err:
             print(f"    Error saving plot {plot_filename}: {save_err}")
        plt.close(fig)

    print("\nFinished plotting GP time effects.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check consistency of GP time effects across elections.")
    parser.add_argument("--output-dir", default="outputs", help="Base directory containing the 'latest' model run link or directory.")
    parser.add_argument("--plot-dir", default=".", help="Directory to save the output plots.")
    args = parser.parse_args()

    # Ensure plot directory exists
    os.makedirs(args.plot_dir, exist_ok=True)

    try:
        trace = load_latest_trace(args.output_dir)
        plot_gp_time_effects(trace, args.plot_dir)
        print("\nAnalysis complete.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model has been run and 'outputs/latest' points to a valid run directory.")
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        traceback.print_exc() 