import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import argparse
import traceback # Import traceback for detailed error reporting

# Function to load trace
def load_latest_trace(base_output_dir="outputs"):
    """Loads the trace from the 'latest' symbolic link."""
    latest_dir = os.path.join(base_output_dir, "latest")
    if not os.path.exists(latest_dir) : # Check if path exists first
        raise FileNotFoundError(f"Latest model directory link/folder not found at {latest_dir}")

    # Resolve the actual path if it's a link
    if os.path.islink(latest_dir):
        actual_run_dir = os.path.realpath(latest_dir)
        print(f"Resolved 'latest' link to: {actual_run_dir}")
    else:
        # If it's not a link, assume it's the directory itself (e.g., if only one run exists)
        actual_run_dir = latest_dir
        print(f"Using directory directly: {actual_run_dir}")

    trace_path = os.path.join(actual_run_dir, "trace.zarr")

    if not os.path.exists(trace_path):
         raise FileNotFoundError(f"Trace file not found at {trace_path}")

    print(f"Loading trace from: {trace_path}")
    return az.from_zarr(trace_path)

def analyze_identifiability(trace, output_dir="."):
    """Analyzes potential identifiability issues between baseline and election effects."""

    if "party_baseline" not in trace.posterior or "election_party_baseline" not in trace.posterior:
        print("Required variables ('party_baseline', 'election_party_baseline') not found in trace.")
        return

    posterior = trace.posterior
    # Handle cases where coords might be missing or different
    parties = posterior.coords.get("parties_complete", None)
    elections = posterior.coords.get("elections", None)

    if parties is None or elections is None:
        print("Could not find 'parties_complete' or 'elections' coordinates in the trace.")
        return
        
    parties = parties.values
    elections = elections.values


    print("\n--- Analyzing Summary Statistics ---")
    summary = az.summary(trace, var_names=["party_baseline", "election_party_baseline"])
    print(summary)

    # Check standard deviations
    pb_sd = summary.loc[summary.index.str.contains("party_baseline", regex=False), "sd"]
    epb_sd = summary.loc[summary.index.str.contains("election_party_baseline", regex=False), "sd"]

    # Check if empty before calculating stats
    if pb_sd.empty or epb_sd.empty:
        print("Could not extract standard deviations for comparison.")
    else:
        print(f"\nParty Baseline SD stats: mean={pb_sd.mean():.3f}, std={pb_sd.std():.3f}, min={pb_sd.min():.3f}, max={pb_sd.max():.3f}")
        print(f"Election Party Baseline SD stats: mean={epb_sd.mean():.3f}, std={epb_sd.std():.3f}, min={epb_sd.min():.3f}, max={epb_sd.max():.3f}")

        if epb_sd.mean() > pb_sd.mean() * 1.5 : # Adjusted threshold slightly
             print("\nWARNING: Average SD of election-specific effects is notably larger than baseline SD.")
             print("This might indicate difficulty in separating these effects.")

    print("\n--- Analyzing Posterior Correlations ---")
    # Reshape data for easier correlation analysis
    # Flatten chains and draws
    try:
        pb_flat = posterior["party_baseline"].stack(sample=("chain", "draw")).values # Shape: (parties_complete, sample)
        # Stack epb_flat correctly
        epb_flat = posterior["election_party_baseline"].stack(sample=("chain", "draw")).values # Shape: (elections, parties_complete, sample)
    except Exception as e:
        print(f"Error reshaping data for correlation analysis: {e}")
        return

    correlations = {}
    n_elections = len(elections)
    n_parties = len(parties)
    
    # Adjust figure size dynamically
    fig_height = max(5, 2.5 * n_elections)
    fig_width = max(10, 2.5 * n_parties)
    fig, axes = plt.subplots(n_elections, n_parties, figsize=(fig_width, fig_height), squeeze=False) # Ensure axes is always 2D

    plot_idx = 0 # Linear index for easier assignment if needed, but use 2D indexing preferably

    print("\nCalculating and plotting correlations/pairs:")
    high_corr_found = False
    for i, election in enumerate(elections):
        election_corrs = []
        # print(f"\nCorrelations for Election: {election}") # Reduce verbosity inside loop
        for j, party in enumerate(parties):
            # Calculate correlation between party_baseline[party] and election_party_baseline[election, party]
            try:
                corr = np.corrcoef(pb_flat[j, :], epb_flat[i, j, :])[0, 1]
                correlations[(election, party)] = corr
                election_corrs.append(corr)
                # print(f"  Corr(pb_{party}, epb_{election},{party}): {corr:.3f}") # Reduce verbosity

                # Plot pair plot for this specific interaction
                ax = axes[i, j] # Use 2D indexing
                az.plot_pair(
                    trace,
                    var_names=["party_baseline", "election_party_baseline"],
                    coords={"elections": [election], "parties_complete": [party]},
                    kind="scatter", # Use scatter for direct visualization
                    scatter_kwargs={'alpha': 0.05}, # Lower alpha for dense plots
                    ax=ax,
                    show=False, # Prevent showing intermediate plots
                    marginal_kwargs={'plot_kwargs': {'color': 'black'}}, # Optional: styling
                    point_estimate_kwargs={'color': 'red'} # Optional: styling
                )
                # Remove default titles generated by plot_pair if needed
                # ax[0,0].set_title("") etc.
                ax.set_title(f"{election[:4]}, {party}\nCorr: {corr:.2f}", fontsize=10) # Shorter title

                # Optional: remove axis labels for inner plots to reduce clutter
                if i < n_elections - 1:
                    ax.set_xlabel("")
                if j > 0:
                    ax.set_ylabel("")

            except Exception as plot_err:
                 print(f"  Error processing pair for {election}, {party}: {plot_err}")
                 # Attempt to add text to the correct axis, but ensure it exists
                 try:
                     ax = axes[i, j] # Ensure ax is defined for this iteration
                     ax.text(0.5, 0.5, "Error", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                 except NameError: # If ax doesn't exist (e.g., loop index error)
                     print("   Cannot add error text to plot axis.")
                 # Continue to next plot

        avg_corr = np.mean(np.abs(election_corrs)) if election_corrs else 0
        # print(f"  Average absolute correlation for election {election}: {avg_corr:.3f}") # Reduce verbosity
        if avg_corr > 0.6: # Stricter threshold
            print(f"  WARNING: High average absolute correlation ({avg_corr:.3f}) for election {election}, potential identifiability issue.")
            high_corr_found = True

    fig.suptitle("Pair Plots: Party Baseline vs Election-Specific Baseline", fontsize=16, y=1.01) # Add main title
    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout tighter, leave space for suptitle if needed
    
    plot_filename = os.path.join(output_dir, "identifiability_pair_plots.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"\nSaved detailed pair plots to {plot_filename}")
    plt.close(fig) # Close the specific figure

    # Overall correlation summary
    if correlations:
        all_corrs = np.abs(list(correlations.values()))
        print(f"\nOverall Average Absolute Correlation: {np.mean(all_corrs):.3f}")
        print(f"Overall Max Absolute Correlation: {np.max(all_corrs):.3f}")

        if np.mean(all_corrs) > 0.5 or high_corr_found:
             print("\nWARNING: High overall or election-specific correlations suggest potential identifiability issues between party_baseline and election_party_baseline.")
             print("Consider simplifying the model or using stronger priors if appropriate.")
    else:
        print("\nCould not calculate overall correlations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for identifiability issues between party_baseline and election_party_baseline in the election model.")
    parser.add_argument("--output-dir", default="outputs", help="Base directory containing the 'latest' model run link or directory.")
    parser.add_argument("--plot-dir", default=".", help="Directory to save the output plots.")
    args = parser.parse_args()

    # Ensure plot directory exists
    os.makedirs(args.plot_dir, exist_ok=True)

    try:
        trace = load_latest_trace(args.output_dir)
        analyze_identifiability(trace, args.plot_dir)
        print("\nAnalysis complete.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model has been run and 'outputs/latest' points to a valid run directory.")
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        traceback.print_exc() 