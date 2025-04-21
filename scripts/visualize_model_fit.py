import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import json
import sys

# Add src directory to path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from src.data.dataset import ElectionDataset
except ImportError as e:
    print(f"Error importing ElectionDataset: {e}")
    print("Ensure the script is run from the project root or the src directory is in PYTHONPATH.")
    sys.exit(1)

def visualize_fit_per_election(model_run_dir: str, output_dir: str):
    """
    Visualizes model fit against observed polls and district results for each historical election.

    Args:
        model_run_dir: Path to the directory containing the model run results
                       (trace.zarr and model_config.json).
        output_dir: Directory to save the visualization plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving fit visualization plots to: {output_dir}")

    # --- Load Model Config --- 
    config_path = os.path.join(model_run_dir, "model_config.json")
    trace_path = os.path.join(model_run_dir, "trace.zarr")

    if not os.path.exists(config_path):
        print(f"Error: model_config.json not found in {model_run_dir}")
        return
    if not os.path.exists(trace_path):
        print(f"Error: trace.zarr not found in {model_run_dir}")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded model config from {config_path}")
    except Exception as e:
        print(f"Error loading model config: {e}")
        return

    # --- Load idata --- 
    try:
        print(f"Loading trace from {trace_path}...")
        idata = az.from_zarr(trace_path)
    except Exception as e:
        print(f"Error loading trace: {e}")
        return

    # --- Load Dataset (using config) --- 
    try:
        print("Loading dataset using model config...")
        # Use config values, provide defaults if keys are missing
        dataset = ElectionDataset(
            election_date=config.get('election_date', '2024-03-10'), # Use saved election date context
            baseline_timescales=config.get('baseline_timescales', [365]),
            election_timescales=config.get('election_timescales', [30]),
            test_cutoff=pd.Timedelta(config['cutoff_date']) if config.get('cutoff_date') else None
        )
    except Exception as e:
        print(f"Error initializing ElectionDataset: {e}")
        return

    # --- Extract Data and Predictions --- 
    print("Extracting observed data and predictions...")
    try:
        political_families = idata.posterior.coords["parties_complete"].values.tolist()
        
        # Observed Polls
        obs_polls_df = dataset.polls_train.copy()
        # Ensure sample size > 0
        obs_polls_df = obs_polls_df[obs_polls_df["sample_size"] > 0]
        for party in political_families:
             if party in obs_polls_df.columns and "sample_size" in obs_polls_df.columns:
                  obs_polls_df[f'{party}_share'] = obs_polls_df[party] / obs_polls_df["sample_size"]
             else:
                  print(f"Warning: Column '{party}' or 'sample_size' missing in polls_train. Cannot calculate share.")
                  obs_polls_df[f'{party}_share'] = 0 # Assign default
        # Add election date string for merging/filtering
        obs_polls_df['election_date_str'] = pd.to_datetime(obs_polls_df['election_date']).dt.strftime('%Y-%m-%d')

        # Predicted Polls (Mean)
        if "p_polls" not in idata.posterior:
             print("Warning: 'p_polls' not found in posterior. Skipping poll fit visualization.")
             pred_polls_df = pd.DataFrame()
        else:
            pred_polls_mean = idata.posterior["p_polls"].mean(dim=["chain", "draw"])
            # Create DataFrame with predictions, using the observation index from idata
            pred_polls_df = pd.DataFrame(
                pred_polls_mean.values,
                index=idata.posterior["observations"].values, # Use the coordinate values as index
                columns=political_families
            )
            pred_polls_df.index.name = 'observation_idx' # Name the index for clarity
            pred_polls_df = pred_polls_df.add_prefix('pred_') # Add prefix to predicted columns
            # Merge with observed polls on index
            # Use index from obs_polls_df that corresponds to idata observations coord
            obs_polls_df_indexed = obs_polls_df.set_index(idata.posterior["observations"].values)
            obs_polls_df_indexed.index.name = 'observation_idx'
            polls_merged = obs_polls_df_indexed.join(pred_polls_df, how='inner')
            print(f"Merged polls shape: {polls_merged.shape}")


        # Observed District Results
        obs_results_df = dataset.results_mult_district.copy()
        if obs_results_df.empty:
             print("Warning: results_mult_district is empty.")
        else:
            obs_results_df = obs_results_df[obs_results_df["sample_size"] > 0]
            for party in political_families:
                if party in obs_results_df.columns and "sample_size" in obs_results_df.columns:
                    obs_results_df[f'{party}_share'] = obs_results_df[party] / obs_results_df["sample_size"]
                else:
                     print(f"Warning: Column '{party}' or 'sample_size' missing in results_mult_district. Cannot calculate share.")
                     obs_results_df[f'{party}_share'] = 0
            obs_results_df['election_date_str'] = pd.to_datetime(obs_results_df['election_date']).dt.strftime('%Y-%m-%d')


        # Predicted District Results (Mean)
        p_results_var = "p_results_district" # Name used in the model
        if p_results_var not in idata.posterior:
             print(f"Warning: '{p_results_var}' not found in posterior. Skipping district result fit visualization.")
             pred_results_df = pd.DataFrame()
        else:
            pred_results_mean = idata.posterior[p_results_var].mean(dim=["chain", "draw"])
            # Check if the observed dimension exists
            if "elections_observed_district" not in idata.posterior[p_results_var].dims:
                 print(f"Error: Dimension 'elections_observed_district' not found in {p_results_var}. Cannot align results.")
                 pred_results_df = pd.DataFrame()
            else:
                result_indices = idata.posterior["elections_observed_district"].values
                # Create DataFrame, using the result indices coordinate
                pred_results_df = pd.DataFrame(
                    pred_results_mean.values,
                    index=result_indices, # Use the coordinate values as index
                    columns=political_families
                )
                pred_results_df.index.name = 'result_obs_idx' # Name the index
                pred_results_df = pred_results_df.add_prefix('pred_') # Add prefix

                # Merge with observed results on index
                # Assuming the index of obs_results_df corresponds to the 'elections_observed_district' coord values
                if obs_results_df.empty:
                     results_merged = pd.DataFrame()
                     print("Skipping merge for results as observed data is empty.")
                else:
                    obs_results_df_indexed = obs_results_df.set_index(result_indices)
                    obs_results_df_indexed.index.name = 'result_obs_idx'
                    results_merged = obs_results_df_indexed.join(pred_results_df, how='inner')
                    print(f"Merged results shape: {results_merged.shape}")

    except KeyError as e:
        print(f"Error accessing data/coordinates: {e}. Check variable names and idata structure.")
        return
    except Exception as e:
        print(f"Error during data extraction: {e}")
        return

    # --- Plotting per Election --- 
    # Get unique historical election dates from district results (most reliable source for actual past elections)
    if obs_results_df.empty:
         print("No historical district results to determine election dates for plotting.")
         historical_election_dates = []
    else:
        historical_election_dates = sorted(obs_results_df['election_date_str'].unique())

    if not historical_election_dates:
        print("No historical election dates found to plot.")
        return
        
    print(f"\nGenerating plots for elections: {historical_election_dates}")

    # Define colors for parties
    party_colors = plt.cm.get_cmap('tab10', len(political_families))
    color_map = {party: party_colors(i) for i, party in enumerate(political_families)}

    for election_dt_str in historical_election_dates:
        print(f"  Plotting for election: {election_dt_str}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        fig.suptitle(f'Model Fit vs. Observed Data - Election: {election_dt_str}', fontsize=16)

        # --- Polls Plot --- 
        ax = axes[0]
        ax.set_title('Polls Fit')
        has_poll_data = False
        if 'polls_merged' in locals() and not polls_merged.empty:
            polls_election = polls_merged[polls_merged['election_date_str'] == election_dt_str]
            if not polls_election.empty:
                has_poll_data = True
                for party in political_families:
                    obs_share_col = f'{party}_share'
                    pred_share_col = f'pred_{party}'
                    if obs_share_col in polls_election.columns and pred_share_col in polls_election.columns:
                        ax.scatter(polls_election[obs_share_col], polls_election[pred_share_col],
                                   label=party, alpha=0.6, color=color_map.get(party))

        if not has_poll_data:
             ax.text(0.5, 0.5, "No poll data for this cycle", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='1:1 Line')
        ax.set_xlabel('Observed Poll Share')
        ax.set_ylabel('Predicted Poll Share (Mean Posterior)')
        ax.set_xlim(0, max(0.5, ax.get_xlim()[1])) # Adjust axis limits based on data, minimum 0.5
        ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
        ax.grid(True, linestyle=':')

        # --- Results Plot --- 
        ax = axes[1]
        ax.set_title('District Results Fit')
        has_result_data = False
        if 'results_merged' in locals() and not results_merged.empty:
            results_election = results_merged[results_merged['election_date_str'] == election_dt_str]
            if not results_election.empty:
                has_result_data = True
                for party in political_families:
                    obs_share_col = f'{party}_share'
                    pred_share_col = f'pred_{party}'
                    if obs_share_col in results_election.columns and pred_share_col in results_election.columns:
                        ax.scatter(results_election[obs_share_col], results_election[pred_share_col],
                                   label=party, alpha=0.6, color=color_map.get(party))
        
        if not has_result_data:
             ax.text(0.5, 0.5, "No result data for this cycle", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='1:1 Line')
        ax.set_xlabel('Observed District Result Share')
        # ax.set_ylabel('Predicted District Result Share (Mean Posterior)') # Y-label shared
        ax.set_xlim(0, max(0.5, ax.get_xlim()[1])) # Adjust axis limits based on data, minimum 0.5
        ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
        ax.grid(True, linestyle=':')

        # Add legend to the side
        handles, labels = axes[0].get_legend_handles_labels()
        # Remove duplicate labels if any (e.g., 1:1 Line)
        unique_labels = dict(zip(labels, handles))
        fig.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(0.98, 0.5))
        plt.subplots_adjust(right=0.85) # Adjust layout to make space for legend

        # Save figure
        plot_filename = f"fit_vs_observed_polls_results_{election_dt_str}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"    Saved plot: {plot_path}")
        except Exception as e:
            print(f"    Error saving plot {plot_path}: {e}")
        plt.close(fig)

    print("\nFit visualization script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model fit against observed polls and results per election.")
    parser.add_argument(
        "model_run_dir",
        help="Path to the directory containing the model run results (trace.zarr and model_config.json)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save visualization plots. Defaults to a 'fit_visualizations' subdirectory within model_run_dir."
    )
    args = parser.parse_args()

    # Set default output directory if not provided
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.model_run_dir, "fit_visualizations")

    visualize_fit_per_election(args.model_run_dir, output_dir) 