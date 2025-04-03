import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import traceback
import seaborn as sns # Using seaborn for potentially better aesthetics

# Assuming ElectionDataset is accessible via src
try:
    from src.data.dataset import ElectionDataset
    from src.config import DEFAULT_BASELINE_TIMESCALE, DEFAULT_ELECTION_TIMESCALES
except ImportError:
    print("Error: Could not import ElectionDataset or config. Ensure PYTHONPATH is set correctly.")
    # As a fallback, try a relative import if run as a script from the root
    try:
        from ..data.dataset import ElectionDataset
        from ..config import DEFAULT_BASELINE_TIMESCALE, DEFAULT_ELECTION_TIMESCALES
    except ImportError:
        ElectionDataset = None # Indicate failure
        DEFAULT_BASELINE_TIMESCALE = [180] # Provide defaults
        DEFAULT_ELECTION_TIMESCALES = [15]


def plot_raw_trajectories(dataset, output_plot_dir="."):
    """Plots the raw poll share trajectories aligned by countdown for each party and election."""

    if dataset is None:
         print("ElectionDataset could not be loaded.")
         return

    # Use polls_train as it's used for model fitting
    polls = dataset.polls_train.copy()

    if 'countdown' not in polls.columns or 'election_date' not in polls.columns:
        print("Required columns ('countdown', 'election_date') not found in polls data.")
        return

    parties = dataset.political_families
    # Use historical election dates defined in the dataset
    historical_elections = sorted([pd.to_datetime(d) for d in dataset.historical_election_dates])

    print("\n--- Plotting Raw Poll Trajectories by Election Cycle ---")

    # Ensure the output directory exists
    trajectory_plot_path = os.path.join(output_plot_dir, "raw_campaign_trajectories")
    os.makedirs(trajectory_plot_path, exist_ok=True)
    print(f"Saving raw trajectory plots to: {trajectory_plot_path}")

    # Use a color palette
    palette = sns.color_palette("husl", len(historical_elections))

    for party in parties:
        if party not in polls.columns:
            print(f"  Skipping party '{party}' - not found in poll data columns.")
            continue

        print(f"  Plotting for party: {party}")
        fig, ax = plt.subplots(figsize=(12, 7))
        legend_handles = []

        # Calculate poll share (handle potential division by zero)
        polls[f'{party}_share'] = polls.apply(
            lambda row: row[party] / row['sample_size'] if row['sample_size'] > 0 else np.nan,
            axis=1
        )
        # Drop rows where share could not be calculated
        party_polls = polls.dropna(subset=[f'{party}_share'])


        for i, election_date in enumerate(historical_elections):
            # Filter polls for this specific election cycle
            cycle_polls = party_polls[party_polls['election_date'] == election_date].copy()

            if cycle_polls.empty:
                # print(f"    No polls found for {party} in {election_date.strftime('%Y-%m-%d')} cycle.")
                continue

            # Sort by countdown for cleaner line plot (optional but good practice)
            cycle_polls = cycle_polls.sort_values('countdown', ascending=False)

            # Use scatter plot to show individual polls
            scatter = ax.scatter(
                cycle_polls['countdown'],
                cycle_polls[f'{party}_share'],
                label=election_date.strftime('%Y'), # Use Year as label
                color=palette[i],
                alpha=0.6,
                s=20 # Smaller points
            )
            legend_handles.append(scatter)

            # Optional: Add a smoothed line (e.g., rolling average or lowess) per cycle
            # Example using rolling average:
            # if len(cycle_polls) > 5: # Need enough points for rolling window
            #     smoothed = cycle_polls.set_index('countdown')[f'{party}_share'].rolling(window=5, center=True, min_periods=1).mean()
            #     ax.plot(smoothed.index, smoothed.values, color=palette[i], alpha=0.9)


        ax.set_xlabel("Days Before Election (Countdown)")
        ax.set_ylabel("Observed Poll Share")
        ax.set_title(f"Raw Poll Trajectories for '{party}' by Election Cycle")
        # Add legend outside the loop using handles
        if legend_handles:
            ax.legend(handles=legend_handles, title="Election Year")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(bottom=0) # Share cannot be negative
        # Invert x-axis to show time progressing towards election day (optional)
        # ax.invert_xaxis()

        # Save the figure
        plot_filename = os.path.join(trajectory_plot_path, f"raw_trajectory_{party}.png")
        try:
            plt.savefig(plot_filename, bbox_inches='tight')
        except Exception as save_err:
             print(f"    Error saving plot {plot_filename}: {save_err}")
        plt.close(fig)

    print("\nFinished plotting raw campaign trajectories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize raw poll trajectories by election cycle.")
    # Use a default future date for dataset loading context if not specified
    default_election_date = (pd.Timestamp.now().replace(month=1, day=1) + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    parser.add_argument("--election-date", default=default_election_date, help="Reference election date for loading dataset context.")
    parser.add_argument("--plot-dir", default="outputs/latest", help="Directory to save the output plots (usually within the 'latest' run).")
    args = parser.parse_args()

    # Ensure plot directory exists
    os.makedirs(args.plot_dir, exist_ok=True)

    if ElectionDataset is None:
         print("Failed to import ElectionDataset. Cannot proceed.")
         exit(1)

    try:
        print(f"Loading dataset with reference date: {args.election_date}")
        # Load dataset using reference date and default timescales
        dataset = ElectionDataset(
            election_date=args.election_date,
            baseline_timescales=DEFAULT_BASELINE_TIMESCALE,
            election_timescales=DEFAULT_ELECTION_TIMESCALES
        )
        plot_raw_trajectories(dataset, args.plot_dir)
        print("\nAnalysis complete.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure data files are available.")
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        traceback.print_exc() 