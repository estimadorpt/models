import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data.dataset import ElectionDataset
import os
import numpy as np
import xarray as xr
import arviz as az
from typing import TYPE_CHECKING, List
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as mtick
if TYPE_CHECKING:
    from src.models.elections_facade import ElectionsFacade
from src.models.dynamic_gp_election_model import DynamicGPElectionModel # Import the dynamic model class

def plot_election_data(dataset: ElectionDataset):
    """
    Visualizes the poll data and election results from an ElectionDataset object.

    Generates two types of plots:
    1. All polls and results over the entire time series.
    2. Individual plots for each election cycle.
    """
    
    polls_df = dataset.polls.copy()
    results_df = dataset.results_mult.copy()
    political_families = dataset.political_families
    historical_election_dates_dt = pd.to_datetime(dataset.historical_election_dates)
    
    # Find the date of the most recent historical election with results
    last_historical_election_date = historical_election_dates_dt.max() if not historical_election_dates_dt.empty else None

    # Convert results counts back to percentages for plotting
    results_df_pct = results_df.copy()
    total_votes = results_df_pct[political_families].sum(axis=1)
    for party in political_families:
        # Avoid division by zero if total_votes is 0 for some reason
        results_df_pct[party] = results_df_pct.apply(
            lambda row: row[party] / row['sample_size'] if row['sample_size'] > 0 else 0, 
            axis=1
        )

    # Ensure poll percentages are also valid (0-1 range)
    for party in political_families:
        polls_df[party] = polls_df[party].clip(0, 1)

    # --- Plot 1: All data over time ---
    plt.figure(figsize=(18, 10))

    # Melt the polls data for easier plotting with seaborn
    polls_melt = polls_df.melt(
        id_vars=['date', 'pollster', 'election_date'], # Keep election_date for filtering
        value_vars=political_families,
        var_name='party',
        value_name='percentage'
    )

    # Melt the results data
    results_melt = results_df_pct.melt(
        id_vars=['date'], # Results only have their own date
        value_vars=political_families,
        var_name='party',
        value_name='percentage'
    )

    # Separate historical polls and future polls
    historical_polls_melt = polls_melt[polls_melt['date'] <= last_historical_election_date] if last_historical_election_date else polls_melt
    future_polls_melt = polls_melt[polls_melt['date'] > last_historical_election_date] if last_historical_election_date else pd.DataFrame()

    # Plot historical polls
    # Use scatterplot instead of lineplot, avoid label conflict
    sns.scatterplot(data=historical_polls_melt, x='date', y='percentage', hue='party', alpha=0.3, marker='o', s=15, legend='auto') # Let Seaborn handle initial legend

    # Plot future polls (if any) with a different marker
    if not future_polls_melt.empty:
        print(f"Plotting {len(future_polls_melt['date'].unique())} future poll dates (after {last_historical_election_date.date()})")
        # Don't add duplicate legend entries for parties
        sns.scatterplot(data=future_polls_melt, x='date', y='percentage', hue='party', alpha=0.6, marker='s', s=25, legend=False)

    # Plot election results
    # Don't add duplicate legend entries for parties
    sns.scatterplot(data=results_melt, x='date', y='percentage', hue='party', s=250, marker='X', edgecolor='black', zorder=5, legend=False)

    # Manually create legend handles for clarity
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter out any potential "_nolegend_" labels if they sneak in
    party_handles = [h for i, h in enumerate(handles) if labels[i] in political_families]
    party_labels = [l for l in labels if l in political_families]
    
    legend_elements = []
    from matplotlib.lines import Line2D
    if not historical_polls_melt.empty:
        legend_elements.append(Line2D([0], [0], marker='o', color='grey', linestyle='', markersize=5, alpha=0.5, label='Historical Poll'))
    if not future_polls_melt.empty:
        legend_elements.append(Line2D([0], [0], marker='s', color='grey', linestyle='', markersize=5, alpha=0.8, label='Future Poll'))
    if not results_melt.empty:
        legend_elements.append(Line2D([0], [0], marker='X', color='grey', linestyle='', markersize=8, markeredgecolor='black', label='Election Result'))

    # Combine party handles and custom handles
    all_handles = party_handles + legend_elements
    all_labels = party_labels + [h.get_label() for h in legend_elements]

    plt.title('All Polls and Election Results Over Time')
    plt.xlabel('Date')
    plt.ylabel('Vote Percentage')
    # Use the combined handles and labels for the legend
    plt.legend(handles=all_handles, labels=all_labels, title='Party / Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.show()

    # --- Plot 2: Individual election cycles ---
    election_dates = sorted(polls_df['election_date'].unique())
    
    print(f"Generating plots for {len(election_dates)} election cycles...")

    for election_date in election_dates:
        election_date_dt = pd.to_datetime(election_date)
        
        # Filter polls for the current cycle (e.g., 1 year before)
        start_date = election_date_dt - pd.Timedelta(days=365*2) # Look back 2 years
        cycle_polls = polls_df[
            (polls_df['election_date'] == election_date_dt) & 
            (polls_df['date'] >= start_date) &
            (polls_df['date'] <= election_date_dt)
        ].copy()
        
        # Get the specific result for this election
        cycle_result_pct = results_df_pct[results_df_pct['date'] == election_date_dt]

        if cycle_polls.empty and cycle_result_pct.empty:
            print(f"Skipping {election_date_dt.date()}: No polls or results found.")
            continue
            
        print(f"Plotting cycle for election: {election_date_dt.date()}")

        plt.figure(figsize=(15, 8))
        
        # Melt cycle polls
        cycle_polls_melt = cycle_polls.melt(
            id_vars=['date', 'pollster'], 
            value_vars=political_families, 
            var_name='party', 
            value_name='percentage'
        )
        
        # Melt cycle result
        cycle_result_melt = cycle_result_pct.melt(
            id_vars=['date'], 
            value_vars=political_families, 
            var_name='party', 
            value_name='percentage'
        )

        # Plot polls for the cycle
        sns.lineplot(data=cycle_polls_melt, x='date', y='percentage', hue='party', style='pollster', alpha=0.6, marker='o', linestyle='--', markersize=5)
        
        # Plot the final election result
        if not cycle_result_melt.empty:
             sns.scatterplot(data=cycle_result_melt, x='date', y='percentage', hue='party', s=250, marker='X', edgecolor='black', zorder=5, legend=False) # legend=False to avoid duplicate legend items
       
        plt.title(f'Polls Leading up to {election_date_dt.date()} Election')
        plt.xlabel('Poll Date')
        plt.ylabel('Vote Percentage')
        plt.axvline(election_date_dt, color='r', linestyle=':', linewidth=2, label=f'Election Day ({election_date_dt.date()})')
        plt.legend(title='Party/Pollster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

def plot_latent_popularity_vs_polls(elections_model, output_dir, include_target_date=True):
    """
    Plots the latent popularity mean and HDI against observed polls over time.
    Also plots the daily average of the mean noisy popularity (latent + house effect).

    Args:
        elections_model: The fitted ElectionsFacade instance.
        output_dir: Directory to save the plot.
        include_target_date: Whether to include the target election date as a vertical line.
    """
    if elections_model.trace is None:
        print("Trace not found. Skipping latent popularity plot.")
        return

    trace = elections_model.trace
    dataset = elections_model.dataset
    model_instance = elections_model.model_instance # Get the specific model instance
    polls_train = dataset.polls_train
    is_dynamic_model = isinstance(model_instance, DynamicGPElectionModel)
    print(f"DEBUG plot_latent_pop: is_dynamic_model = {is_dynamic_model}")

    # Determine which latent variable and time coordinate to use
    if is_dynamic_model:
        latent_var_name = "latent_popularity_calendar_trajectory"
        time_coord_name = "calendar_time"
        latent_time_values = pd.to_datetime(trace.posterior[time_coord_name].values)
    else: # Static model
        latent_var_name = "latent_popularity_t"
        time_coord_name = "date"
        # Ensure date coordinate is datetime
        if time_coord_name in trace.posterior.coords:
             latent_time_values = pd.to_datetime(trace.posterior[time_coord_name].values)
        else:
             print(f"Warning: Coordinate '{time_coord_name}' not found in posterior for static model. Cannot plot latent trajectory.")
             return

    if latent_var_name not in trace.posterior:
        print(f"Latent variable '{latent_var_name}' not found in trace. Skipping plot.")
        return

    latent_pop = trace.posterior[latent_var_name]
    latent_mean = latent_pop.mean(dim=["chain", "draw"])
    # Call az.hdi(), expecting an xarray.Dataset
    hdi_dataset = az.hdi(trace.posterior, var_names=[latent_var_name])
    # Access the DataArray within the Dataset using the variable name
    latent_hdi_da = hdi_dataset[latent_var_name]

    # <<< --- Add Debugging Prints Here --- >>>
    print("\n--- Debugging HDI DataArray --- ")
    print(f"Type: {type(latent_hdi_da)}")
    print(f"Dims: {latent_hdi_da.dims}")
    print(f"Coords: {latent_hdi_da.coords}")
    print(f"Shape: {latent_hdi_da.shape}")
    # Check if 'hdi' dimension exists and its coordinates
    if 'hdi' in latent_hdi_da.dims:
        print(f"HDI dim coordinates: {latent_hdi_da['hdi'].values}")
    else:
        # Find potential dimension with size 2
        potential_hdi_dims = [dim for dim, size in latent_hdi_da.sizes.items() if size == 2]
        print(f"Potential HDI dims (size 2): {potential_hdi_dims}")
        for dim in potential_hdi_dims:
            try:
                print(f"  Coords for dim '{dim}': {latent_hdi_da[dim].values}")
            except Exception as e:
                print(f"  Could not get coords for dim '{dim}': {e}")
    print("--- End Debugging HDI --- \n")
    # <<< --- End Debugging Prints --- >>>

    # --- Calculate Daily Average Noisy Popularity ---
    noisy_pop_daily_mean = None
    noisy_pop_daily_dates = None
    if "noisy_popularity_polls" in trace.posterior:
        try:
            noisy_pop = trace.posterior["noisy_popularity_polls"]
            # Calculate mean across samples
            noisy_pop_mean_obs = noisy_pop.mean(dim=["chain", "draw"]) # Shape (observations, parties)

            # Create a pandas Series for dates indexed by observation coord
            # Ensure observation coordinates match polls_train index
            obs_coords = noisy_pop_mean_obs['observations'].values
            if not np.array_equal(obs_coords, polls_train.index.values):
                 # Attempt to reindex if coords are just integers 0..N-1
                 if np.array_equal(obs_coords, np.arange(len(polls_train))):
                     print("Attempting to align observation coords with polls_train index.")
                     poll_dates = pd.to_datetime(polls_train['date']).values
                 else:
                      raise ValueError("Observation coordinates in trace do not match polls_train index.")
            else:
                 poll_dates = pd.to_datetime(polls_train.loc[obs_coords, 'date']).values

            # Add dates as a coordinate to the xarray DataArray
            noisy_pop_mean_obs = noisy_pop_mean_obs.assign_coords(date=("observations", poll_dates))

            # Group by date and average
            # Use dropna=False if needed, handle potential NaT dates if any
            noisy_pop_daily_mean_xr = noisy_pop_mean_obs.groupby("date").mean()
            noisy_pop_daily_dates = noisy_pop_daily_mean_xr['date'].values
            noisy_pop_daily_mean = noisy_pop_daily_mean_xr.values # Numpy array (days, parties)
            print("Successfully calculated daily average noisy popularity.")
        except Exception as e:
            print(f"Warning: Could not calculate daily average noisy popularity: {e}")
            noisy_pop_daily_mean = None
            noisy_pop_daily_dates = None
    else:
        print("Info: 'noisy_popularity_polls' not found in trace, skipping noisy popularity line.")


    # --- Plotting ---
    parties = dataset.political_families
    n_parties = len(parties)
    n_cols = 3
    n_rows = (n_parties + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    # --- Inspect polls_train before melt ---
    print("\n--- Describing polls_train[parties] before melt ---")
    print(polls_train[parties].describe())
    print("--------------------------------------------------")
    # --- End Inspection ---

    polls_melt = polls_train.melt(
        id_vars=['date', 'pollster', 'sample_size'],
        value_vars=parties,
        var_name='party',
        value_name='vote_count'
    )
    polls_melt['date'] = pd.to_datetime(polls_melt['date'])

    for i, party in enumerate(parties):
        ax = axes[i]

        # Plot Latent Popularity Mean and HDI (Scale 0-1 data by 100)
        latent_mean_party = latent_mean.sel(parties_complete=party)
        ax.plot(latent_time_values, latent_mean_party * 100, label="Latent Popularity (Mean)")

        # Use the DataArray accessed above (latent_hdi_da)
        hdi_coords = latent_hdi_da['hdi'].values
        lower_coord = hdi_coords[0]
        upper_coord = hdi_coords[1]
        ax.fill_between(
            latent_time_values,
            latent_hdi_da.sel(parties_complete=party, hdi=lower_coord) * 100, # Scale 0-1 data by 100
            latent_hdi_da.sel(parties_complete=party, hdi=upper_coord) * 100, # Scale 0-1 data by 100
            alpha=0.3, label="94% HDI"
        )

        # Plot Daily Average Noisy Popularity (if available) (Scale 0-1 data by 100)
        if noisy_pop_daily_mean is not None and noisy_pop_daily_dates is not None:
             party_idx_noisy = parties.index(party)
             ax.plot(noisy_pop_daily_dates, noisy_pop_daily_mean[:, party_idx_noisy] * 100, # Scale 0-1 data by 100
                     label="Noisy Pop (Daily Avg)", color='green', linestyle='--', alpha=0.8)


        # Plot Observed Polls (Calculate percentage from counts)
        party_polls = polls_melt[polls_melt['party'] == party]
        # Calculate percentage: (count / sample_size) * 100
        # Handle potential division by zero if sample_size is 0 or NaN
        observed_percentage = np.where(
            party_polls['sample_size'] > 0,
            (party_polls['vote_count'] / party_polls['sample_size']) * 100,
            np.nan # Assign NaN if sample_size is not positive
        )
        ax.scatter(party_polls['date'], observed_percentage, alpha=0.6, s=10, label="Observed Polls", color='orange')

        ax.set_title(party)
        ax.set_ylabel("Vote Share")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter()) # Use default formatter for 0-100 data
        ax.legend()
        ax.grid(True, linestyle=':')
        # Add reasonable y-limits for percentage data
        # ax.set_ylim(-5, 105) # Set y-axis limits from -5% to 105% <-- Commented out

        # Add vertical line for target election date
        if include_target_date and dataset.election_date:
            try:
                election_dt = pd.to_datetime(dataset.election_date)
                ax.axvline(election_dt, color='red', linestyle='--', alpha=0.7, label=f"Target ({dataset.election_date})")
            except Exception as e:
                 print(f"Warning: Could not plot vertical line for election date {dataset.election_date}: {e}")


    # Add vertical lines for historical election dates
    historical_dates_dt = pd.to_datetime(dataset.historical_election_dates)
    for ax in axes[:n_parties]: # Only plot on axes with data
        for election_date in historical_dates_dt:
            ax.axvline(election_date, color='grey', linestyle=':', alpha=0.5)


    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Latent Popularity Trajectory vs Observed Polls", fontsize=16, y=1.02)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap

    # Save plot
    plot_path = os.path.join(output_dir, "latent_popularity_vs_polls.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Latent popularity plot saved to {plot_path}")

def plot_latent_component_contributions(elections_model, output_dir):
    """Plots the contribution of different latent components to the final popularity."""
    try:
        # --- Check Model Type ---
        if isinstance(elections_model.model_instance, DynamicGPElectionModel):
            print("Skipping latent component plot: Not applicable for DynamicGPElectionModel.")
            return
        # --- End Check ---

        if elections_model.trace is None or elections_model.trace.posterior is None:
            print("Error: Posterior trace not found. Cannot plot component contributions.")
            return

        posterior = elections_model.trace.posterior

        # Check required variables for static model
        required_vars = ["party_baseline", "election_party_baseline", "party_time_effect"]
        missing_vars = [var for var in required_vars if var not in posterior]
        if missing_vars:
            print(f"Error: Missing required variables for component plot: {missing_vars}. Available: {list(posterior.data_vars)}")
            return

        party_baseline = posterior["party_baseline"]
        election_baseline = posterior["election_party_baseline"]
        time_effect = posterior["party_time_effect"]

        political_families = elections_model.dataset.political_families
        num_parties = len(political_families)

        # Use the target election index for plotting current cycle components
        # Assuming target election is the last one in the coordinates
        election_idx = -1 # Index for the target election

        num_cols = 3
        num_rows = (num_parties + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharex=True)
        axes = axes.flatten()

        for i, party in enumerate(political_families):
            ax = axes[i]
            try:
                 party_idx = list(posterior['parties_complete'].values).index(party) # Get index robustly
            except ValueError:
                 print(f"Warning: Party '{party}' not found in posterior coordinates. Skipping component plot for this party.")
                 continue

            # Calculate means of components for the target election
            mean_party_base = party_baseline.isel(parties_complete=party_idx).mean().values
            mean_election_base = election_baseline.isel(elections=election_idx, parties_complete=party_idx).mean().values
            # Time effect mean over countdown for the target election
            mean_time_effect = time_effect.isel(elections=election_idx, parties_complete=party_idx).mean(dim=["chain", "draw"]).values

            countdown_coords = time_effect["countdown"].values

            # Plot components relative to zero
            ax.axhline(0, color='gray', linestyle='--', lw=0.5)
            ax.bar("Overall Baseline", mean_party_base, label=f'Party Baseline ({mean_party_base:.2f})')
            ax.bar("Election Baseline", mean_election_base, label=f'Election Baseline ({mean_election_base:.2f})')

            # Find min/max of time effect for context
            min_time_eff = mean_time_effect.min()
            max_time_eff = mean_time_effect.max()
            ax.bar("Time Effect Range", max_time_eff - min_time_eff, bottom=min_time_eff, alpha=0.6, label=f'Time Effect (Range: {min_time_eff:.2f} to {max_time_eff:.2f})')

            # Calculate total latent mean (excluding non-competing mask) at countdown=0
            # Ensure countdown=0 exists, find its index
            try:
                 t0_index = list(countdown_coords).index(0)
                 total_latent_mean_t0 = mean_party_base + mean_election_base + mean_time_effect[t0_index]
                 ax.bar("Total Latent Mean (t=0)", total_latent_mean_t0, color='purple', alpha=0.7, label=f'Total Latent (t=0) ({total_latent_mean_t0:.2f})')
            except ValueError:
                 print("Warning: Countdown=0 not found, cannot plot Total Latent Mean (t=0).")


            ax.set_title(f"{party}")
            ax.set_ylabel("Latent Scale Contribution")
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Mean Latent Component Contributions (Target Election)", fontsize=16, y=1.05)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "latent_component_contributions.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Latent component contribution plot saved to {plot_path}")
        plt.close(fig)

    except KeyError as e:
        print(f"Error plotting components: Missing key {e}. Ensure static model trace is loaded.")
        import traceback
        traceback.print_exc() # Print traceback for key errors too
    except Exception as e:
        print(f"An error occurred during component contribution plotting: {e}")
        import traceback
        traceback.print_exc()

def plot_recent_polls(elections_model, output_dir, days_limit=90):
    """
    Generates and saves a plot showing raw poll data collected since the 
    most recent historical election.

    Parameters:
    ----------
    elections_model : ElectionsFacade
        The facade object containing the dataset.
    output_dir : str
        The directory where the plots will be saved.
    days_limit : int, optional
        The number of days from the most recent election to include in the plot.
    """
    print(f"Generating recent polls plot in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    dataset = elections_model.dataset
    all_polls = dataset.polls # Use the original polls dataframe with raw percentages
    political_families = dataset.political_families
    
    # Find the date of the most recent historical election
    if not dataset.historical_election_dates:
        print("Error: No historical election dates found in dataset.")
        return
    last_election_date_str = max(dataset.historical_election_dates) # Find the actual latest date
    last_election_date_dt = pd.to_datetime(last_election_date_str)
    print(f"Plotting polls since the last historical election: {last_election_date_str}")

    # Filter polls to include only those after the last election
    recent_polls = all_polls[all_polls['date'] > last_election_date_dt].copy()

    if recent_polls.empty:
        print(f"No polls found after {last_election_date_str}. Skipping recent polls plot.")
        return

    # Check if poll percentages need conversion (e.g., from 0-100 to 0-1)
    if recent_polls[political_families].max().max() > 1.1: 
        print(f"    Converting recent poll percentages from 0-100 to 0-1.")
        for party in political_families:
            recent_polls[party] = recent_polls[party] / 100.0

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(political_families)))

    # Melt the recent polls data for plotting
    recent_polls_melt = recent_polls.melt(
        id_vars=['date', 'pollster'], 
        value_vars=political_families, 
        var_name='party', 
        value_name='percentage'
    )

    # Plot raw poll results as scatter points
    sns.scatterplot(data=recent_polls_melt, x='date', y='percentage', hue='party', style='pollster', 
                    alpha=0.7, s=50, ax=ax, palette=colors[:len(political_families)]) # Use seaborn for consistency

    # Add vertical line for the last election date
    ax.axvline(last_election_date_dt, color='black', linestyle='--', linewidth=1.5, 
               label=f'Last Election ({last_election_date_str})')

    ax.set_title(f'Recent Polls Since Last Election ({last_election_date_str})')
    ax.set_xlabel('Poll Date')
    ax.set_ylabel('Raw Vote Percentage')
    ax.legend(title='Party / Pollster', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, None) # Start y-axis at 0
    
    # Set sensible x-axis limits starting from the last election
    plot_start_date = last_election_date_dt
    # Extend slightly beyond the last poll date, or use today if no polls
    plot_end_date = recent_polls['date'].max() if not recent_polls.empty else pd.Timestamp.now()
    ax.set_xlim(left=plot_start_date - pd.Timedelta(days=5), right=plot_end_date + pd.Timedelta(days=10)) 
    
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the figure
    plot_filename = os.path.join(output_dir, f"recent_polls_since_{last_election_date_str}.png")
    plt.savefig(plot_filename, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_filename}")

    print("Finished generating recent polls plot.")

def plot_house_effects_heatmap(elections_model, output_dir):
    """Plots a heatmap of the mean house effects."""
    try:
        # --- Check Model Type ---
        if isinstance(elections_model.model_instance, DynamicGPElectionModel):
            # Check if house effects actually exist in this specific dynamic model trace
            if elections_model.trace is None or "house_effects" not in elections_model.trace.posterior:
                 print("Skipping house effects heatmap: 'house_effects' not found in trace for this DynamicGPElectionModel run.")
                 return
            else:
                 print("Info: Generating house effects heatmap for DynamicGPElectionModel (requires house effects to be included in sampling).")
        # --- End Check ---


        if elections_model.trace is None or elections_model.trace.posterior is None:
            print("Error: Posterior trace not found. Cannot plot house effects.")
            return

        posterior = elections_model.trace.posterior

        if "house_effects" not in posterior:
             print("Error: 'house_effects' not found in posterior trace.")
             return

        house_effects = posterior["house_effects"]
        mean_house_effects = house_effects.mean(dim=["chain", "draw"])

        # Convert to DataFrame for easier plotting with seaborn
        # Ensure coords are strings if they are not already
        pollsters = [str(p) for p in mean_house_effects["pollsters"].values]
        parties = [str(p) for p in mean_house_effects["parties_complete"].values]

        df_house_effects = pd.DataFrame(mean_house_effects.values, index=pollsters, columns=parties)

        # Check for convergence issues (optional but recommended)
        try:
            ess_bulk = az.ess(house_effects).to_dataframe()['ess_bulk'].unstack()
            low_ess_mask = ess_bulk < 400 # Threshold can be adjusted
        except Exception as ess_err:
             print(f"Warning: Could not calculate ESS for house effects: {ess_err}. Skipping low ESS annotation.")
             low_ess_mask = pd.DataFrame(False, index=df_house_effects.index, columns=df_house_effects.columns) # Assume all False


        plt.figure(figsize=(max(10, len(parties) * 0.8), max(8, len(pollsters) * 0.4)))

        # Determine center for diverging palette, typically 0
        center_val = 0

        # Choose colormap
        cmap = "RdBu" # Red-Blue diverging palette is good for positive/negative effects

        ax = sns.heatmap(
            df_house_effects,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=center_val,
            linewidths=.5,
            cbar_kws={'label': 'Mean House Effect (Latent Scale)'}
        )

        # Add markers for low ESS values
        if low_ess_mask.any().any():
             print("Info: Adding yellow borders to heatmap cells with Bulk ESS < 400.")
             for y, pollster in enumerate(df_house_effects.index):
                 for x, party in enumerate(df_house_effects.columns):
                     # Check if pollster/party exists in low_ess_mask before accessing
                     if pollster in low_ess_mask.index and party in low_ess_mask.columns:
                          if low_ess_mask.loc[pollster, party]:
                              ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='yellow', lw=2, clip_on=False))
                     else:
                          print(f"Warning: Could not find {pollster} or {party} in low_ess_mask.")

        ax.set_title("Mean House Effects per Pollster and Party", fontsize=16)
        ax.set_xlabel("Political Family / Party")
        ax.set_ylabel("Pollster")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Add a note if low ESS was detected
        if low_ess_mask.any().any():
             plt.text(0.5, -0.15, '* Yellow border indicates Bulk ESS < 400', size=10, ha="center", transform=ax.transAxes)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout

        plot_path = os.path.join(output_dir, "house_effects_heatmap.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"House effects heatmap saved to {plot_path}")
        plt.close()

    except KeyError as e:
        print(f"Error plotting house effects: Missing key {e}.")
        import traceback
        traceback.print_exc() # Print traceback for key errors too
    except Exception as e:
        print(f"An error occurred during house effects heatmap plotting: {e}")
        import traceback
        traceback.print_exc()

def plot_forecasted_election_distribution(elections_model: 'ElectionsFacade', output_dir: str):
    """
    Generates and saves a plot visualizing the posterior distribution
    of the inferred latent popularity on election day using boxenplots.

    Parameters:
    ----------
    elections_model : ElectionsFacade
        The facade object containing the dataset and inference trace.
    output_dir : str
        The directory where the plot will be saved.
    """
    print(f"Generating election day latent popularity distribution plot in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Get posterior samples of latent popularity at election day
    latent_pop_samples = elections_model.get_election_day_latent_popularity()

    if latent_pop_samples is None:
        print("Error: Failed to get latent popularity samples for election day. Skipping plot.")
        return

    # 2. Convert to DataFrame for easier plotting with Seaborn
    latent_pop_df = latent_pop_samples.to_dataframe().reset_index() # Simpler conversion
    
    # Rename the value column for clarity
    value_col_name = latent_pop_samples.name if latent_pop_samples.name else 'latent_popularity' # Use variable name or default
    if value_col_name in latent_pop_df.columns:
        latent_pop_df = latent_pop_df.rename(columns={value_col_name: 'latent_proportion'})
    else:
        # Handle cases where the name might be different or missing
        potential_value_cols = [col for col in latent_pop_df.columns if col not in ['sample', 'parties_complete']]
        if len(potential_value_cols) == 1:
             latent_pop_df = latent_pop_df.rename(columns={potential_value_cols[0]: 'latent_proportion'})
             print(f"Warning: Renamed value column from '{potential_value_cols[0]}' to 'latent_proportion'.")
        else:
             print(f"Error: Could not identify the value column (expected '{value_col_name}' or similar). Columns found: {latent_pop_df.columns}")
             return
             
    # Check for NaNs (unlikely for latent pop, but good practice)
    nan_count = latent_pop_df['latent_proportion'].isna().sum()
    if nan_count > 0:
        print(f"Warning: Plotting data contains {nan_count} NaN values. These will be excluded.")
        latent_pop_df = latent_pop_df.dropna(subset=['latent_proportion'])
        if latent_pop_df.empty:
             print("Error: All latent popularity samples were NaN. Cannot generate plot.")
             return

    # 3. Create the plot using Seaborn boxenplot
    plt.figure(figsize=(12, 7))
    
    # Order parties by median latent proportion (descending)
    party_order = latent_pop_df.groupby('parties_complete')['latent_proportion'].median().sort_values(ascending=False).index
    
    num_parties = len(party_order)
    palette = sns.color_palette("tab10", num_parties)
    party_color_map = {party: palette[i] for i, party in enumerate(party_order)}

    sns.boxenplot(
        data=latent_pop_df,
        x='parties_complete',
        y='latent_proportion',
        order=party_order,
        palette=party_color_map,
    )

    plt.title(f'Distribution of Inferred Latent Popularity on Election Day ({elections_model.election_date})')
    plt.xlabel('Political Party / Alliance')
    plt.ylabel('Inferred Latent Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, None) 
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # 4. Save the plot
    # Keep filename similar for consistency, or rename if desired
    plot_filename = os.path.join(output_dir, f"election_day_latent_distribution_{elections_model.election_date}.png")
    try:
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved plot: {plot_filename}")
    except Exception as e:
        print(f"Error saving latent distribution plot: {e}")
    finally:
        plt.close()

    print("Finished generating election day latent distribution plot.") 