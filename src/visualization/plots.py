import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data.dataset import ElectionDataset
import os
import numpy as np
import xarray as xr
import arviz as az
from typing import TYPE_CHECKING, List, Dict
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
if TYPE_CHECKING:
    from src.models.elections_facade import ElectionsFacade
from src.models.dynamic_gp_election_model import DynamicGPElectionModel # Import the dynamic model class

# --- Define Standard Party Colors ---
PARTY_COLORS = {
    'PS': '#FFC0CB',       # Pink
    'AD': '#FF8C00',       # Orange
    'CH': '#00008B',       # Dark Blue
    'IL': '#ADD8E6',       # Light Blue
    'BE': '#8B0000',       # Dark Red / Maroon
    'CDU': '#DC143C',      # Crimson
    'PAN': '#008000',      # Green
    'L': '#808080',        # Grey
    'other': '#D3D3D3'     # Light Grey (for completeness)
}

def _get_party_color(party_name):
    """Helper function to get party color, defaulting to grey."""
    return PARTY_COLORS.get(party_name, '#808080') # Default to Grey

def plot_election_data(dataset: ElectionDataset, output_dir: str):
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
    fig, ax = plt.subplots(figsize=(18, 10))

    # Use the defined party color map
    palette = {party: _get_party_color(party) for party in political_families}

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

    # Plot historical polls with specific colors
    sns.scatterplot(data=historical_polls_melt, x='date', y='percentage', hue='party',
                    alpha=0.3, marker='o', s=15, legend='auto', palette=palette, ax=ax)

    # Plot future polls (if any) with specific colors
    if not future_polls_melt.empty:
        print(f"Plotting {len(future_polls_melt['date'].unique())} future poll dates (after {last_historical_election_date.date()})")
        sns.scatterplot(data=future_polls_melt, x='date', y='percentage', hue='party',
                        alpha=0.6, marker='s', s=25, legend=False, palette=palette, ax=ax)

    # Plot election results with specific colors
    sns.scatterplot(data=results_melt, x='date', y='percentage', hue='party',
                    s=250, marker='X', edgecolor='black', zorder=5, legend=False, palette=palette, ax=ax)

    # Manually create legend handles for clarity
    handles, labels = ax.get_legend_handles_labels()
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

    ax.set_title('All Polls and Election Results Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Vote Percentage')
    ax.legend(handles=all_handles, labels=all_labels, title='Party / Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    # Save the plot
    plot_filename = os.path.join(output_dir, "all_polls_results_over_time.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close(fig)

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

        fig_cycle, ax_cycle = plt.subplots(figsize=(15, 8))

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

        # Plot polls for the cycle using defined colors
        sns.lineplot(data=cycle_polls_melt, x='date', y='percentage', hue='party',
                     style='pollster', alpha=0.6, marker='o', linestyle='--',
                     markersize=5, palette=palette, ax=ax_cycle)
        
        # Plot the final election result using defined colors
        if not cycle_result_melt.empty:
             sns.scatterplot(data=cycle_result_melt, x='date', y='percentage', hue='party',
                             s=250, marker='X', edgecolor='black', zorder=5, legend=False,
                             palette=palette, ax=ax_cycle)

        ax_cycle.set_title(f'Polls Leading up to {election_date_dt.date()} Election')
        ax_cycle.set_xlabel('Poll Date')
        ax_cycle.set_ylabel('Vote Percentage')
        ax_cycle.axvline(election_date_dt, color='r', linestyle=':', linewidth=2, label=f'Election Day ({election_date_dt.date()})')
        ax_cycle.legend(title='Party/Pollster', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_cycle.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_cycle.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.tight_layout()
        # Save the plot
        cycle_filename = os.path.join(output_dir, f"cycle_polls_{election_date_dt.strftime('%Y%m%d')}.png")
        plt.savefig(cycle_filename)
        print(f"Cycle plot saved to {cycle_filename}")
        plt.close(fig_cycle)

def plot_latent_popularity_vs_polls(elections_model, output_dir, include_target_date=True):
    """
    Plots the latent popularity mean and HDI against observed polls over time.
    Handles both StaticBaselineElectionModel and DynamicGPElectionModel.

    Args:
        elections_model: The fitted ElectionsFacade instance.
        output_dir: Directory to save the plot.
        include_target_date: Whether to include the target election date as a vertical line.
    """
    if elections_model.trace is None or not hasattr(elections_model.trace, 'posterior'):
        print("Posterior trace not found. Skipping latent popularity plot.")
        return

    trace = elections_model.trace
    dataset = elections_model.dataset
    model_instance = elections_model.model_instance # Get the specific model instance
    polls_train = dataset.polls_train
    political_families = dataset.political_families
    is_dynamic_model = isinstance(model_instance, DynamicGPElectionModel)
    print(f"DEBUG plot_latent_pop: is_dynamic_model = {is_dynamic_model}")

    latent_variable_da = None
    time_values = None

    # Determine which latent variable and time coordinate to use
    # We now directly use the deterministic variable if available
    deterministic_pop_var = "latent_popularity_national"
    if deterministic_pop_var in trace.posterior:
        print(f"Using deterministic national popularity variable: {deterministic_pop_var}")
        latent_variable_da = trace.posterior[deterministic_pop_var].copy()
        time_coord_name = "calendar_time"
        if time_coord_name in latent_variable_da.coords:
            time_values = pd.to_datetime(latent_variable_da[time_coord_name].values)
        else:
            print(f"Warning: Time coordinate '{time_coord_name}' not found in {deterministic_pop_var}.")
    # Fallback to old logic ONLY if the new variable isn't found (for compatibility?)
    elif isinstance(model_instance, DynamicGPElectionModel):
        potential_var_name = "national_trend_pt" # Pre-softmax
        time_coord_name = "calendar_time"
        if potential_var_name in trace.posterior:
            print(f"Warning: {deterministic_pop_var} not found. Falling back to pre-softmax {potential_var_name} and applying manual softmax.")
            latent_variable_da_raw = trace.posterior[potential_var_name].copy()
            
            # <<< Keep the manual numpy softmax ONLY as a fallback >>>
            try:
                values = latent_variable_da_raw.values
                party_axis = -1
                max_val = np.max(values, axis=party_axis, keepdims=True)
                stable_values = values - max_val
                exp_values = np.exp(stable_values)
                sum_exp_values = np.sum(exp_values, axis=party_axis, keepdims=True)
                epsilon = 1e-9
                softmax_values = exp_values / (sum_exp_values + epsilon)
                latent_variable_da = xr.DataArray(
                    softmax_values,
                    coords=latent_variable_da_raw.coords,
                    dims=latent_variable_da_raw.dims,
                    name=(latent_variable_da_raw.name + "_manual_softmax") if latent_variable_da_raw.name else "latent_manual_softmax"
                )
                print("Applied MANUAL numpy softmax (fallback).")
                # Ensure coords are restored (might be redundant)
                if 'parties_complete' in latent_variable_da_raw.coords and 'parties_complete' not in latent_variable_da.coords:
                    latent_variable_da = latent_variable_da.assign_coords(parties_complete=latent_variable_da_raw['parties_complete'])
            except Exception as e:
                print(f"Error applying fallback manual numpy softmax: {e}. Proceeding with pre-softmax values.")
                latent_variable_da = latent_variable_da_raw # Revert if fallback softmax fails
            # <<< End fallback softmax >>>

            if time_coord_name in latent_variable_da.coords:
                time_values = pd.to_datetime(latent_variable_da[time_coord_name].values)
            else:
                 print(f"Warning: Time coordinate '{time_coord_name}' not found in {potential_var_name}.")
        else:
            print(f"Dynamic model variable '{potential_var_name}' not found.")
    # --- Keep Static Model Logic --- #
    elif isinstance(model_instance, StaticBaselineElectionModel): # Static model
        potential_var_name = "latent_popularity_trajectory"
        time_coord_name = "countdown"
        if potential_var_name in trace.posterior:
            print(f"Found static model variable: {potential_var_name}")
            latent_variable_da = trace.posterior[potential_var_name].copy()
            # Select last election cycle and map countdown to dates
            if 'elections' in latent_variable_da.coords and 'countdown' in latent_variable_da.coords:
                try:
                    last_election_date_str = latent_variable_da['elections'].values[-1]
                    last_election_date = pd.to_datetime(last_election_date_str)
                    latent_variable_da = latent_variable_da.sel(elections=last_election_date_str)
                    countdown_values = latent_variable_da[time_coord_name].values
                    time_values = last_election_date - pd.to_timedelta(countdown_values, unit='D')
                    print(f"Mapped static model countdown to dates using last election: {last_election_date.date()}")
                except Exception as e:
                     print(f"Error processing static model trajectory: {e}")
            else:
                print("Warning: Static model trajectory missing 'elections' or 'countdown' coords.")
        else:
            print(f"Static model variable '{potential_var_name}' not found.")
    # --- End Static Model Logic --- #
    else:
        print(f"Error: Could not determine appropriate latent variable. Model type: {type(model_instance)}")
        return

    # --- Check if successful --- 
    if latent_variable_da is None:
        print(f"Error: Could not find or process a suitable latent trajectory variable. Skipping plot.")
        return
    if time_values is None:
        print("Error: Time values could not be determined. Skipping plot.")
        return

    # --- Calculate mean and HDI --- 
    print("Calculating mean and HDI...")
    mean_latent_popularity = latent_variable_da.mean(dim=["chain", "draw"])
    hdi_latent_popularity = az.hdi(latent_variable_da, hdi_prob=0.94)
    print("Mean and HDI calculated.")

    # --- Remove Noisy Popularity Section (for clarity) ---
    noisy_pop_daily_mean = None
    noisy_pop_daily_dates = None
    print("Skipping noisy popularity calculation/plotting for clarity.")

    # +++ Add Debugging +++
    print("\nDEBUG: Inspecting mean_latent_popularity before plotting loop:")
    print(f"  Type: {type(mean_latent_popularity)}")
    if hasattr(mean_latent_popularity, 'dims'):
        print(f"  Dims: {mean_latent_popularity.dims}")
    if hasattr(mean_latent_popularity, 'coords'):
        print(f"  Coords: {list(mean_latent_popularity.coords.keys())}")
    # print(mean_latent_popularity) # Optional: Print the whole object if small enough
    print("DEBUG: End inspection ---\n")
    # +++ End Debugging +++

    # --- Plotting ---
    parties = dataset.political_families
    n_parties = len(parties)
    n_cols = 3
    n_rows = (n_parties + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    # Prepare melted poll data (outside loop)
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
        # Select party from the already calculated mean DataArray
        latent_mean_party = mean_latent_popularity.sel(parties_complete=party)
        ax.plot(time_values, latent_mean_party * 100, label="Latent Popularity (Mean)")

        # Plot HDI using the calculated HDI dataset (which is an xarray Dataset)
        # Select the specific party from the HDI Dataset
        # The variable within the HDI dataset has the name of the original variable
        hdi_data_array_party = hdi_latent_popularity.sel(parties_complete=party)
        # Check if the latent_var_name exists as a variable in the hdi dataset
        if latent_variable_da.name in hdi_data_array_party:
             hdi_plot_data = hdi_data_array_party[latent_variable_da.name]
             ax.fill_between(
                 time_values,
                 hdi_plot_data.isel(hdi=0) * 100, # Use index 0 for lower bound
                 hdi_plot_data.isel(hdi=1) * 100, # Use index 1 for upper bound
                 alpha=0.3, label="94% HDI"
             )
        else:
             print(f"Warning: Could not find HDI variable '{latent_variable_da.name}' for party {party}. Skipping HDI plot.")

        # Removed noisy popularity plot line

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
        # Ensure y-limits accommodate the data
        ax.autoscale(enable=True, axis='y') 
        # ax.set_ylim(bottom=max(-5, ax.get_ylim()[0]), top=min(105, ax.get_ylim()[1])) # Optional adjusted limits

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

def plot_forecasted_election_distribution(
    elections_model: 'ElectionsFacade',
    output_dir: str,
    date_mode: str = "election_day", # Added date_mode
    filename: str = "forecast_distribution_forestplot.png" # Changed default filename
):
    """
    Generates a forest plot showing the posterior distribution of predicted vote shares.
    Uses the date corresponding to the specified date_mode.

    Args:
        elections_model (ElectionsFacade): The facade instance.
        output_dir (str): Directory to save the plot.
        date_mode (str): Mode used for popularity extraction ('election_day', 'last_poll', 'today').
        filename (str): Filename for saving the plot.
    """
    if elections_model is None or elections_model.trace is None:
        print("Model or trace not available. Cannot generate forecast distribution plot.")
        return

    fig_obj = None # Initialize fig_obj
    try:
        print(f"Generating latent popularity distribution plot ({date_mode}) in: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Use the NEW method, specifying the desired date_mode
        target_pop_dist = elections_model.get_latent_popularity(date_mode=date_mode)

        # Determine the actual date used for the title
        plot_date = None
        if target_pop_dist is not None and 'calendar_time' in target_pop_dist.coords:
             try:
                 # Extract the actual date selected by get_latent_popularity
                 plot_date = pd.Timestamp(target_pop_dist['calendar_time'].item()).normalize()
             except Exception as e:
                 print(f"Warning: Could not extract exact date from popularity data, using mode logic. Error: {e}")

        # Fallback or explicit date determination based on mode if extraction failed
        if plot_date is None:
            if date_mode == 'election_day':
                plot_date = pd.Timestamp(elections_model.election_date).normalize()
            elif date_mode == 'last_poll' and elections_model.dataset is not None:
                try:
                    plot_date = elections_model.dataset.polls_train['date'].max().normalize()
                except Exception: # Handle case where polls_train might be empty or date column missing
                    print("Warning: Could not determine last poll date from dataset.")
                    plot_date = pd.Timestamp("today").normalize() # Fallback
            else: # Default to today if mode is 'today' or last_poll failed
                plot_date = pd.Timestamp("today").normalize()

        if target_pop_dist is None:
            print(f"Error: Could not retrieve latent popularity from the model for mode '{date_mode}'.")
            # Create an empty placeholder plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Could not retrieve latent popularity data for '{date_mode}'.",
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            # Use determined plot_date in title if available
            title_date_str = plot_date.strftime('%Y-%m-%d') if plot_date else date_mode
            ax.set_title(f"Forecast Distribution Unavailable ({title_date_str})")
            plot_filename = os.path.join(output_dir, filename.replace(".png", "_unavailable.png"))
            plt.savefig(plot_filename)
            plt.close(fig)
            return

        # Determine party colors
        # Use coordinate directly, assuming it's clean now. If not, needs preprocessing.
        party_coord_name = next((name for name in ['parties_complete', 'party', 'parties'] if name in target_pop_dist.coords), None)
        if party_coord_name is None:
             print("Error: Could not find party coordinate in popularity data.")
             # Handle error appropriately, maybe create error plot as below
             return
             
        parties = target_pop_dist[party_coord_name].values
        colors = [_get_party_color(p) for p in parties] # Assuming _get_party_color exists

        # --- Prepare data dictionary for plotting --- 
        # Re-implementing dictionary creation
        plot_data_dict = {}
        try:
            # Reshape data to have samples dimension
            reshaped_data = target_pop_dist.stack(sample=("chain", "draw"))
            for party in parties:
                 # Extract samples as numpy array
                 party_data = reshaped_data.sel({party_coord_name: party}).values
                 plot_data_dict[str(party)] = party_data # Key = party name, Value = samples array
            
            # Removed the conversion to Dataset step
            # plot_dataset = az.convert_to_dataset(plot_data_dict)

        except Exception as data_prep_err:
             print(f"Error preparing data dictionary for plot_forest: {data_prep_err}")
             return # Exit if data prep fails
             
        # --- Plotting ---
        # Pass the dictionary and explicitly specify var_names using dict keys
        fig = az.plot_forest(
            plot_data_dict, # Pass the dictionary
            var_names=list(plot_data_dict.keys()), # Explicitly use dict keys as var_names
            hdi_prob=0.94,
            figsize=(10, max(6, len(parties)*0.6)), # Adjusted height based on num parties
            model_names=[""], # Prevent default model name prefix
            combined=True # Combine chains for cleaner plot
        )

        # Enhance the plot
        if isinstance(fig, np.ndarray):
            main_ax = fig.ravel()[0]
            fig_obj = main_ax.figure
        else:
            fig_obj = fig # az.plot_forest returns a Figure object directly now
            main_ax = fig_obj.axes[0]

        # Use determined plot_date in the title
        title_date_str = plot_date.strftime('%Y-%m-%d')
        mode_str = date_mode.replace("_", " ").title()
        fig_obj.suptitle(f'Posterior Vote Shares ({mode_str}: {title_date_str})', fontsize=16)
        main_ax.set_xlabel("Predicted Vote Share", fontsize=12)
        main_ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        # Remove manual y-axis label setting - let plot_forest handle it
        # try:
        #      main_ax.set_yticks(np.arange(len(parties)))
        #      main_ax.set_yticklabels(parties)
        # except Exception as e:
        #      print(f"Warning: Could not manually set y-tick labels: {e}")
             
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the plot using the provided filename
        plot_filename = os.path.join(output_dir, filename)
        plt.savefig(plot_filename)
        print(f"Forecast distribution plot saved to {plot_filename}")
        plt.close(fig_obj)

    except Exception as e:
        print(f"Error generating forecast distribution plot: {e}")
        import traceback
        traceback.print_exc()
        # Optionally create an error plot
        if fig_obj is not None: plt.close(fig_obj) # Close if figure exists
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating plot:\n{e}", 
                    ha='center', va='center', fontsize=10, wrap=True)
            ax.set_title(f"Forecast Distribution Error ({date_mode.replace('_',' ').title()})") # Use date_mode in error title
            plot_filename = os.path.join(output_dir, filename.replace(".png", "_error.png")) # Use provided filename base
            plt.savefig(plot_filename)
            plt.close(fig)
        except Exception as inner_e:
             print(f"Error creating error placeholder plot: {inner_e}")

    finally:
        if fig_obj is not None: plt.close(fig_obj)

def plot_reliability_diagram(calibration_data: Dict[str, np.ndarray], title: str, filename: str):
    """
    Plots a reliability diagram based on calibration data.

    Args:
        calibration_data: Dictionary containing 'mean_predicted_prob', 
                          'mean_observed_prob', and 'bin_counts'.
        title: Title for the plot.
        filename: Full path to save the plot image.
    """
    mean_predicted = calibration_data['mean_predicted_prob']
    mean_observed = calibration_data['mean_observed_prob']
    counts = calibration_data['bin_counts']

    # Filter out bins with zero counts (where probs are NaN)
    valid_bins = ~np.isnan(mean_predicted)
    mean_predicted = mean_predicted[valid_bins]
    mean_observed = mean_observed[valid_bins]
    counts = counts[valid_bins]

    if len(mean_predicted) == 0:
        print(f"Skipping reliability diagram '{title}': No valid bins with data.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot the calibration curve
    ax.plot(mean_predicted, mean_observed, 'o-', label='Model Calibration', markersize=8, color='blue')

    # Plot the perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    # Add counts as text annotations (optional, can get crowded)
    # for i, count in enumerate(counts):
    #     ax.text(mean_predicted[i], mean_observed[i] + 0.02, f'{count}', 
    #             ha='center', va='bottom', fontsize=8)

    # Create a simple histogram for the distribution of predictions
    ax_hist = ax.twinx() # Share the x-axis
    bin_edges_plot = calibration_data['bin_edges'][valid_bins] # Get edges for valid bins
    counts_plot = counts # Counts corresponding to valid bins

    # Calculate widths based on the *valid* bin edges for plotting
    if len(bin_edges_plot) > 1:
        widths = np.diff(bin_edges_plot)
        # Add width for the last bin (use width of second-to-last if available)
        if len(widths) > 0:
             widths = np.append(widths, widths[-1])
        else: # Only two edges, one bin
             widths = [bin_edges_plot[1] - bin_edges_plot[0]]
    elif len(bin_edges_plot) == 1:
        # Only one valid bin edge means one bar. Estimate width.
        # Use 1/n_bins as approximate width
        n_bins_total = len(calibration_data['mean_predicted_prob']) # Original number of bins
        widths = [1.0 / n_bins_total if n_bins_total > 0 else 0.1] 
    else: # No valid bins
        widths = []

    # Ensure widths array matches the number of counts/bars
    if len(widths) != len(counts_plot):
         print(f"Warning: Mismatch in counts ({len(counts_plot)}) and widths ({len(widths)}). Skipping histogram.")
         can_plot_hist = False
    elif len(bin_edges_plot) == 0 or len(counts_plot) == 0:
         can_plot_hist = False
    else:
         can_plot_hist = True

    if can_plot_hist:
         # Use bin_edges_plot as the left edge for bars
         ax_hist.bar(bin_edges_plot, counts_plot, width=widths, alpha=0.2, align='edge', color='gray', label='Prediction Count')
         ax_hist.set_ylabel('Count', color='gray')
         ax_hist.tick_params(axis='y', labelcolor='gray')
         ax_hist.set_yscale('log') # Use log scale for counts often
    else:
         print("Could not plot histogram due to bin/width mismatch or no data.")

    ax.set_xlabel('Mean Predicted Probability (in bin)')
    ax.set_ylabel('Mean Observed Proportion (in bin)')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper left')
    
    # Set limits and aspect ratio
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal', adjustable='datalim') # Changed 'box' to 'datalim'

    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Reliability diagram saved to {filename}")
    except Exception as e:
        print(f"Error saving reliability diagram '{filename}': {e}")
    plt.close(fig) 

def plot_latent_trend_since_last_election(elections_model: 'ElectionsFacade', output_dir: str):
    """
    Plots the latent popularity mean and HDI for all parties from the last
    historical election up to the target election date, overlaying observed polls.

    Args:
        elections_model: The fitted ElectionsFacade instance.
        output_dir: Directory to save the plot.
    """
    if elections_model.trace is None:
        print("Trace not found. Skipping latent trend plot since last election.")
        return

    trace = elections_model.trace
    dataset = elections_model.dataset
    model_instance = elections_model.model_instance
    polls_train = dataset.polls_train.copy()
    political_families = dataset.political_families
    is_dynamic_model = isinstance(model_instance, DynamicGPElectionModel)

    # Determine start and end dates
    historical_dates_dt = pd.to_datetime(dataset.historical_election_dates)
    if historical_dates_dt.empty:
        print("Warning: No historical election dates found. Cannot determine start date.")
        # Optionally default to start of data or some fixed period
        start_date = polls_train['date'].min()
    else:
        start_date = historical_dates_dt.max()
    
    end_date = pd.to_datetime(dataset.election_date)
    
    print(f"Plotting latent trend from {start_date.date()} to {end_date.date()}")

    # Determine latent variable and time coordinate (assuming dynamic model primarily)
    # TODO: Add handling for static model if needed
    if not is_dynamic_model:
        print("Warning: Plotting latent trend since last election currently optimized for dynamic_gp model.")
        # Fallback or specific implementation for static model needed here
        latent_var_name = "latent_popularity_t" # Adjust if static model uses different var name
        time_coord_name = "date"
    else:
        # Use the new deterministic variable directly
        latent_var_name = "latent_popularity_national" 
        time_coord_name = "calendar_time"

    if latent_var_name not in trace.posterior:
        # Add a fallback check for older variable name for compatibility?
        fallback_var_name = "latent_popularity_calendar_trajectory"
        if fallback_var_name in trace.posterior:
             print(f"Warning: Variable '{latent_var_name}' not found. Using fallback '{fallback_var_name}'.")
             latent_var_name = fallback_var_name
        else:
             print(f"Latent variable '{latent_var_name}' (or fallback '{fallback_var_name}') not found in trace. Skipping plot.")
             return
             
    if time_coord_name not in trace.posterior.coords:
         print(f"Time coordinate '{time_coord_name}' not found in trace. Skipping plot.")
         return

    # Extract latent popularity and time
    latent_popularity = trace.posterior[latent_var_name]
    latent_time_values = pd.to_datetime(trace.posterior[time_coord_name].values)

    # Filter latent popularity data by date range
    date_mask = (latent_time_values >= start_date) & (latent_time_values <= end_date)
    latent_popularity_filtered = latent_popularity.isel({time_coord_name: date_mask})
    latent_time_filtered = latent_time_values[date_mask]

    if latent_time_filtered.size == 0:
         print("No latent popularity data found within the specified date range. Skipping plot.")
         return

    # Calculate mean and HDI
    mean_latent_popularity = latent_popularity_filtered.mean(dim=["chain", "draw"])
    hdi_result_dataset = az.hdi(latent_popularity_filtered, hdi_prob=0.94) # Returns xr.Dataset

    # Filter polls data
    polls_train['date'] = pd.to_datetime(polls_train['date'])
    polls_filtered = polls_train[(polls_train['date'] >= start_date) & (polls_train['date'] <= end_date)]

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(political_families))) # Use tab10 colormap
    party_color_map = {party: colors[i] for i, party in enumerate(political_families)}

    # Plot latent popularity (mean and HDI)
    for i, party in enumerate(political_families):
        color = _get_party_color(party) # Use defined color
        # Plot mean line (Remove label)
        ax.plot(latent_time_filtered, mean_latent_popularity.sel(parties_complete=party),
                color=color, linewidth=1.5)
        # Plot HDI area (Remove label)
        # Select the DataArray from the HDI dataset using the original variable name
        hdi_data_array = hdi_result_dataset[latent_var_name]
        # Select the party from the DataArray
        hdi_party_da = hdi_data_array.sel(parties_complete=party)
        # Use integer position selection assuming [lower, upper] order
        lower_bound = hdi_party_da.isel(hdi=0) # Index 0 for lower bound
        upper_bound = hdi_party_da.isel(hdi=1) # Index 1 for upper bound
        # Convert DataArrays to numpy arrays before passing to matplotlib
        ax.fill_between(latent_time_filtered, lower_bound.values, upper_bound.values,
                        color=color, alpha=0.2)

    # Plot observed polls (Remove label)
    for i, party in enumerate(political_families):
        color = _get_party_color(party) # Use defined color
        party_polls = polls_filtered[polls_filtered[party] > 0] # Only plot where party exists
        if party not in party_polls.columns:
             continue
        if party_polls[party].max() > 1.5:
             if 'sample_size' in party_polls.columns and party_polls['sample_size'].notna().all() and (party_polls['sample_size'] > 0).all():
                  poll_percentages = party_polls[party] / party_polls['sample_size']
             else:
                  print(f"Warning: Cannot calculate percentage for polls for {party}. Skipping.")
                  continue
        else:
             poll_percentages = party_polls[party]

        ax.scatter(party_polls['date'], poll_percentages,
                   color=color, alpha=0.5, s=15, marker='o')

    # Formatting
    ax.set_title(f'Latent Popularity Trend and Polls: {start_date.date()} to {end_date.date()}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Vote Percentage')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    # Add vertical lines *with labels* here for the legend
    line_last_election = ax.axvline(start_date, color='grey', linestyle=':', linewidth=1.5, label=f'Last Election ({start_date.date()})')
    line_target_election = ax.axvline(end_date, color='red', linestyle=':', linewidth=1.5, label=f'Target Election ({end_date.date()})')

    # --- Create Legend Manually --- 
    legend_handles = []
    # Add handles for parties
    for party in political_families:
         legend_handles.append(Line2D([0], [0], color=_get_party_color(party), lw=2, label=party))
    # Add handles for vertical lines
    legend_handles.append(line_last_election)
    legend_handles.append(line_target_election)
    
    # Use the manually created handles for the legend
    ax.legend(handles=legend_handles, title='Party / Event', bbox_to_anchor=(1.04, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # Save the plot
    filename = os.path.join(output_dir, "latent_trend_since_last_election.png")
    plt.savefig(filename)
    print(f"Latent trend plot saved to {filename}")
    plt.close(fig) 

def plot_seat_distribution_histograms(seats_df: pd.DataFrame, output_dir: str, 
                                  date_mode: str = "election_day", 
                                  filename="seat_distribution_histograms.png"):
    """
    Generates histograms of predicted seat distributions for each party from simulation samples.

    Args:
        seats_df (pd.DataFrame): DataFrame where each row is a sample of predicted seats.
                                 Columns should include party names and their seat counts.
                                 May also include 'diaspora_scenario_applied' and 'original_sample_id'.
        output_dir (str): Directory to save the plot.
        date_mode (str): Prediction date mode (e.g., 'election_day') for the plot title and filename.
        filename (str): Custom filename for the plot.
    """
    if seats_df is None or seats_df.empty:
        print("Seat distribution data is empty. Skipping histogram plot.")
        return

    # Identify party columns by excluding known non-party columns
    # and assuming others are party seat counts (numeric).
    known_non_party_cols = ['diaspora_scenario_applied', 'original_sample_id', 'sample_index'] # Add any other known non-party cols
    
    party_columns = []
    for col in seats_df.columns:
        if col not in known_non_party_cols:
            # Attempt to convert to numeric to see if it's a seat count column
            try:
                pd.to_numeric(seats_df[col])
                party_columns.append(col)
            except (ValueError, TypeError):
                # If conversion fails, it's not a numeric party seat column
                print(f"Debug plot_seat_hist: Column '{col}' is not numeric, excluded from party columns.")
                pass 
    
    # Ensure party_columns are sorted for consistent plot order if desired
    party_columns = sorted(party_columns)

    if not party_columns:
        print("No valid party columns found for seat distribution histograms. Skipping plot.")
        return

    num_parties = len(party_columns)
    if num_parties == 0:
        print("No party columns found in seats_df. Skipping seat distribution histograms.")
        return

    # Determine number of rows and columns for the subplot grid
    # Aim for a somewhat square layout, but prioritize more columns if many parties
    if num_parties <= 4:
        ncols = num_parties
        nrows = 1
    elif num_parties <= 8:
        ncols = 4
        nrows = 2
    elif num_parties <= 9: # Special case for 3x3
        ncols = 3
        nrows = 3
    elif num_parties <= 12:
        ncols = 4
        nrows = 3
    else: # Default for more than 12 parties, might need adjustment
        ncols = 4
        nrows = (num_parties + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = np.array(axes).flatten() # Flatten to 1D array for easier iteration

    for i, party in enumerate(party_columns):
        if i >= len(axes):
            break # Should not happen if nrows/ncols are calculated correctly
        ax = axes[i]
        party_color = _get_party_color(party) # Use helper to get color
        
        # Actual seat counts for the party
        seat_counts = seats_df[party].astype(int)
        
        # Calculate mean and median
        mean_seats = seat_counts.mean()
        median_seats = seat_counts.median()
        
        # Determine a reasonable number of bins
        # Handle cases where min and max are the same to avoid issues with range
        min_val = seat_counts.min()
        max_val = seat_counts.max()
        if min_val == max_val:
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1) # Ensure at least one bin
        else:
            # Use integer bins from min to max seat count
            bins = np.arange(min_val -0.5, max_val + 1.5, 1) # Bins centered on integers
        
        sns.histplot(seat_counts, bins=bins, kde=False, ax=ax, color=party_color, stat="probability")
        
        ax.set_title(f'{party}', fontsize=12)
        ax.set_xlabel('Number of Seats', fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add vertical lines for mean and median
        ax.axvline(mean_seats, color='k', linestyle='dashed', linewidth=1, label=f'Mean: {mean_seats:.1f}')
        ax.axvline(median_seats, color='dimgray', linestyle='dotted', linewidth=1, label=f'Median: {median_seats:.1f}')
        ax.legend(fontsize=8)

        # Ensure x-axis ticks are integers
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Hide any unused subplots
    for j in range(num_parties, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Predicted Seat Distributions ({date_mode.replace("_", " ").title()})', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    
    # Construct filename
    # The 'filename' argument passed to the function can be used directly if it's already specific.
    # If not, we can construct it like before:
    # filename_to_use = filename if filename != "seat_distribution_histograms.png" else f"seat_histograms_{date_mode}.png"
    # For now, let's assume the passed filename is the intended one:
    final_filename = os.path.join(output_dir, filename)
    
    plt.savefig(final_filename)
    print(f"Seat distribution histograms saved to {final_filename}")
    plt.close(fig)

def plot_poll_bias_forest(elections_model, output_dir):
    """
    Generates a forest plot for the poll_bias parameter.

    Args:
        elections_model: The ElectionsFacade instance containing the trace.
        output_dir: Directory to save the plot.
    """
    if elections_model.trace is None or 'posterior' not in elections_model.trace:
        print("Warning: No posterior trace found. Skipping poll bias forest plot.")
        return

    idata = elections_model.trace
    var_name = "poll_bias"

    if var_name not in idata.posterior:
        print(f"Warning: Variable '{var_name}' not found in posterior data. Skipping poll bias forest plot.")
        return

    try:
        print(f"Generating forest plot for {var_name}...")
        plot_kwargs = {
            "var_names": [var_name],
            "hdi_prob": 0.94,
            "figsize": (10, max(4, idata.posterior[var_name].sizes.get("parties_complete", 8) * 0.5)), # Adjust height based on number of parties
            "backend": "matplotlib",
            "combined": True,  # Plot chains combined
            "textsize": 10
        }

        az.plot_forest(idata, **plot_kwargs)
        plt.axvline(0, color='grey', linestyle='--', label='Zero Bias') # Add vertical line at zero
        plt.title("Posterior Distribution of Average Poll Bias (Relative)")
        plt.xlabel("Bias (Logit Scale)")
        plt.tight_layout()

        filename = os.path.join(output_dir, "forest_plot_poll_bias.png")
        plt.savefig(filename)
        plt.close()
        print(f"Poll bias forest plot saved to {filename}")

    except Exception as e:
        print(f"Error generating forest plot for {var_name}: {e}")
        if plt.gcf().get_axes(): # Close plot if axes were created but error occurred
            plt.close() 