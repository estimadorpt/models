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
if TYPE_CHECKING:
    from src.models.elections_facade import ElectionsFacade

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

def plot_latent_popularity_vs_polls(
    elections_model: 'ElectionsFacade', 
    output_dir: str,
    include_target_date: bool = False
):
    """
    Generates and saves plots comparing estimated latent popularity (mean + HDI) 
    against raw poll data for each historical election cycle AND the target election cycle. 
    Adds arrows to show the difference between raw polls and model's noisy popularity estimate.

    Parameters:
    ----------
    elections_model : ElectionsFacade
        The facade object containing the dataset and inference trace.
    output_dir : str
        The directory where the plots will be saved.
    include_target_date : bool, optional
        Included for compatibility, the target date is now always plotted if possible.
    """
    print(f"Generating latent popularity vs polls plots in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    latent_target_var = "latent_popularity_trajectory"
    noisy_target_var = "noisy_popularity"

    if elections_model.trace is None or latent_target_var not in elections_model.trace.posterior:
        print(f"Error: Trace data or '{latent_target_var}' not found in posterior trace.")
        return

    trace_posterior = elections_model.trace.posterior
    latent_pop_data = trace_posterior[latent_target_var]
    trace_elections = list(latent_pop_data['elections'].values) # Elections available in trace

    # Check for noisy popularity and 'observations' coordinate upfront
    plot_arrows = False
    noisy_pop_data = None
    if noisy_target_var in trace_posterior:
        noisy_pop_data = trace_posterior[noisy_target_var]
        if 'observations' in noisy_pop_data.coords:
            plot_arrows = True
            print(f"Found '{noisy_target_var}' with 'observations' coord. Will plot adjustment arrows.")
        else:
            print(f"Warning: Found '{noisy_target_var}' but 'observations' coordinate is missing. Cannot draw arrows.")
    else:
        print(f"Warning: '{noisy_target_var}' not found in posterior trace. Cannot draw adjustment arrows.")

    dataset = elections_model.dataset
    political_families = dataset.political_families
    historical_election_dates = sorted(list(dataset.historical_election_dates))
    target_election_date_str = elections_model.election_date # Target date from facade

    # Determine the list of election dates to generate plots for
    elections_to_plot = list(historical_election_dates) # Start with historical
    
    # Add target date if it's different from the last historical AND present in the trace coordinates
    last_hist_date_str = historical_election_dates[-1] if historical_election_dates else None
    target_date_is_distinct_and_in_trace = (
        target_election_date_str is not None and
        target_election_date_str != last_hist_date_str and
        target_election_date_str in trace_elections
    )
    
    if target_date_is_distinct_and_in_trace:
        elections_to_plot.append(target_election_date_str)
        print(f"Adding target election date '{target_election_date_str}' to plots.")
    elif target_election_date_str is not None and target_election_date_str != last_hist_date_str:
        print(f"Warning: Target election date '{target_election_date_str}' not found in trace coordinates ('elections'). Cannot plot its specific latent trend.")
    
    if not elections_to_plot:
        print("Error: No election dates identified for plotting.")
        return

    print(f"Will generate plots for election cycles ending: {elections_to_plot}")

    # --- Loop Through Election Cycles ---
    for idx, election_date_str in enumerate(elections_to_plot):
        election_date_dt = pd.to_datetime(election_date_str)
        print(f"  - Plotting for cycle ending: {election_date_str}")

        is_target_cycle = (election_date_str == target_election_date_str and target_date_is_distinct_and_in_trace)
        
        # Determine plot start/end dates and title
        if is_target_cycle:
            # Target cycle starts after the last historical election
            plot_start_date = pd.to_datetime(last_hist_date_str) + pd.Timedelta(days=1)
            plot_end_date = election_date_dt # End at the target date
            plot_title = f'Latent Popularity vs Polls for Target Election {election_date_str}'
            data_selection_election_str = election_date_str # Use target date to select data
            print(f"    (Target Cycle Plot: {plot_start_date.date()} to {plot_end_date.date()})")
        elif idx == 0: # First historical election
            plot_start_date = election_date_dt - pd.Timedelta(days=365*2) # Default lookback
            plot_end_date = election_date_dt
            plot_title = f'Latent Popularity vs Polls for {election_date_str} Election'
            data_selection_election_str = election_date_str
            print(f"    (Historical Cycle Plot: {plot_start_date.date()} to {plot_end_date.date()})")
        else: # Subsequent historical elections
            previous_election_str = elections_to_plot[idx-1]
            plot_start_date = pd.to_datetime(previous_election_str) + pd.Timedelta(days=1)
            plot_end_date = election_date_dt
            plot_title = f'Latent Popularity vs Polls for {election_date_str} Election'
            data_selection_election_str = election_date_str
            print(f"    (Historical Cycle Plot: {plot_start_date.date()} to {plot_end_date.date()})")
            
        # --- Filter Latent Popularity Data ---
        try:
            # Select using the election date string for the cycle being processed
            if data_selection_election_str not in latent_pop_data['elections'].values:
                 print(f"Error: Election date '{data_selection_election_str}' not found in latent_pop_data 'elections' coordinates: {latent_pop_data['elections'].values}")
                 continue 
            latent_pop_election = latent_pop_data.sel(elections=data_selection_election_str)
        except Exception as e:
            print(f"Error selecting latent data for election '{data_selection_election_str}': {e}")
            continue

        # Calculate mean and HDI
        latent_mean = latent_pop_election.mean(dim=["chain", "draw"])
        latent_hdi = az.hdi(latent_pop_election, hdi_prob=0.94)

        # Get countdown and corresponding dates relative to the cycle's end date
        countdown_values = latent_pop_data['countdown'].values
        # Note: dates are calculated relative to the END date of the selected data cycle (data_selection_election_str)
        # This is correct as the GP is defined relative to its election endpoint.
        cycle_dates = pd.to_datetime(data_selection_election_str) - pd.to_timedelta(countdown_values, unit='D')

        # Filter dates to the PLOT range (plot_start_date to plot_end_date)
        date_mask = (cycle_dates >= plot_start_date) & (cycle_dates <= plot_end_date)
        plot_dates_filtered = cycle_dates[date_mask]
        
        if len(plot_dates_filtered) == 0:
            print(f"    Warning: No latent trajectory time points found within the plot range {plot_start_date.date()} to {plot_end_date.date()}. Skipping plot.")
            continue
            
        # --- Filter Polls Data ---
        # Select polls associated with the current cycle's END date (election_date_str)
        # AND falling within the plot's date range.
        cycle_polls = dataset.polls_train[
            (dataset.polls_train['election_date'] == election_date_dt) & # Linked to this cycle end
            (dataset.polls_train['date'] >= plot_start_date) &
            (dataset.polls_train['date'] <= plot_end_date)
        ].copy()

        if cycle_polls.empty:
            print(f"    Warning: No polls found associated with election '{election_date_str}' within the range {plot_start_date.date()} to {plot_end_date.date()}.")
        
        # Convert poll counts to percentages
        for party in political_families:
            # Check if party column exists and handle potential missing columns safely
            if party in cycle_polls.columns and 'sample_size' in cycle_polls.columns:
                cycle_polls[party] = cycle_polls.apply(
                    lambda row: row[party] / row['sample_size'] if row['sample_size'] > 0 else 0, 
                    axis=1
                ).clip(0, 1) # Ensure valid percentage range
            elif party not in cycle_polls.columns:
                 print(f"    Debug: Column for party '{party}' not found in cycle_polls for {election_date_str}.")
                 cycle_polls[party] = 0.0 # Assign default if missing? Or handle differently?
            # Handle missing sample_size if necessary

        # Get election result (only for historical dates)
        cycle_result_pct = pd.DataFrame() # Default to empty
        if not is_target_cycle:
            cycle_result = dataset.results_mult[dataset.results_mult['election_date'] == election_date_dt]
            if not cycle_result.empty:
                cycle_result_pct = cycle_result.copy()
                for party in political_families:
                    if party in cycle_result_pct.columns and 'sample_size' in cycle_result_pct.columns:
                        cycle_result_pct[party] = cycle_result_pct.apply(
                            lambda row: row[party] / row['sample_size'] if row['sample_size'] > 0 else 0,
                            axis=1
                        ).clip(0, 1)
                    elif party not in cycle_result_pct.columns:
                        cycle_result_pct[party] = 0.0

        # --- Create Plot ---
        plt.figure(figsize=(15, 8))
        num_parties = len(political_families)
        colors = plt.cm.tab10(np.linspace(0, 1, num_parties)) if num_parties > 0 else []
        
        party_color_map = {party: colors[i] for i, party in enumerate(political_families)}

        # Plot latent popularity mean and HDI
        for i, party in enumerate(political_families):
            party_color = party_color_map.get(party, 'grey') # Use map
            
            # Apply date_mask correctly to latent data (indexed by countdown)
            latent_mean_party = latent_mean.sel(parties_complete=party).values[date_mask]
            hdi_party_da = latent_hdi[latent_target_var].sel(parties_complete=party)
            hdi_lower_filtered = hdi_party_da.isel(countdown=date_mask, hdi=0).values
            hdi_upper_filtered = hdi_party_da.isel(countdown=date_mask, hdi=1).values

            plt.plot(plot_dates_filtered, latent_mean_party, label=f'{party} (Latent Mean)', color=party_color, linestyle='-')
            plt.fill_between(plot_dates_filtered, hdi_lower_filtered, hdi_upper_filtered, color=party_color, alpha=0.2, label=f'_nolegend_') # No duplicate legend

            # Plot raw poll points
            if not cycle_polls.empty and party in cycle_polls.columns:
                plt.scatter(cycle_polls['date'], cycle_polls[party], color=party_color, marker='o', s=30, alpha=0.6, label=f'_nolegend_')
            
            # Plot arrows if possible
            if plot_arrows and not cycle_polls.empty and party in cycle_polls.columns:
                for poll_index, poll_row in cycle_polls.iterrows():
                    if poll_index not in noisy_pop_data['observations'].values:
                        continue # Skip if index mismatch
                       
                    raw_poll_y = poll_row[party]
                    poll_x = poll_row['date']
                    
                    try:
                        noisy_samples = noisy_pop_data.sel(observations=poll_index, parties_complete=party)
                        noisy_est_y = noisy_samples.mean().item()
                        
                        plt.arrow(
                            poll_x, raw_poll_y, 0, noisy_est_y - raw_poll_y, 
                            color=party_color, alpha=0.5, 
                            head_width=pd.Timedelta(days=1), # Adjust head width relative to date axis
                            head_length=0.01, # Adjust head length relative to percentage axis
                            length_includes_head=True, zorder=3
                        )
                    except Exception as e:
                        # print(f"Debug: Error processing arrow for poll index {poll_index}, party {party}: {e}") # Reduced verbosity
                        pass # Continue plotting other elements


        # Plot election result (if available for this cycle)
        if not cycle_result_pct.empty:
            result_date = cycle_result_pct['date'].iloc[0]
            for i, party in enumerate(political_families):
                 party_color = party_color_map.get(party, 'grey')
                 if party in cycle_result_pct.columns:
                     plt.scatter(result_date, cycle_result_pct[party].iloc[0], color=party_color, s=250, marker='X', edgecolor='black', zorder=5, label=f'_nolegend_')

        # --- Finalize Plot ---
        plt.title(plot_title)
        plt.xlabel('Date')
        plt.ylabel('Vote Percentage')
        plt.ylim(0, None)
        plt.axvline(plot_end_date, color='grey', linestyle=':', linewidth=2, label='Cycle End Date') # Use plot_end_date
        
        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = []
        for i, party in enumerate(political_families):
            party_color = party_color_map.get(party, 'grey')
            legend_elements.append(Line2D([0], [0], color=party_color, lw=2, label=party))
        
        legend_elements.append(Line2D([0], [0], marker='o', color='grey', linestyle='', markersize=5, alpha=0.6, label='Raw Poll'))
        if plot_arrows:
             legend_elements.append(Line2D([0], [0], color='grey', lw=0, marker='|', markersize=6, alpha=0.7, label='Model Est. Adj.'))
        if not is_target_cycle: # Only show result marker for historical
             legend_elements.append(Line2D([0], [0], marker='X', color='grey', linestyle='', markersize=8, markeredgecolor='black', label='Election Result'))
        legend_elements.append(Line2D([0], [0], color='grey', linestyle=':', linewidth=2, label='Cycle End Date'))
        
        plt.legend(handles=legend_elements, title='Party / Element', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Adjust x-axis limits slightly
        plt.xlim(plot_start_date - pd.Timedelta(days=5), plot_end_date + pd.Timedelta(days=10))
        plt.gcf().autofmt_xdate() # Auto-format dates
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Save the plot
        plot_filename = os.path.join(output_dir, f"latent_vs_polls_{election_date_str}.png")
        plt.savefig(plot_filename, dpi=300)
        print(f"    Saved plot: {plot_filename}")
        plt.close()

    print("Finished generating latent popularity plots.")

def plot_latent_component_contributions(elections_model: 'ElectionsFacade', output_dir: str):
    """
    Generates and saves plots showing the pre-softmax component contributions 
    (baseline, election baseline, time effect) to the latent popularity trajectory 
    for each party and historical election cycle.

    Parameters:
    ----------
    elections_model : ElectionsFacade
        The facade object containing the dataset and inference trace.
    output_dir : str
        The directory where the plots will be saved.
    """
    print(f"Generating latent component contribution plots in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Check required variables
    required_vars = ["party_baseline", "election_party_baseline", "party_time_effect"]
    if elections_model.trace is None:
        print("Error: Trace data not found.")
        return
    trace_posterior = elections_model.trace.posterior
    for var in required_vars:
        if var not in trace_posterior:
            print(f"Error: Required variable '{var}' not found in trace posterior.")
            print(f"Available variables: {list(trace_posterior.keys())}")
            return

    dataset = elections_model.dataset
    political_families = dataset.political_families
    historical_election_dates = dataset.historical_election_dates

    # Get coordinate data once
    party_baseline_mean = trace_posterior.party_baseline.mean(dim=["chain", "draw"])
    election_party_baseline_mean = trace_posterior.election_party_baseline.mean(dim=["chain", "draw"])
    party_time_effect_mean = trace_posterior.party_time_effect.mean(dim=["chain", "draw"])
    countdown_coord = trace_posterior.party_time_effect.coords["countdown"].values

    # Ensure trace coordinates match dataset elections
    trace_elections = trace_posterior.elections.values
    if not set(historical_election_dates).issubset(set(trace_elections)):
         print("Warning: Mismatch between historical election dates in dataset and trace coordinates.")
         common_elections = sorted(list(set(historical_election_dates) & set(trace_elections)))
         if not common_elections:
             print("Error: No common election dates found between dataset and trace.")
             return
         print(f"Plotting for common elections: {common_elections}")
         elections_to_plot = common_elections
    else:
         elections_to_plot = sorted(historical_election_dates)

    for idx, election_date_str in enumerate(elections_to_plot):
        election_date_dt = pd.to_datetime(election_date_str)
        print(f"  - Processing components for election: {election_date_str}")

        # Determine the start date for the plot
        if idx == 0: 
            plot_start_date = election_date_dt - pd.Timedelta(days=365*2) 
        else:
            previous_election_str = elections_to_plot[idx-1]
            plot_start_date = pd.to_datetime(previous_election_str)
        
        # Calculate date range for this election
        dates = election_date_dt - pd.to_timedelta(countdown_coord, unit='D')
        date_mask = dates >= plot_start_date
        filtered_dates = dates[date_mask]
        if len(filtered_dates) == 0:
            print(f"    Skipping {election_date_str}: No time points after {plot_start_date.date()}")
            continue

        for party in political_families:
            print(f"    - Plotting components for party: {party}")
            try:
                # Extract mean component values for this party/election
                pb_party = party_baseline_mean.sel(parties_complete=party).item() # Scalar
                epb_party = election_party_baseline_mean.sel(elections=election_date_str, parties_complete=party).item() # Scalar
                pte_party_full = party_time_effect_mean.sel(elections=election_date_str, parties_complete=party) # DataArray (countdown,)
                
                # Calculate total baseline and pre-softmax sum
                total_baseline_party = pb_party + epb_party
                mu_sum_party_full = total_baseline_party + pte_party_full
                
                # Filter time-varying components using the date mask
                pte_party_filtered = pte_party_full.values[date_mask]
                mu_sum_party_filtered = mu_sum_party_full.values[date_mask]

                # Create plot
                fig, ax = plt.subplots(figsize=(15, 7))

                # Plot constant baselines
                ax.axhline(pb_party, color='grey', linestyle=':', lw=1.5, label=f'Party Baseline ({pb_party:.2f})')
                ax.axhline(total_baseline_party, color='purple', linestyle='--', lw=1.5, label=f'Total Baseline ({total_baseline_party:.2f})')

                # Plot time-varying GP effect
                ax.plot(filtered_dates, pte_party_filtered, color='green', linestyle='-.', lw=2, label='Time Effect (GP)')

                # Plot the sum (input to softmax)
                ax.plot(filtered_dates, mu_sum_party_filtered, color='black', linestyle='-', lw=2.5, label='Sum (Pre-Softmax Mu)')

                # Add election day vertical line
                ax.axvline(election_date_dt, color='red', linestyle=':', linewidth=1.5, label='Election Day')

                ax.set_title(f'Latent Popularity Components (Pre-Softmax) - Election {election_date_str} - Party {party}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Component Value (Logit Scale)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, which='major', linestyle='--', linewidth=0.5)
                
                # Set x-axis limits
                ax.set_xlim(left=plot_start_date, right=election_date_dt + pd.Timedelta(days=10))
                fig.autofmt_xdate() 
                plt.tight_layout(rect=[0, 0, 0.85, 1])

                # Save the figure
                plot_filename = os.path.join(output_dir, f"latent_components_{election_date_str}_{party}.png")
                plt.savefig(plot_filename, dpi=150)
                plt.close(fig)

            except Exception as e:
                 print(f"    Failed to plot components for {party} in {election_date_str}: {e}")
                 import traceback
                 traceback.print_exc()

    print("Finished generating latent component contribution plots.")

def plot_recent_polls(elections_model: 'ElectionsFacade', output_dir: str):
    """
    Generates and saves a plot showing raw poll data collected since the 
    most recent historical election.

    Parameters:
    ----------
    elections_model : ElectionsFacade
        The facade object containing the dataset.
    output_dir : str
        The directory where the plots will be saved.
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

def plot_house_effects_heatmap(elections_model: 'ElectionsFacade', output_dir: str):
    """
    Generates and saves a heatmap of the mean posterior house effects for each pollster and party.

    Parameters:
    ----------
    elections_model : ElectionsFacade
        The facade object containing the inference trace.
    output_dir : str
        The directory where the plots will be saved.
    """
    print(f"Generating house effects heatmap in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    house_effect_var = "house_effects" # Variable name in the trace

    if elections_model.trace is None or house_effect_var not in elections_model.trace.posterior:
        print(f"Error: Trace data or '{house_effect_var}' not found in the model's posterior trace.")
        if elections_model.trace is not None:
            print(f"Available variables in trace.posterior: {list(elections_model.trace.posterior.keys())}")
        return

    trace_posterior = elections_model.trace.posterior
    house_effects_data = trace_posterior[house_effect_var]

    # Calculate the mean across chains and draws
    mean_house_effects = house_effects_data.mean(dim=["chain", "draw"])

    # Convert to pandas DataFrame for easier plotting with seaborn
    # Ensure coordinates are preserved correctly
    df_house_effects = mean_house_effects.to_dataframe(name='mean_effect').unstack()
    # The column names will be multi-index (mean_effect, party), simplify it
    df_house_effects.columns = df_house_effects.columns.droplevel(0)

    # Create the heatmap
    plt.figure(figsize=(max(10, len(mean_house_effects.parties_complete)*1.2), 
                          max(8, len(mean_house_effects.pollsters)*0.5)))
    
    # Use a diverging colormap centered at 0
    sns.heatmap(df_house_effects, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=.5)
    
    plt.title('Mean House Effects (Latent Space Adjustment)')
    plt.xlabel('Party')
    plt.ylabel('Pollster')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the figure
    plot_filename = os.path.join(output_dir, "house_effects_heatmap.png")
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Saved heatmap: {plot_filename}") 

    print("Finished generating house effects heatmap.")

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
    latent_pop_df = latent_pop_samples.stack(sample=('chain', 'draw')).to_dataframe().reset_index()
    
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