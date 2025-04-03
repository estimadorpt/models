import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data.dataset import ElectionDataset
import os
import numpy as np
import xarray as xr
import arviz as az
from typing import TYPE_CHECKING
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

def plot_latent_popularity_vs_polls(elections_model: 'ElectionsFacade', output_dir: str):
    """
    Generates and saves plots comparing estimated latent popularity (mean + HDI) 
    against raw poll data for each historical election cycle.

    Parameters:
    ----------
    elections_model : ElectionsFacade
        The facade object containing the dataset and inference trace.
    output_dir : str
        The directory where the plots will be saved.
    """
    print(f"Generating latent popularity vs polls plots in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Use the new variable name
    target_var = "latent_popularity_trajectory"

    if elections_model.trace is None or target_var not in elections_model.trace.posterior:
        print(f"Error: Trace data or '{target_var}' not found in the model.")
        # Optionally, print available variables for debugging
        if elections_model.trace is not None:
            print(f"Available variables in trace.posterior: {list(elections_model.trace.posterior.keys())}")
        return

    dataset = elections_model.dataset
    trace_posterior = elections_model.trace.posterior
    latent_pop_data = trace_posterior[target_var] # Use the target variable
    political_families = dataset.political_families
    historical_election_dates = dataset.historical_election_dates

    # Ensure trace coordinates match dataset elections
    if not set(historical_election_dates).issubset(set(latent_pop_data['elections'].values)):
         print("Warning: Mismatch between historical election dates in dataset and trace coordinates.")
         # Use elections available in both
         common_elections = sorted(list(set(historical_election_dates) & set(latent_pop_data['elections'].values)))
         if not common_elections:
             print("Error: No common election dates found between dataset and trace.")
             return
         print(f"Plotting for common elections: {common_elections}")
         elections_to_plot = common_elections
    else:
         elections_to_plot = sorted(historical_election_dates)


    for idx, election_date_str in enumerate(elections_to_plot):
        election_date_dt = pd.to_datetime(election_date_str)
        print(f"  - Plotting for election: {election_date_str}")

        # Determine the start date for the plot
        if idx == 0: # First election in the list
            # Default to 2 years before election day
            plot_start_date = election_date_dt - pd.Timedelta(days=365*2) 
        else:
            # Use the date of the previous election
            previous_election_str = elections_to_plot[idx-1]
            plot_start_date = pd.to_datetime(previous_election_str)
        print(f"    Plotting data from: {plot_start_date.date()}")

        # --- Debug: Print structure before access ---
        # Use the correct variable for debugging
        print(f"    Structure of trace_posterior[{target_var}]:") 
        print(f"      Coords: {latent_pop_data.coords}")
        print(f"      Dims: {latent_pop_data.dims}")
        # --- End Debug ---

        try:
            # Select latent popularity for the specific election using the correct variable
            latent_pop_election = latent_pop_data.sel(elections=election_date_str)
            
            # Calculate mean and HDI across chains and draws
            mean_latent_pop = latent_pop_election.mean(dim=["chain", "draw"])
            hdi_latent_pop = az.hdi(latent_pop_election, hdi_prob=0.94) # Results in xarray with 'hdi' coord

            # Get countdown values and convert to full date range
            countdown_values = latent_pop_election.coords["countdown"].values
            # Reverse countdown to get dates relative to election_date
            dates = election_date_dt - pd.to_timedelta(countdown_values, unit='D') 
            
            # --- Filter dates and corresponding data based on plot_start_date ---
            date_mask = dates >= plot_start_date
            filtered_dates = dates[date_mask]
            if len(filtered_dates) == 0:
                print(f"    Skipping {election_date_str}: No trajectory data after {plot_start_date.date()}")
                continue
            # --- End Filter --- 

            # Filter polls for this specific election cycle AND date range
            cycle_polls = dataset.polls[dataset.polls['election_date'] == election_date_dt].copy()
            filtered_cycle_polls = cycle_polls[cycle_polls['date'] >= plot_start_date]
            # Note: No need to print skip message here, as we might still plot the trajectory even if no polls in the window
            
            # Convert poll percentages (using filtered polls) to 0-1 if needed
            if not filtered_cycle_polls.empty and filtered_cycle_polls[political_families].max().max() > 1.1: 
                 print(f"    Converting poll percentages from 0-100 to 0-1 for {election_date_str}")
                 for party in political_families:
                     # Apply conversion only to the filtered df to avoid modifying original data elsewhere
                     filtered_cycle_polls[party] = filtered_cycle_polls[party] / 100.0
            
            # --- Create the plot for this election ---
            fig, ax = plt.subplots(figsize=(15, 8))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(political_families)))

            # --- Renormalize poll data for plotting --- 
            renormalized_polls = filtered_cycle_polls.copy()
            modeled_parties_sum = renormalized_polls[political_families].sum(axis=1)
            # Avoid division by zero if sum is zero (e.g., very old polls with no modeled parties)
            modeled_parties_sum = modeled_parties_sum.replace(0, np.nan)
            for party in political_families:
                renormalized_polls[party] = renormalized_polls[party] / modeled_parties_sum
            # --- End Renormalization ---

            for i, party in enumerate(political_families):
                color = colors[i]
                
                # Extract party data from calculated mean and HDI
                mean_party = mean_latent_pop.sel(parties_complete=party)
                # Access the DataArray within the HDI Dataset first
                hdi_da = hdi_latent_pop['latent_popularity_trajectory']
                # Now select the party and HDI bounds from the DataArray
                hdi_party_lower = hdi_da.sel(parties_complete=party, hdi='lower')
                hdi_party_higher = hdi_da.sel(parties_complete=party, hdi='higher')

                # Apply date mask to the numpy arrays for this party
                filtered_mean_party = mean_party.values[date_mask]
                filtered_hdi_lower = hdi_party_lower.values[date_mask]
                filtered_hdi_higher = hdi_party_higher.values[date_mask]

                # Plot mean latent popularity (filtered)
                ax.plot(filtered_dates, filtered_mean_party, label=party, color=color, lw=2)
                
                # Plot HDI region (filtered)
                ax.fill_between(filtered_dates, filtered_hdi_lower, filtered_hdi_higher, color=color, alpha=0.2)

                # Plot raw polls for this party (filtered and RENORMALIZED)
                # Use the renormalized_polls DataFrame
                polls_party = renormalized_polls[['date', party]].dropna()
                if not polls_party.empty:
                    ax.scatter(polls_party['date'], polls_party[party], color=color, alpha=0.6, s=15, label=f'_nolegend_')

            # Add election day vertical line
            ax.axvline(election_date_dt, color='black', linestyle='--', linewidth=1.5, label='Election Day')

            ax.set_title(f'Latent Popularity vs Polls - Election {election_date_str}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Vote Percentage')
            ax.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, which='major', linestyle='--', linewidth=0.5)
            ax.set_ylim(0, None) # Start y-axis at 0
            
            # Set x-axis limits based on filtered dates
            ax.set_xlim(left=plot_start_date, right=election_date_dt + pd.Timedelta(days=10))
            
            # Improve date formatting on x-axis
            fig.autofmt_xdate() 

            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
            
            # Save the figure
            plot_filename = os.path.join(output_dir, f"latent_vs_polls_{election_date_str}.png")
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig) # Close the figure to free memory
            print(f"    Saved plot: {plot_filename}")

        except KeyError as e:
             # Improved error message
             print(f"    Skipping {election_date_str}: KeyError encountered. Variable/Coord causing error: {e}.")
             print(f"    Check if '{e}' exists in latent_pop_election or hdi_latent_pop structure shown above.")
        except Exception as e:
             print(f"    Failed to plot {election_date_str}: {e}")
             import traceback
             traceback.print_exc() # Print full traceback for debugging

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