import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data.dataset import ElectionDataset

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

# Example Usage (requires creating a dataset object first)
# if __name__ == '__main__':
#     # Create a dummy dataset object for testing - replace with actual dataset creation
#     # This requires your ElectionDataset class and data loading functions to be accessible
#     try:
#         # Assuming your dataset script is runnable and can create an object
#         # You might need to adjust paths or parameters
#         from src.config import ELECTION_DATE, BASELINE_TIMESCALE, ELECTION_TIMESCALE, TEST_CUTOFF
#         dataset = ElectionDataset(
#             election_date=ELECTION_DATE, 
#             baseline_timescales=BASELINE_TIMESCALE, 
#             election_timescales=ELECTION_TIMESCALE,
#             test_cutoff=TEST_CUTOFF
#         )
#         plot_election_data(dataset)
#     except ImportError as e:
#         print(f"Could not import ElectionDataset or config. Make sure paths are correct. Error: {e}")
#     except Exception as e:
#         print(f"An error occurred during dataset creation or plotting: {e}") 