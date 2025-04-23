"""
Analyzes historical district-level election results to understand vote patterns
for specific parties, focusing on:
- First appearance year (overall and per district).
- Vote share in the first appearance election.
- Frequency of zero or low vote shares.

This helps diagnose potential data-related issues that might affect model parameters
like beta sensitivity, especially for newer or smaller parties.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Assuming the script is run from the root 'models' directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.data.dataset import ElectionDataset
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure you are running this script from the root 'models' directory,")
    print("or that the 'src' directory is in your PYTHONPATH.")
    sys.exit(1)

def analyze_vote_patterns(dataset: ElectionDataset, parties_to_analyze: list):
    """
    Analyzes vote patterns for the specified parties using the loaded dataset.

    Args:
        dataset: An initialized ElectionDataset object.
        parties_to_analyze: A list of party acronyms (strings) to analyze.
    """
    if not hasattr(dataset, 'results_mult_district') or dataset.results_mult_district is None or dataset.results_mult_district.empty:
        print("Error: District results (results_mult_district) not found or empty in the dataset.")
        return

    results_df = dataset.results_mult_district.copy()

    # Ensure 'election_date' is datetime
    results_df['election_date'] = pd.to_datetime(results_df['election_date'])
    results_df['year'] = results_df['election_date'].dt.year

    print("\n--- Data Properties Check ---")
    print(f"Total district-election rows: {len(results_df)}")
    for party in parties_to_analyze:
        if party in results_df.columns:
            nan_count = results_df[party].isna().sum()
            if nan_count > 0:
                print(f"!!! Found {nan_count} NaN values for party '{party}' !!!")
            # Check NaNs specifically for 2011 if relevant
            if party == 'PAN' and 2011 in results_df['year'].unique():
                nan_count_2011 = results_df[results_df['year'] == 2011][party].isna().sum()
                if nan_count_2011 > 0:
                    print(f"    -> Found {nan_count_2011} NaN values for PAN specifically in 2011 data.")
        else:
            print(f"Party '{party}' not found for NaN check.")

    # --- Fill NaNs with 0 BEFORE calculating shares/frequencies --- ##
    # Based on hypothesis that NaNs should be zeros for missing party results
    print("\nFilling NaN values with 0 before analysis...")
    parties_present = [p for p in parties_to_analyze if p in results_df.columns]
    results_df[parties_present] = results_df[parties_present].fillna(0)
    # ------------------------------------------------------------ ##

    # --- Debug: Check specific Braganca 2011 row ---
    braganca_2011 = results_df[(results_df['Circulo'] == 'Bragança') & (results_df['year'] == 2011)]
    if not braganca_2011.empty:
        print("\n--- DEBUG: Bragança 2011 Data (from results_mult_district) ---")
        print(braganca_2011[['Circulo', 'year', 'PAN']].to_string())
        print("-----------------------------------------------------------")
    else:
        print("\n--- DEBUG: Bragança 2011 row NOT FOUND in results_mult_district ---")
    # --- End Debug ---

    print("\n--- Analyzing Vote Patterns (after filling NaNs) ---")
    print(f"Parties: {', '.join(parties_to_analyze)}")
    print(f"Districts: {results_df['Circulo'].nunique()}")
    print(f"Elections: {sorted(results_df['year'].unique())}")

    analysis_summary = {}

    for party in parties_to_analyze:
        if party not in results_df.columns:
            print(f"\nWarning: Party '{party}' not found in results columns. Skipping.")
            continue

        print(f"\n--- Analysis for {party} ---")
        party_summary = {}

        # Ensure sample_size column exists
        if 'sample_size' not in results_df.columns:
            print("Error: 'sample_size' column missing from results data. Cannot calculate shares.")
            continue
        if results_df['sample_size'].eq(0).any():
            print("Warning: Some sample sizes are zero. Shares for these entries will be NaN.")
            # Avoid division by zero, calculate share only where sample_size > 0
            results_df[f'{party}_share'] = np.where(
                results_df['sample_size'] > 0,
                results_df[party] / results_df['sample_size'],
                np.nan # Assign NaN where sample_size is 0
            )
        else:
             # Calculate share for the current party
             results_df[f'{party}_share'] = results_df[party] / results_df['sample_size']


        # Filter data for the party where share > 0
        # Use the calculated share column for analysis of presence
        party_results = results_df[results_df[f'{party}_share'] > 0][['year', 'Circulo', f'{party}_share']]

        # 1. First Appearance (Overall)
        if not party_results.empty:
            first_year_overall = party_results['year'].min()
            party_summary['first_year_overall'] = first_year_overall
            print(f"First year with non-zero votes (any district): {first_year_overall}")

            # Vote share in first year (average across districts where they appeared)
            first_year_votes = party_results[party_results['year'] == first_year_overall]
            avg_first_year_share = first_year_votes[f'{party}_share'].mean() * 100 # Use share column, as percentage
            party_summary['avg_share_first_year'] = avg_first_year_share
            print(f"Avg. vote share in first year (where present): {avg_first_year_share:.2f}%")

            # First Appearance (Per District)
            first_year_per_district = party_results.groupby('Circulo')['year'].min()
            party_summary['first_year_per_district'] = first_year_per_district.to_dict()
            # print(f"First year per district:\n{first_year_per_district}") # Can be verbose

        else:
            party_summary['first_year_overall'] = None
            party_summary['avg_share_first_year'] = None
            party_summary['first_year_per_district'] = {}
            print(f"Party '{party}' never had non-zero votes in the dataset.")

        # 2. Frequency of Zero Votes (still check raw count == 0 for this)
        total_observations = len(results_df)
        # It's safer to check if the RAW vote count is 0, not the share,
        # in case of floating point inaccuracies if share was calculated.
        zero_votes = results_df[results_df[party] == 0]
        zero_vote_freq = (len(zero_votes) / total_observations) * 100
        party_summary['zero_vote_freq_pct'] = zero_vote_freq
        print(f"Frequency of zero vote share: {zero_vote_freq:.2f}% ({len(zero_votes)}/{total_observations} district-elections)")

        # 3. Frequency of Low Votes (use the calculated share)
        low_vote_threshold = 0.005 # 0.5%
        # Use the share column and filter out exact zeros which are covered above
        low_votes = results_df[(results_df[f'{party}_share'] > 0) & (results_df[f'{party}_share'] < low_vote_threshold)]
        low_vote_freq = (len(low_votes) / total_observations) * 100
        party_summary['low_vote_freq_pct'] = low_vote_freq
        print(f"Frequency of low (0% < share < {low_vote_threshold*100:.1f}%) vote share: {low_vote_freq:.2f}% ({len(low_votes)}/{total_observations} district-elections)")

        # --- NEW: 4. Distribution of Non-Zero Vote Shares ---
        non_zero_shares = results_df[results_df[f'{party}_share'] > 0][f'{party}_share']
        if not non_zero_shares.empty:
             print("\n  Distribution of NON-ZERO Vote Shares:")
             desc = non_zero_shares.describe(percentiles=[.1, .25, .5, .75, .9])
             print(f"    Count: {desc.loc['count']:.0f}")
             print(f"    Mean:  {desc.loc['mean']*100:.2f}%")
             print(f"    Std:   {desc.loc['std']*100:.2f}%")
             print(f"    Min:   {desc.loc['min']*100:.2f}%")
             print(f"    10%:   {desc.loc['10%']*100:.2f}%")
             print(f"    25%:   {desc.loc['25%']*100:.2f}%")
             print(f"    50% (Median): {desc.loc['50%']*100:.2f}%")
             print(f"    75%:   {desc.loc['75%']*100:.2f}%")
             print(f"    90%:   {desc.loc['90%']*100:.2f}%")
             print(f"    Max:   {desc.loc['max']*100:.2f}%")
             # Store key stats for summary table if needed later
             party_summary['non_zero_mean_share'] = desc.loc['mean']
             party_summary['non_zero_median_share'] = desc.loc['50%']
             party_summary['non_zero_std_share'] = desc.loc['std']
        else:
             print("\n  No non-zero vote shares found for distribution analysis.")
        # --- END NEW --- 

        # Clean up the temporary share column (optional, but good practice)
        results_df.drop(columns=[f'{party}_share'], inplace=True)

        analysis_summary[party] = party_summary

    print("\n--- Comparison Summary ---")
    print(f"{'Party':<10} | {'First Year':<12} | {'Avg Share 1st Yr (%)':<20} | {'Zero Freq (%)':<15} | {'Low Freq (<0.5%) (%)':<20}")
    print("-" * 85)
    for party, summary in analysis_summary.items():
        fy = summary.get('first_year_overall', 'N/A')
        as1y = f"{summary.get('avg_share_first_year', 0):.2f}" if summary.get('avg_share_first_year') is not None else 'N/A'
        zf = f"{summary.get('zero_vote_freq_pct', 0):.2f}"
        lf = f"{summary.get('low_vote_freq_pct', 0):.2f}"
        print(f"{party:<10} | {fy:<12} | {as1y:<20} | {zf:<15} | {lf:<20}")

    return analysis_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Party Vote Patterns in Historical Election Data")
    parser.add_argument(
        "--election-date",
        default=datetime.now().strftime('%Y-%m-%d'), # Default to today for context
        help="Target election date (YYYY-MM-DD) to provide context for dataset loading (usually doesn't affect historical data loading)",
    )
    parser.add_argument(
        "--parties",
        nargs='+',
        default=["PAN", "L", "IL"], # Default to the parties of interest
        help="List of party acronyms to analyze (e.g., PAN L IL PS AD)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (currently minimal effect)",
    )

    args = parser.parse_args()

    print(f"Using Election Date Context: {args.election_date}")

    # Instantiate ElectionDataset - this handles loading polls and results
    # We need to provide timescales even if not directly used by this script.
    # Use defaults similar to src/main.py
    default_baseline_timescale = [365*4] # Example default, adjust if needed
    default_election_timescale = [90]    # Example default, adjust if needed
    try:
        dataset = ElectionDataset(
            election_date=args.election_date,
            baseline_timescales=default_baseline_timescale,
            election_timescales=default_election_timescale
        )
    except FileNotFoundError as e:
         print(f"Error: Data file not found during dataset initialization: {e}")
         print("Ensure the necessary data files (e.g., legislativas_*.parquet) are present in the 'data/' directory.")
         sys.exit(1)
    except Exception as e:
         print(f"Error initializing ElectionDataset: {e}")
         sys.exit(1)


    # Run the analysis
    analyze_vote_patterns(dataset, args.parties)

    print("\nAnalysis complete.") 