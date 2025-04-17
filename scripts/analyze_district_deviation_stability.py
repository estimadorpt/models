import pandas as pd
import numpy as np
import glob
import os

# --- Imports from project ---
# Add src to path if necessary, or run from root with python -m scripts.analyze...
import sys
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.loaders import load_election_results
from src.data.dataset import ElectionDataset # To get historical dates and parties

# --- Statsmodels Import ---
try:
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels not found. Run 'pip install statsmodels' to enable sensitivity fit analysis.")
    STATSMODELS_AVAILABLE = False
# --- End Statsmodels Import ---

# --- Configuration ---
# No longer need ELECTION_RESULTS_DIR or FILE_PATTERN
# Number of top variable/poor fit combinations to display
TOP_N = 20
# Columns expected from load_election_results (based on ElectionDataset usage)
DATE_COL = "election_date"
DISTRICT_COL = "Circulo" # Changed to Circulo
PARTY_COL = "party_id" # Expecting long format output
VOTES_COL = "votes"
# --- End Configuration ---

def calculate_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates vote percentages and deviations from national average."""

    # Ensure date column is datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # --- Ensure VOTES_COL is numeric --- 
    df[VOTES_COL] = pd.to_numeric(df[VOTES_COL], errors='coerce')
    original_rows = len(df)
    df.dropna(subset=[VOTES_COL], inplace=True) # Drop rows where votes couldn't be converted
    if len(df) < original_rows:
        print(f"Warning: Dropped {original_rows - len(df)} rows due to non-numeric vote counts.")
    # --- End Ensure Numeric --- 
    
    # Filter out zero total votes before calculating percentage to avoid NaNs/Infs
    group_totals = df.groupby([DATE_COL, DISTRICT_COL])[VOTES_COL].sum()
    valid_groups = group_totals[group_totals > 0].reset_index()[[DATE_COL, DISTRICT_COL]]
    if len(valid_groups) < len(group_totals):
        print(f"Warning: Excluding {len(group_totals) - len(valid_groups)} district-election pairs with zero total votes.")
        df = df.merge(valid_groups, on=[DATE_COL, DISTRICT_COL], how='inner')
        
    if df.empty:
        print("Error: DataFrame is empty after filtering zero-vote districts. Cannot proceed.")
        return pd.DataFrame() # Return empty df
        
    # 1. Calculate total votes per district per election
    df['total_district_votes'] = df.groupby([DATE_COL, DISTRICT_COL])[VOTES_COL].transform('sum')

    # 2. Calculate vote percentage for each party in each district per election
    df['vote_percentage'] = df[VOTES_COL] / df['total_district_votes'] * 100

    # 3. Calculate national average vote percentage for each party per election
    total_national_votes_party = df.groupby([DATE_COL, PARTY_COL])[VOTES_COL].sum()
    total_national_votes_election = df.groupby(DATE_COL)[VOTES_COL].sum()
    national_avg_perc = (total_national_votes_party / total_national_votes_election.reindex(total_national_votes_party.index, level=0) * 100)
    national_avg_perc = national_avg_perc.rename('national_avg_percentage')

    # 4. Merge national average back and calculate deviation
    df = df.merge(national_avg_perc, on=[DATE_COL, PARTY_COL], how='left')
    df['deviation'] = df['vote_percentage'] - df['national_avg_percentage']

    # Handle cases where a party might not exist nationally in one election?
    df['deviation'] = df['deviation'].fillna(0) # Assume 0 deviation if national avg is missing
    # Also fillna for national_avg_percentage itself if merge failed
    df['national_avg_percentage'] = df['national_avg_percentage'].fillna(0)

    print("\nCalculated Vote Percentages and Deviations.")
    print(df[[DATE_COL, DISTRICT_COL, PARTY_COL, 'vote_percentage', 'national_avg_percentage', 'deviation']].head())

    return df

def analyze_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes the stability (std dev) of deviations over time."""
    
    if df.empty:
        print("Skipping stability analysis: input DataFrame is empty.")
        return pd.DataFrame()

    # Ensure we have enough data points per group to calculate std dev meaningfully
    min_elections_threshold = 2
    deviation_stats = df.groupby([DISTRICT_COL, PARTY_COL])['deviation'].agg(
        deviation_std='std',
        deviation_mean='mean',
        election_count='size'
    ).reset_index()

    deviation_stats = deviation_stats[deviation_stats['election_count'] >= min_elections_threshold]

    stability_results = deviation_stats.sort_values(by='deviation_std', ascending=False)

    print(f"\nAnalyzed stability across {stability_results.shape[0]} District-Party combinations (with >={min_elections_threshold} elections).")
    return stability_results

# --- NEW FUNCTION: Analyze Sensitivity Fit --- 
def analyze_sensitivity_fit(df: pd.DataFrame, min_elections_threshold: int = 3) -> pd.DataFrame:
    """Analyzes how well national avg predicts district vote % via linear regression (Absolute Levels)."""
    if not STATSMODELS_AVAILABLE:
        print("Skipping sensitivity fit analysis: statsmodels is not installed.")
        return pd.DataFrame()
        
    if df.empty:
        print("Skipping sensitivity fit analysis: input DataFrame is empty.")
        return pd.DataFrame()
        
    results = []
    grouped = df.groupby([DISTRICT_COL, PARTY_COL])

    print(f"\nAnalyzing sensitivity fit (Absolute Levels) for {len(grouped)} district-party groups...")
    
    for name, group in grouped:
        # Need enough data points for regression
        if len(group) < min_elections_threshold:
            continue
        
        # Use original columns directly
        group_clean = group.copy()
        # Rename for formula compatibility if needed (e.g., if names have spaces/special chars)
        # In this case, names seem okay, but renaming is safer practice.
        group_clean.rename(columns={'vote_percentage': 'vote_perc', 'national_avg_percentage': 'national_avg_perc'}, inplace=True)

        try:
            # Fit OLS model: district_vote_percentage ~ national_avg_percentage
            model = smf.ols('vote_perc ~ national_avg_perc', data=group_clean).fit()
            
            results.append({
                DISTRICT_COL: name[0],
                PARTY_COL: name[1],
                'r_squared': model.rsquared, # R-squared of the absolute level model
                'intercept': model.params['Intercept'], 
                'beta': model.params['national_avg_perc'], # Slope for absolute levels
                'beta_p_value': model.pvalues['national_avg_perc'],
                'election_count': len(group)
            })
        except Exception as e:
            print(f"Warning: Could not fit absolute level model for group {name}: {e}")

    if not results:
        print("No sensitivity analysis (absolute levels) results generated.")
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    # Sort by R-squared ascending (worst fits first)
    results_df = results_df.sort_values(by='r_squared', ascending=True)
    
    print(f"Analyzed sensitivity fit (absolute levels) across {results_df.shape[0]} District-Party combinations (with >={min_elections_threshold} elections).")
    return results_df
# --- End NEW FUNCTION --- 

def main():
    print("\n--- Starting District Deviation Stability Analysis ---")

    # 1. Load Data using the project's loader
    print("Loading election data using src.data.loaders.load_election_results...")
    try:
        election_dates_to_load = ElectionDataset.historical_election_dates
        parties_to_load = ElectionDataset.political_families
        election_df_wide = load_election_results(
            election_dates=election_dates_to_load,
            political_families=parties_to_load,
            aggregate_national=False
        )

        if election_df_wide is None or election_df_wide.empty:
            print("Failed to load election data or returned empty DataFrame. Exiting.")
            return

        print(f"Loaded district-level data shape (wide): {election_df_wide.shape}")
        print(f"Wide format columns: {election_df_wide.columns.tolist()}")
        actual_party_cols = [p for p in parties_to_load if p in election_df_wide.columns]
        if not actual_party_cols:
            print("Error: No party columns found in the loaded data to melt.")
            return
            
        print("Melting DataFrame to long format...")
        id_cols = [DATE_COL, DISTRICT_COL]
        for col in ['date', 'pollster', 'sample_size']:
             if col in election_df_wide.columns:
                  id_cols.append(col)
        election_df = pd.melt(
            election_df_wide,
            id_vars=id_cols,
            value_vars=actual_party_cols,
            var_name=PARTY_COL,
            value_name=VOTES_COL
        )
        print(f"Melted data shape (long): {election_df.shape}")
        print(f"Long format columns: {election_df.columns.tolist()}")

        if not all(col in election_df.columns for col in [DATE_COL, DISTRICT_COL, PARTY_COL, VOTES_COL]):
             print(f"Error: Melted DataFrame missing expected columns. Expected: {DATE_COL}, {DISTRICT_COL}, {PARTY_COL}, {VOTES_COL}. Got: {election_df.columns.tolist()}")
             return

    except ImportError as e:
        print(f"ImportError: Failed to import from src.data.loaders or src.data.dataset.")
        print(f"Ensure you are running this script from the project root directory, e.g., using 'python scripts/analyze_district_deviation_stability.py'")
        print(f"Or ensure the project root is in your PYTHONPATH. Error details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading or melting: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Calculate Deviations (also calculates vote_percentage and national_avg_percentage needed for sensitivity analysis)
    deviation_df = calculate_deviations(election_df)
    
    if deviation_df.empty:
         print("Aborting after calculate_deviations returned empty DataFrame.")
         return

    # 3. Analyze Stability (Original Analysis)
    stability_results = analyze_stability(deviation_df)
    if not stability_results.empty:
        print(f"\n--- Top {TOP_N} Most Variable District-Party Deviations (by Standard Deviation) ---")
        print(stability_results.head(TOP_N).to_string())

    # 4. Analyze Sensitivity Fit (Absolute Level Analysis)
    sensitivity_fit_results = analyze_sensitivity_fit(deviation_df)
    if not sensitivity_fit_results.empty:
        print(f"\n--- Top {TOP_N} Poorest Fits for Absolute Level Sensitivity Model (by R-squared) ---")
        # Select and format columns for printing
        cols_to_show = [DISTRICT_COL, PARTY_COL, 'r_squared', 'intercept', 'beta', 'beta_p_value', 'election_count']
        print(sensitivity_fit_results[cols_to_show].head(TOP_N).to_string(float_format="%.3f"))

    # Optional: Save full results
    # output_stability_file = "district_deviation_stability.csv"
    # stability_results.to_csv(output_stability_file, index=False)
    # print(f"\nFull stability results saved to {output_stability_file}")
    # if not sensitivity_fit_results.empty:
    #     output_sensitivity_file = "district_sensitivity_fit.csv"
    #     sensitivity_fit_results.to_csv(output_sensitivity_file, index=False)
    #     print(f"Full sensitivity fit results saved to {output_sensitivity_file}")

if __name__ == "__main__":
    main() 