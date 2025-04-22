import os
import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Adjust path to import from src ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (scripts directory)
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the system path (this assumes your workspace root is the parent of 'scripts')
sys.path.insert(0, parent_dir)
# --- End path adjustment ---

try:
    from src.data.dataset import ElectionDataset
    # Check if statsmodels is available
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import ElectionDataset from src.data.dataset.")
    print("Ensure the script is run from the workspace root or the path is correctly set.")
    # Fallback or exit if necessary
    ElectionDataset = None 
    STATSMODELS_AVAILABLE = False # Assume statsmodels might also fail if src path is wrong

# --- Constants ---
# Define column names consistently (adapt if different in your actual data)
DISTRICT_COL = 'Circulo' # Column for district name
PARTY_COL = 'Partido' # Column for party name
DATE_COL = 'election_date' # Column for election date
VOTE_PERC_COL = 'vote_percentage' # Column for vote percentage (used as target in regression)
NATIONAL_VOTE_PERC_COL = 'national_avg_percentage' # Column for national vote percentage (used as predictor)
MIN_ELECTIONS_THRESHOLD = 3 # Minimum number of elections needed for regression per district-party

# --- Main Analysis Function ---
def analyze_district_deviation(dataset: ElectionDataset, output_dir: str = "outputs/analysis"):
    """
    Analyzes the relationship between national and district vote percentages.

    Fits a linear regression model for each district-party combination:
        district_vote_percentage ~ national_vote_percentage

    Args:
        dataset: An instantiated ElectionDataset object containing historical results.
        output_dir: Directory to save analysis results (plots and CSV).
    """
    if not STATSMODELS_AVAILABLE:
        print("Error: statsmodels package is required for this analysis. Please install it.")
        return

    if dataset is None:
        print("Error: ElectionDataset could not be loaded.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving analysis results to: {output_dir}")

    # Get required dataframes
    try:
        district_results = dataset.results_mult_district.copy()
        national_results = dataset.results_oos.copy()
    except AttributeError as e:
        print(f"Error accessing dataframes from dataset object: {e}")
        print("Ensure the dataset was loaded correctly and contains 'results_mult_district' and 'results_oos'.")
        return

    if district_results.empty or national_results.empty:
        print("Error: District or national results data is empty.")
        return

    print(f"District results shape: {district_results.shape}")
    print(f"National results shape: {national_results.shape}")

    # --- Data Preprocessing ---
    # We expect raw vote counts for party columns and a 'sample_size' column 
    # calculated by the loader.
    
    # Get list of party columns present in both dataframes
    party_cols_dist = [p for p in dataset.political_families if p in district_results.columns]
    party_cols_nat = [p for p in dataset.political_families if p in national_results.columns]
    common_party_cols = list(set(party_cols_dist) & set(party_cols_nat))
    if not common_party_cols:
        print("Error: No common party columns found between district and national results.")
        return
    print(f"Using common party columns: {common_party_cols}")

    # Ensure sample_size columns exist
    if 'sample_size' not in district_results.columns:
        print("Error: 'sample_size' column missing from district results.")
        return
    if 'sample_size' not in national_results.columns:
        print("Error: 'sample_size' column missing from national results.")
        return
        
    # 1. Calculate percentages FROM counts
    print("Calculating vote percentages from counts...")
    district_perc_dfs = []
    for party in common_party_cols:
        # Calculate percentage, handle division by zero
        perc = district_results[party].astype(float) / district_results['sample_size'].astype(float)
        # Fill NaN/inf resulting from division by zero with 0
        perc.replace([np.inf, -np.inf], np.nan, inplace=True)
        perc.fillna(0, inplace=True)
        df_party = district_results[[DATE_COL, DISTRICT_COL]].copy()
        df_party[PARTY_COL] = party
        df_party[VOTE_PERC_COL] = perc
        district_perc_dfs.append(df_party)
    district_data_long = pd.concat(district_perc_dfs, ignore_index=True)
    print(f"Created long district percentage data shape: {district_data_long.shape}")

    national_perc_dfs = []
    for party in common_party_cols:
        perc = national_results[party].astype(float) / national_results['sample_size'].astype(float)
        perc.replace([np.inf, -np.inf], np.nan, inplace=True)
        perc.fillna(0, inplace=True)
        df_party = national_results[[DATE_COL]].copy()
        df_party[PARTY_COL] = party
        df_party[NATIONAL_VOTE_PERC_COL] = perc
        national_perc_dfs.append(df_party)
    national_data_long = pd.concat(national_perc_dfs, ignore_index=True)
    print(f"Created long national percentage data shape: {national_data_long.shape}")

    # 2. Ensure date columns are datetime objects (already done in loader, but check)
    try:
        district_data_long[DATE_COL] = pd.to_datetime(district_data_long[DATE_COL])
        national_data_long[DATE_COL] = pd.to_datetime(national_data_long[DATE_COL])
    except KeyError as e:
        print(f"Error: Date column '{DATE_COL}' not found after creating long format: {e}")
        return

    # 3. Prepare for merge (Data is already in long format)
    # district_data = district_results[[DATE_COL, DISTRICT_COL, PARTY_COL, VOTE_PERC_COL]].copy()
    # national_data = national_results[national_cols_to_select].copy()

    # 4. Merge district and national percentage data (now in long format)
    print("Merging long district and national percentage results...")
    merged_data = pd.merge(
        district_data_long,
        national_data_long,
        on=[DATE_COL, PARTY_COL],
        how='left' # Keep all district results, match national where possible
    )

    # Check for merge issues (should be less likely now)
    if merged_data[NATIONAL_VOTE_PERC_COL].isnull().any():
        print("Warning: Some district results could not be matched with national results.")
        # Optionally, print or analyze the unmatched rows
        # print(merged_data[merged_data[NATIONAL_VOTE_PERC_COL].isnull()])
        merged_data.dropna(subset=[NATIONAL_VOTE_PERC_COL], inplace=True) # Drop rows without national avg

    print(f"Merged data shape after dropping NaNs: {merged_data.shape}")
    if merged_data.empty:
        print("Error: No data remaining after merging and dropping NaNs.")
        return

    # --- Perform Regression Analysis ---
    regression_results = []
    grouped = merged_data.groupby([DISTRICT_COL, PARTY_COL])
    num_groups = len(grouped)
    print(f"Analyzing {num_groups} district-party combinations...")
    
    processed_groups = 0
    for name, group in grouped:
        district_name, party_name = name
        
        # Ensure sufficient data points for regression
        if len(group) < MIN_ELECTIONS_THRESHOLD:
            continue
            
        # Clean data for this group (e.g., remove NaNs just in case)
        group_clean = group[[VOTE_PERC_COL, NATIONAL_VOTE_PERC_COL]].dropna()
        
        if len(group_clean) < MIN_ELECTIONS_THRESHOLD:
            continue # Skip if not enough valid points after potential dropna

        try:
            # Define and fit the OLS model
            # Use rename for formula compatibility if columns have spaces/special chars
            group_clean.rename(columns={VOTE_PERC_COL: 'district_perc', 
                                          NATIONAL_VOTE_PERC_COL: 'national_perc'}, inplace=True)
            
            model = smf.ols('district_perc ~ national_perc', data=group_clean).fit()

            regression_results.append({
                DISTRICT_COL: district_name,
                PARTY_COL: party_name,
                'intercept': model.params['Intercept'],
                'slope': model.params['national_perc'], # Sensitivity
                'r_squared': model.rsquared,
                'p_value_slope': model.pvalues['national_perc'],
                'n_elections': len(group_clean)
            })
        except Exception as e:
            print(f"Warning: Could not fit model for group {name}: {e}")

        processed_groups += 1
        if processed_groups % 100 == 0:
             print(f"  Processed {processed_groups}/{num_groups} groups...")

    print(f"Finished regression for {len(regression_results)} groups with sufficient data.")
    
    if not regression_results:
        print("Error: No regression results were generated. Check data and thresholds.")
        return

    results_df = pd.DataFrame(regression_results)

    # Save regression results
    results_df.to_csv(os.path.join(output_dir, "district_deviation_regression.csv"), index=False)
    print("Regression results saved to CSV.")

    # --- Visualize Results ---
    print("Generating summary visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style

    # 1. Histogram of Slopes (Sensitivity)
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['slope'], kde=True, bins=30)
    plt.axvline(1.0, color='r', linestyle='--', label='Slope = 1 (Matches National Swing)')
    plt.title('Distribution of Slopes (District Sensitivity vs. National)')
    plt.xlabel('Slope (Sensitivity)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histogram_slopes.png"))
    plt.close()

    # 2. Histogram of Intercepts
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['intercept'], kde=True, bins=30)
    plt.axvline(0.0, color='r', linestyle='--', label='Intercept = 0')
    plt.title('Distribution of Intercepts (Baseline District Offset)')
    plt.xlabel('Intercept (Offset in % points)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histogram_intercepts.png"))
    plt.close()

    # 3. Histogram of R-squared values
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['r_squared'], kde=True, bins=20)
    plt.title('Distribution of R-squared Values (Goodness of Fit)')
    plt.xlabel('R-squared')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histogram_r_squared.png"))
    plt.close()

    # 4. Scatter plot of Intercept vs. Slope (Optional, can be busy)
    plt.figure(figsize=(10, 8))
    # Filter out extreme outliers for better visualization if needed
    filtered_df = results_df[(results_df['slope'].between(-2, 4)) & (results_df['intercept'].between(-0.5, 0.5))]
    sns.scatterplot(data=filtered_df, x='intercept', y='slope', hue='r_squared', size='n_elections', palette='viridis', alpha=0.7)
    plt.axhline(1.0, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(0.0, color='grey', linestyle='--', alpha=0.5)
    plt.title('Intercept vs. Slope for District-Party Combinations')
    plt.xlabel('Intercept (Baseline Offset)')
    plt.ylabel('Slope (Sensitivity)')
    plt.legend(title='R-squared / N Elections', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.savefig(os.path.join(output_dir, "scatter_intercept_slope.png"))
    plt.close()

    print("Visualizations saved.")
    print("Analysis complete.")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Running District Deviation Analysis Script...")

    # --- Load Dataset ---
    # Use a recent or upcoming election date for context when loading dataset
    # This date primarily affects which polls might be included if the dataset
    # class filters them, but for historical results, it mainly provides context.
    # You might want to make this configurable if needed.
    context_election_date = '2026-01-01' # Example future date
    
    # Default timescales (might not be strictly necessary for just results analysis)
    # but potentially required by the Dataset constructor
    default_baseline_ts = [90, 180, 365] 
    default_election_ts = [14, 30, 60]

    dataset_instance = None
    if ElectionDataset:
        try:
            print(f"Instantiating ElectionDataset with context date: {context_election_date}")
            dataset_instance = ElectionDataset(
                election_date=context_election_date,
                baseline_timescales=default_baseline_ts,
                election_timescales=default_election_ts,
                # test_cutoff=None # Load all historical data
            )
            print("ElectionDataset loaded successfully.")
        except Exception as e:
            print(f"Error loading ElectionDataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Cannot proceed without ElectionDataset class.")
        
    # --- Run Analysis ---
    if dataset_instance:
        # Define where to save the output
        analysis_output_dir = os.path.join("outputs", "district_analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        analyze_district_deviation(dataset_instance, output_dir=analysis_output_dir)
    else:
        print("Skipping analysis because the dataset could not be loaded.") 