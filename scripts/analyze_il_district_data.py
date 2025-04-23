import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
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

def analyze_il_district_data(election_date_context: str, output_dir: str):
    """
    Analyzes the historical district-level election results for the 'IL' party
    to investigate potential causes for convergence issues related to district effects.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving analysis plots and data to: {output_dir}")

    print("\nLoading dataset...")
    try:
        # Instantiate dataset - use dummy timescales if needed, as we only care about results
        dataset = ElectionDataset(
            election_date=election_date_context,
            baseline_timescales=[365], # Dummy value
            election_timescales=[30]    # Dummy value
        )
    except Exception as e:
        print(f"Error initializing ElectionDataset: {e}")
        return

    # Access district results
    if not hasattr(dataset, 'results_mult_district') or dataset.results_mult_district is None or dataset.results_mult_district.empty:
        print("Error: Historical district results (results_mult_district) not found or empty in the dataset.")
        return

    df_results = dataset.results_mult_district.copy()
    print(f"Loaded district results with shape: {df_results.shape}")
    print(f"Columns available: {df_results.columns.tolist()}")

    # --- Define required columns ---
    party_col = 'L'
    district_col = 'Circulo'
    date_col = 'election_date'
    sample_size_col = 'sample_size' # This represents total votes in the district

    # --- Check if required columns exist ---
    required_cols = [district_col, date_col, sample_size_col]
    if party_col not in df_results.columns:
        print(f"Error: Party column '{party_col}' not found in district results.")
        return
    if not all(col in df_results.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_results.columns]
        print(f"Error: Missing required columns in district results: {missing}")
        return

    print(f"Analyzing results for party: '{party_col}'")

    # --- Calculate Vote Share ---
    # Ensure sample size is not zero to avoid division errors
    df_results = df_results[df_results[sample_size_col] > 0].copy()
    if df_results.empty:
        print("Error: No district results found with non-zero sample size.")
        return

    df_results[f'{party_col}_share'] = df_results[party_col] / df_results[sample_size_col]
    share_col = f'{party_col}_share'

    # Convert election_date to string for easier grouping/plotting labels
    df_results[date_col] = pd.to_datetime(df_results[date_col]).dt.strftime('%Y-%m-%d')

    # --- Analysis ---

    # 1. Variance of L Share *across districts* for each election
    print(f"\n--- Variance of {party_col} Share Across Districts (per Election) ---")
    variance_per_election = df_results.groupby(date_col)[share_col].var().sort_index()
    print(variance_per_election)
    variance_per_election_path = os.path.join(output_dir, f'{party_col}_share_variance_across_districts.csv')
    variance_per_election.to_csv(variance_per_election_path)
    print(f"Saved variance across districts data to {variance_per_election_path}")

    plt.figure(figsize=(10, 6))
    variance_per_election.plot(kind='bar', title=f'Variance of {party_col} Share Across Districts per Election')
    plt.ylabel('Vote Share Variance')
    plt.xlabel('Election Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{party_col}_share_variance_across_districts.png'))
    plt.close()

    # 2. Variance of L Share *within each district* across elections
    print(f"\n--- Variance of {party_col} Share Within Districts (across Elections) ---")
    variance_within_district = df_results.groupby(district_col)[share_col].var().sort_values(ascending=False)
    print(variance_within_district)
    variance_within_district_path = os.path.join(output_dir, f'{party_col}_share_variance_within_districts.csv')
    variance_within_district.to_csv(variance_within_district_path)
    print(f"Saved variance within districts data to {variance_within_district_path}")

    plt.figure(figsize=(12, max(6, len(variance_within_district) * 0.3))) # Adjust height based on # districts
    variance_within_district.plot(kind='barh', title=f'Variance of {party_col} Share Within Each District (Across Elections)')
    plt.xlabel('Vote Share Variance')
    plt.ylabel('District')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{party_col}_share_variance_within_districts.png'))
    plt.close()

    # 3. Boxplot of L Share per Election
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_results, x=date_col, y=share_col, order=sorted(df_results[date_col].unique()))
    plt.title(f'Distribution of {party_col} Vote Share Across Districts per Election')
    plt.ylabel('Vote Share')
    plt.xlabel('Election Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{party_col}_share_boxplot_per_election.png'))
    plt.close()

    # 4. Boxplot of L Share per District
    plt.figure(figsize=(12, max(6, len(df_results[district_col].unique()) * 0.3))) # Adjust height
    # Sort districts by median share for better visualization
    median_order = df_results.groupby(district_col)[share_col].median().sort_values().index
    sns.boxplot(data=df_results, x=share_col, y=district_col, order=median_order)
    plt.title(f'Distribution of {party_col} Vote Share Across Elections per District')
    plt.xlabel('Vote Share')
    plt.ylabel('District')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{party_col}_share_boxplot_per_district.png'))
    plt.close()

    # 5. Check for very low sample sizes (total votes) in districts where L contested
    party_results = df_results[df_results[party_col] > 0] # Changed variable name
    low_sample_threshold = 1000 # Example threshold for low total votes
    low_sample_districts = party_results[party_results[sample_size_col] < low_sample_threshold] # Use party_results
    if not low_sample_districts.empty:
        print(f"\n--- Districts with Low Total Votes (<{low_sample_threshold}) where {party_col} received votes ---")
        print(low_sample_districts[[date_col, district_col, sample_size_col, party_col, share_col]])
        low_sample_districts.to_csv(os.path.join(output_dir, f'{party_col}_low_total_votes_districts.csv'), index=False) # Uses party_col


    print("\nAnalysis script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze IL District-Level Election Results Data")
    parser.add_argument("--election-date", required=True, help="Target election date (YYYY-MM-DD) used for dataset context")
    parser.add_argument("--output-dir", default="outputs/l_district_data_analysis", help="Directory to save analysis results")
    args = parser.parse_args()

    analyze_il_district_data(args.election_date, args.output_dir) 