import pandas as pd
from typing import Dict, List

# Assuming load_election_results is accessible, either imported 
# directly or via a helper module.
# For now, let's add a placeholder import path
# from ..data.loaders import load_election_results
# We will need to adjust this import based on the final project structure.

# Placeholder for actual import - replace with correct path
from src.data.loaders import load_election_results


def forecast_district_votes_uns(
    national_forecast_shares: Dict[str, float], 
    last_election_date: str, 
    political_families: List[str],
    election_dates: List[str] # Need this to pass to load_election_results
) -> pd.DataFrame:
    """
    Forecasts district-level vote counts using a Uniform National Swing (UNS).

    Calculates the national swing between the last election and the forecast,
    applies it uniformly to the last election's district results, and converts
    back to estimated vote counts.

    Args:
        national_forecast_shares (Dict[str, float]): Forecasted national vote shares 
                                                     (e.g., {'PS': 0.35, 'AD': 0.30}).
        last_election_date (str): The date of the last election (YYYY-MM-DD) to use as baseline.
        political_families (List[str]): List of relevant party names.
        election_dates (List[str]): Full list of election dates required by load_election_results.

    Returns:
        pd.DataFrame: DataFrame with columns ['Circulo'] + political_families,
                      containing the forecasted integer vote counts for each party 
                      in each district. Returns empty DataFrame on error.
    """
    print(f"\n--- Forecasting District Votes using UNS based on {last_election_date} ---")
    
    # --- 1. Load Data --- 
    try:
        # Load national results for the last election
        national_results_df = load_election_results(
            election_dates=election_dates, 
            political_families=political_families, 
            aggregate_national=True
        )
        last_election_national = national_results_df[national_results_df['date'] == pd.to_datetime(last_election_date)]
        if last_election_national.empty:
            print(f"Error: Could not find national results for {last_election_date}")
            return pd.DataFrame()
        # Should only be one row, take the first
        last_election_national = last_election_national.iloc[0]
        
        # Load district results for the last election
        district_results_df = load_election_results(
            election_dates=election_dates, 
            political_families=political_families, 
            aggregate_national=False
        )
        last_election_district = district_results_df[district_results_df['date'] == pd.to_datetime(last_election_date)]
        if last_election_district.empty:
            print(f"Error: Could not find district results for {last_election_date}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading election results: {e}")
        return pd.DataFrame()

    # --- 2. Calculate National Shares & Swing --- 
    # Ensure all parties from forecast are in political_families for consistency
    parties_in_forecast = list(national_forecast_shares.keys())
    all_parties = sorted(list(set(political_families + parties_in_forecast)))
    
    # Calculate national shares from last election results (counts)
    national_total_votes = last_election_national[political_families].sum()
    if national_total_votes == 0:
        print("Error: National total votes for last election are zero. Cannot calculate shares.")
        return pd.DataFrame()
        
    last_election_national_shares = last_election_national[political_families].astype(float) / national_total_votes
    # Ensure forecast dict has entries for all parties (default to 0 if missing)
    forecast_shares_series = pd.Series({p: national_forecast_shares.get(p, 0.0) for p in all_parties})
    last_election_national_shares = last_election_national_shares.reindex(all_parties, fill_value=0.0)

    # Calculate swing
    national_swing = forecast_shares_series - last_election_national_shares
    print("\nNational Swing Calculation:")
    print(pd.DataFrame({'Forecast': forecast_shares_series, 'Last Election': last_election_national_shares, 'Swing': national_swing}))

    # --- 3. Apply Swing to District Shares --- 
    # Calculate district shares from last election counts
    district_results_indexed = last_election_district.set_index('Circulo')
    # Use 'sample_size' which is the sum of votes for political_families per district
    district_total_votes = district_results_indexed['sample_size'].copy()
    last_election_district_shares = district_results_indexed[political_families].copy()

    # Avoid division by zero for districts with zero total votes in the baseline
    valid_districts = district_total_votes[district_total_votes > 0].index
    if len(valid_districts) < len(district_total_votes):
        invalid_districts = district_total_votes[district_total_votes <= 0].index.tolist()
        print(f"Warning: Districts with zero total votes in baseline election: {invalid_districts}. Shares cannot be calculated.")
        # Keep only valid districts for share calculation
        last_election_district_shares = last_election_district_shares.loc[valid_districts]
        district_total_votes = district_total_votes.loc[valid_districts]
        
    # Calculate shares only for valid districts
    last_election_district_shares = last_election_district_shares.astype(float).div(district_total_votes, axis=0)
    last_election_district_shares = last_election_district_shares.reindex(columns=all_parties, fill_value=0.0)
    
    # Apply swing (broadcasting the national swing series)
    forecasted_district_shares = last_election_district_shares.add(national_swing, axis=1)

    # --- 4. Adjust and Re-normalize --- 
    # Clip negative shares to zero
    forecasted_district_shares = forecasted_district_shares.clip(lower=0)
    
    # Re-normalize shares within each district to sum to 1
    row_sums = forecasted_district_shares.sum(axis=1)
    # Avoid division by zero if a district's shares all became zero after clipping
    valid_rows = row_sums[row_sums > 0].index
    if len(valid_rows) < len(forecasted_district_shares):
        zero_sum_districts = row_sums[row_sums <= 0].index.tolist()
        print(f"Warning: Districts with zero total shares after swing/clip: {zero_sum_districts}. Cannot normalize.")
        # Handle these districts - maybe assign equal shares or keep as zero?
        # For now, we'll normalize only valid rows. Zero-sum rows will result in zero votes.
        forecasted_district_shares.loc[zero_sum_districts, :] = 0.0 # Explicitly set to zero

    if not valid_rows.empty:
        forecasted_district_shares.loc[valid_rows] = forecasted_district_shares.loc[valid_rows].div(row_sums.loc[valid_rows], axis=0)

    # --- 5. Convert back to Counts --- 
    # Use the total votes from the *last election* in each district as the basis
    forecasted_district_counts = forecasted_district_shares.mul(district_total_votes, axis=0)
    
    # Round to nearest integer for vote counts
    forecasted_district_counts = forecasted_district_counts.round().astype(int)
    
    # Add back any districts that had zero votes initially, giving them zero forecast counts
    all_district_names = last_election_district['Circulo'].unique()
    forecasted_district_counts = forecasted_district_counts.reindex(all_district_names, fill_value=0)
    
    # Select only the original political_families columns for the final output
    final_forecast_counts = forecasted_district_counts.reset_index()[['Circulo'] + political_families]

    print("\nForecasted District Vote Counts (Head):")
    print(final_forecast_counts.head())
    print(f"Forecast generated for {len(final_forecast_counts)} districts.")

    return final_forecast_counts 