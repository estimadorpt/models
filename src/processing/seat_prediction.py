import pandas as pd
from pprint import pprint
from typing import Dict, List, Optional, Tuple
import os # Needed for join path if loading config inside
import json # Needed for saving results
import time
import traceback
import numpy as np
import arviz as az
import xarray as xr

# Assuming the project structure allows these imports
# Adjust paths if necessary based on final structure
try:
    from src.data.loaders import load_district_config, load_election_results
    from src.processing.forecasting import forecast_district_votes_uns
    from src.processing.electoral_systems import calculate_dhondt
    # If DATA_DIR is needed for loaders, import it
    from src.config import DATA_DIR
    from src.data.dataset import ElectionDataset # Needed for type hinting and accessing dataset properties
    from src.visualization.plots import plot_seat_distribution_histograms # Needed if plotting is moved here
except ImportError as e:
    print(f"Error importing dependencies in seat_prediction.py: {e}")
    # Re-raise or handle appropriately depending on desired behavior
    raise

# Define type hints for clarity
XrDataArray = 'xr.DataArray' # Use string forward reference if xarray isn't imported directly
XarrayDataset = 'xr.Dataset' # Use string forward reference for the posterior

def calculate_seat_predictions(
    national_forecast_shares: Dict[str, float],
    last_election_date: str,
    political_families: List[str],
    election_dates: List[str]
) -> Dict[str, int]:
    """
    Predicts the national seat allocation based on a national vote forecast.

    Uses Uniform National Swing (UNS) to estimate district votes and then
    applies the D'Hondt method per district.
    This version is optimized for being called multiple times (e.g., in a simulation loop).

    Args:
        national_forecast_shares (Dict[str, float]): The forecasted national vote shares (MUST be normalized).
        last_election_date (str): The date of the last election (YYYY-MM-DD) to use as baseline for swing.
        political_families (List[str]): List of relevant party names.
        election_dates (List[str]): Full list of historical election dates required by data loaders.

    Returns:
        Dict[str, int]: A dictionary mapping party names to their predicted total seats.
                        Returns None if critical errors occur (e.g., data loading failure).
    """
    # print("--- Calculating Seat Predictions --- ") # Too verbose for loop

    # 1. Load District Configuration (Seats per District)
    # Consider caching this outside the loop in the caller if performance is critical
    district_config = load_district_config()
    if not district_config:
        print("Error: Failed to load district configuration. Aborting seat prediction run.")
        return None # Return None to indicate failure
    # print(f"Loaded configuration for {len(district_config)} districts.") # Verbose

    # 2. Forecast District Votes using UNS
    # print("Forecasting district votes using UNS...") # Verbose
    # Pass already normalized shares
    forecasted_district_counts = forecast_district_votes_uns(
        national_forecast_shares=national_forecast_shares,
        last_election_date=last_election_date,
        political_families=political_families,
        election_dates=election_dates
    )

    if forecasted_district_counts is None or forecasted_district_counts.empty:
        print("Error: Failed to forecast district votes.") # Keep error message
        return None # Return None to indicate failure

    # 3. Allocate Seats per District using D'Hondt
    total_seats_allocation = {party: 0 for party in political_families}
    # district_allocations = {} # Not needed if only returning total

    # print("\nAllocating Seats per District (D'Hondt)...") # Verbose
    if 'Circulo' not in forecasted_district_counts.columns:
         print("Error: Forecasted district counts DataFrame is missing 'Circulo' column.")
         return None
         
    num_districts_processed = 0
    for _, district_row in forecasted_district_counts.iterrows():
        try:
            circulo_name = district_row['Circulo']
            if circulo_name not in district_config:
                # print(f"Warning: District '{circulo_name}' found in forecast but not in seat config. Skipping.") # Verbose
                continue

            num_seats = district_config[circulo_name]
            votes_dict = district_row[political_families].astype(int).to_dict()

            district_seat_allocation = calculate_dhondt(votes_dict, num_seats)
            # district_allocations[circulo_name] = district_seat_allocation # Not needed

            for party, seats in district_seat_allocation.items():
                if party in total_seats_allocation:
                    total_seats_allocation[party] += seats
            num_districts_processed += 1
        except Exception as e:
             print(f"Error processing row for district '{district_row.get('Circulo', 'Unknown')}': {e}")
             continue

    # print(f"Seat allocation completed for {num_districts_processed} districts.") # Verbose

    # 4. Final Aggregated Results & Verification (Optional for caller)
    expected_total = sum(district_config.values())
    total_predicted = sum(total_seats_allocation.values())
    if total_predicted != expected_total:
        # Less alarming warning, as small differences can occur
        # print(f"Note: Total predicted seats ({total_predicted}) differ slightly from expected ({expected_total}).")
        pass # Caller can perform this check on the summary if needed

    # Return the raw allocation, let caller remove zeros if desired
    # final_allocation = {party: seats for party, seats in total_seats_allocation.items() if seats > 0}
    return total_seats_allocation 

def simulate_seat_allocation(
    target_pop_posterior: xr.DataArray, # Changed hint to reflect it's likely Dataset
    dataset: ElectionDataset,
    num_samples_for_seats: int,
    pred_dir: str,
    prediction_date_mode: str,
    hdi_prob: float = 0.94, # Add hdi_prob as parameter
    debug: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Simulates seat allocation based on predicted national vote shares and historical district results.

    Args:
        target_pop_posterior: Xarray Dataset containing posterior samples of latent popularity (vote shares).
                               Must have dimensions 'chain', 'draw', and a party dimension coordinate.
        dataset: The ElectionDataset object containing political_families, election_dates, etc.
        num_samples_for_seats: The maximum number of posterior samples to use for the simulation.
        pred_dir: Directory to save prediction outputs (like CSVs).
        prediction_date_mode: String indicating the prediction mode ('election_day', etc.) for filenames.
        hdi_prob: The probability mass to include in the highest density interval for summary stats.
        debug: Boolean flag for enabling debug print statements.

    Returns:
        A tuple containing:
        - pd.DataFrame: DataFrame with seat allocation samples (or None if simulation fails).
        - pd.DataFrame: DataFrame with seat allocation summary statistics (or None if simulation fails).
    """
    print(f"\nStarting seat prediction simulation using up to {num_samples_for_seats} samples...")
    # Initialize return values
    seats_df_results: Optional[pd.DataFrame] = None
    seat_summary_results: Optional[pd.DataFrame] = None

    # Extract necessary info from dataset
    political_families = dataset.political_families
    election_dates = dataset.election_dates
    last_election_date_for_swing = None

    # --- Pre-load Data and Calculate Baselines --- 
    pre_load_successful = False # Initialize flag
    last_election_national_shares = None
    last_election_district_shares = None
    district_config = None
    district_total_votes_baseline = None
    all_district_names = None

    try:
        if debug: print("Pre-loading data for seat simulation...")
        # 1. Find latest election date string
        if election_dates:
            parsed_dates = [pd.to_datetime(d, errors='coerce') for d in election_dates]
            valid_dates = [d for d in parsed_dates if pd.notna(d)]
            if valid_dates:
                    last_election_date_dt = max(valid_dates)
                    original_indices = [i for i, dt in enumerate(parsed_dates) if dt == last_election_date_dt]
                    if original_indices:
                        original_index = original_indices[0]
                        # Handle cases where dates might be stored as strings or datetime objects
                        if isinstance(election_dates[original_index], str):
                            last_election_date_for_swing = election_dates[original_index]
                        else:
                            try: # Attempt conversion if not string
                                last_election_date_for_swing = pd.to_datetime(election_dates[original_index]).strftime('%Y-%m-%d')
                            except Exception:
                                 last_election_date_for_swing = last_election_date_dt.strftime('%Y-%m-%d') # Fallback
                    else:
                        last_election_date_for_swing = last_election_date_dt.strftime('%Y-%m-%d')
        
        if not last_election_date_for_swing:
                raise ValueError("Could not determine last election date for swing calculation.")
        if debug: print(f"(Using {last_election_date_for_swing} as baseline election for swing)")

        # 2. Load District Config
        district_config = load_district_config()
        if not district_config: raise ValueError("Failed to load district configuration.")
        all_district_names = list(district_config.keys()) # Get list of names
        if debug: print(f"Loaded config for {len(all_district_names)} districts.")

        # 3. Load National Results for Baseline Election
        national_results_df_all = load_election_results(election_dates, political_families, aggregate_national=True)
        last_election_national_row = national_results_df_all[national_results_df_all['date'] == pd.to_datetime(last_election_date_for_swing)]
        if last_election_national_row.empty: raise ValueError(f"Could not find national results for baseline date {last_election_date_for_swing}")
        last_election_national_row = last_election_national_row.iloc[0]
        # Ensure numeric conversion for safety
        national_total_votes_baseline = pd.to_numeric(last_election_national_row[political_families], errors='coerce').sum()
        if pd.isna(national_total_votes_baseline) or national_total_votes_baseline <= 0: 
            raise ValueError(f"Baseline national total votes are invalid ({national_total_votes_baseline}).")
        last_election_national_shares = pd.to_numeric(last_election_national_row[political_families], errors='coerce').fillna(0) / national_total_votes_baseline
        last_election_national_shares = last_election_national_shares.reindex(political_families, fill_value=0.0)
        if debug: print("Loaded baseline national shares.")

        # 4. Load District Results for Baseline Election
        district_results_df_all = load_election_results(election_dates, political_families, aggregate_national=False)
        last_election_district_df = district_results_df_all[district_results_df_all['date'] == pd.to_datetime(last_election_date_for_swing)].copy()
        if last_election_district_df.empty: raise ValueError(f"Could not find district results for baseline date {last_election_date_for_swing}")
        if 'Circulo' not in last_election_district_df.columns: raise ValueError("'Circulo' column missing in loaded district results.")
        # Ensure 'Circulo' is suitable as index (e.g., string)
        last_election_district_df['Circulo'] = last_election_district_df['Circulo'].astype(str)
        last_election_district_df = last_election_district_df.set_index('Circulo')
        # Ensure 'sample_size' is numeric
        district_total_votes_baseline = pd.to_numeric(last_election_district_df['sample_size'], errors='coerce').fillna(0)
        # Ensure party columns are numeric by applying to_numeric to each column
        party_cols_numeric = last_election_district_df[political_families].apply(pd.to_numeric, errors='coerce').fillna(0)
        last_election_district_shares = party_cols_numeric.copy()
        valid_districts_baseline = district_total_votes_baseline[district_total_votes_baseline > 0].index
        # Perform division only for valid districts
        if not valid_districts_baseline.empty:
             last_election_district_shares.loc[valid_districts_baseline] = last_election_district_shares.loc[valid_districts_baseline].div(district_total_votes_baseline.loc[valid_districts_baseline], axis=0)
        # Handle districts with zero votes (set shares to zero)
        zero_vote_districts = district_total_votes_baseline[district_total_votes_baseline <= 0].index
        if not zero_vote_districts.empty:
             last_election_district_shares.loc[zero_vote_districts, :] = 0.0
        # Reindex to ensure all parties and districts are present
        last_election_district_shares = last_election_district_shares.reindex(columns=political_families, fill_value=0.0)
        last_election_district_shares = last_election_district_shares.reindex(index=all_district_names, fill_value=0.0)
        district_total_votes_baseline = district_total_votes_baseline.reindex(all_district_names, fill_value=0)
        if debug: print("Loaded and processed baseline district shares and totals.")
        
        print("Pre-loading complete.")
        pre_load_successful = True
        
    except Exception as preload_err:
            print(f"Error pre-loading data for seat simulation: {preload_err}")
            if debug: traceback.print_exc()
            # pre_load_successful remains False
            # Return None tuple immediately if pre-load fails
            return None, None
    # --- End Pre-load Data --- 

    # Proceed only if pre-loading was successful
    if pre_load_successful:
        # --- Dynamically find party coordinate name --- 
        party_dim_name = None
        party_coord_values = None
        expected_num_parties = len(political_families)
        # Iterate through dimensions AND coordinates of the input DataArray/Dataset
        for dim_name in target_pop_posterior.dims:
            if dim_name in target_pop_posterior.coords:
                if target_pop_posterior[dim_name].size == expected_num_parties:
                    coord_values = target_pop_posterior[dim_name].values.tolist()
                    # Check if coordinate values match political families (order doesn't matter)
                    if sorted(coord_values) == sorted(political_families):
                        party_dim_name = dim_name
                        party_coord_values = coord_values # Store the actual coord values
                        if debug: print(f"Found party coordinate dimension: '{party_dim_name}' with values: {party_coord_values}")
                        break # Found it
        
        if not party_dim_name:
            print(f"Error: Could not determine party coordinate dimension in target_pop_posterior matching political_families: {political_families}. Aborting simulation.")
            # Check available coords/dims for debugging
            if debug:
                 print("Available dimensions: ", list(target_pop_posterior.dims))
                 print("Available coordinates: ", list(target_pop_posterior.coords))
            return None, None # Abort if party dimension not found
        
        # --- Start Simulation Loop --- 
        seat_allocation_samples = []
        try:
            # Stack chain and draw dimensions for easier iteration
            # Check if input is Dataset or DataArray
            if isinstance(target_pop_posterior, xr.Dataset):
                # If Dataset, assume first data var is the one (or add logic to find it)
                data_var_name = list(target_pop_posterior.data_vars)[0]
                if debug: print(f"Using data variable '{data_var_name}' from posterior Dataset.")
                target_data_array = target_pop_posterior[data_var_name]
            elif isinstance(target_pop_posterior, xr.DataArray):
                # If DataArray, use it directly
                if debug: print(f"Using input DataArray directly for posterior samples.")
                target_data_array = target_pop_posterior
            else:
                raise TypeError(f"Unsupported type for target_pop_posterior: {type(target_pop_posterior)}")

            stacked_posterior = target_data_array.stack(sample=("chain", "draw"))
            total_draws = len(stacked_posterior['sample'])
            samples_to_process = min(num_samples_for_seats, total_draws)
            print(f"Processing {samples_to_process} out of {total_draws} available posterior samples...")
            loop_start_time = time.time()

            for i in range(samples_to_process):
                if (i + 1) % 500 == 0: print(f"  Processed {i+1}/{samples_to_process} samples...")
                # Get the vote share sample for this iteration
                sample_shares_xr = stacked_posterior.isel(sample=i)
                
                # Convert xarray slice to dictionary {party_name: share}
                # Use the discovered party_coord_values to ensure correct mapping
                sample_shares_dict = {party: sample_shares_xr.sel(**{party_dim_name: party}).item() 
                                        for party in party_coord_values}

                # Normalize shares in the sample (robust against potential small negative values or sum != 1)
                total_share = sum(sample_shares_dict.values())
                if total_share > 1e-9: # Avoid division by zero/very small numbers
                    normalized_shares = {p: max(0, s / total_share) for p, s in sample_shares_dict.items()}
                    # Renormalize again after clipping negatives
                    final_total = sum(normalized_shares.values())
                    if final_total > 1e-9:
                         normalized_shares = {p: s / final_total for p, s in normalized_shares.items()}
                    else: # If all were negative or zero initially
                         normalized_shares = {p: 0.0 for p in political_families} # Assign zero to all
                else:
                    normalized_shares = {p: 0.0 for p in political_families} # Assign zero if initial sum is zero/negative
                
                # Convert to pandas Series, ensuring all political families are present
                sample_shares_series = pd.Series(normalized_shares).reindex(political_families, fill_value=0.0)

                # Calculate Swing for this sample (Predicted Shares - Baseline Shares)
                national_swing = sample_shares_series - last_election_national_shares
                
                # Apply Uniform Swing to District Shares
                # Add swing: df.add(series, axis='columns' or 1)
                forecasted_district_shares = last_election_district_shares.add(national_swing, axis=1)
                
                # Adjust and Re-normalize District Shares
                # 1. Clip negative shares to zero
                forecasted_district_shares = forecasted_district_shares.clip(lower=0)
                # 2. Re-normalize rows (districts) so they sum to 1
                row_sums = forecasted_district_shares.sum(axis=1)
                # Avoid division by zero for districts where all forecasted shares became zero
                valid_rows_mask = row_sums > 1e-9
                if valid_rows_mask.any():
                    forecasted_district_shares.loc[valid_rows_mask] = forecasted_district_shares.loc[valid_rows_mask].div(row_sums[valid_rows_mask], axis=0)
                # Ensure districts with zero sum remain zero (already handled by clipping and division logic, but explicit check is safe)
                forecasted_district_shares.loc[~valid_rows_mask, :] = 0.0 
                                    
                # Convert Forecasted Shares back to Estimated Vote Counts
                # Multiply shares by the baseline total votes for each district
                # Ensure alignment using reindexed baseline totals
                forecasted_district_counts_df = forecasted_district_shares.mul(district_total_votes_baseline, axis=0)
                # Round to nearest integer for vote counts
                forecasted_district_counts_df = forecasted_district_counts_df.round().astype(int)

                # Allocate Seats district by district using D'Hondt
                sample_total_seats = {party: 0 for party in political_families}
                try:
                    for circulo_name, num_seats in district_config.items():
                        # Get vote counts for the current district as a dict
                        # Use .get(circulo_name, pd.Series(0, index=political_families)) for robustness if a district is missing counts
                        votes_series = forecasted_district_counts_df.loc[circulo_name] if circulo_name in forecasted_district_counts_df.index else pd.Series(0, index=political_families)
                        votes_dict = votes_series.to_dict()
                        
                        # Skip allocation if no seats are assigned to the district or total votes are zero
                        if num_seats > 0 and sum(votes_dict.values()) > 0:
                             district_seat_allocation = calculate_dhondt(votes_dict, num_seats)
                             # Aggregate seats nationally
                             for party, seats in district_seat_allocation.items():
                                 if party in sample_total_seats: sample_total_seats[party] += seats
                    
                    # Store the total seats for this sample
                    sample_total_seats['sample_index'] = i # Add sample index for reference
                    seat_allocation_samples.append(sample_total_seats)
                except Exception as dhondt_err:
                        print(f"Warning: Error during D'Hondt allocation for sample {i}, district {circulo_name}: {dhondt_err}")
                        # Optionally skip this sample or handle error differently
                        continue # Skip to next sample
            
            loop_end_time = time.time()
            print(f"Seat simulation loop finished in {loop_end_time - loop_start_time:.2f} seconds.")
            # --- End Simulation Loop --- 

            # --- Process Seat Simulation Results --- 
            if seat_allocation_samples:
                seats_df = pd.DataFrame(seat_allocation_samples).fillna(0)
                # Ensure all political families are columns, even if they got 0 seats everywhere
                for party in political_families:
                    if party not in seats_df.columns:
                        seats_df[party] = 0
                # Define column order (sample_index first, then sorted parties)
                party_cols = sorted([p for p in political_families if p in seats_df.columns]) # Get existing party cols sorted
                cols_order = ['sample_index'] + party_cols
                seats_df = seats_df[cols_order]
                # Ensure party columns are integer type
                seats_df[party_cols] = seats_df[party_cols].astype(int)
                
                print("\n--- Seat Prediction Simulation Summary ---")
                # Calculate summary statistics using ArviZ
                # Convert party columns to a dictionary suitable for az.summary
                seat_summary = az.summary(seats_df[party_cols].to_dict(orient='list'), 
                                            hdi_prob=hdi_prob, 
                                            kind='stats', # Use 'stats' for mean, sd, hdi
                                            round_to=1)
                print(seat_summary.to_string())
                print("------------------------------------------")
                
                # Save results to CSV
                try:
                    seats_samples_path = os.path.join(pred_dir, f"seat_samples_{prediction_date_mode}.csv")
                    seats_df.to_csv(seats_samples_path, index=False)
                    print(f"Full seat prediction samples saved to {seats_samples_path}")
                    
                    seat_summary_path = os.path.join(pred_dir, f"seat_summary_{prediction_date_mode}.csv")
                    seat_summary.to_csv(seat_summary_path)
                    print(f"Seat prediction summary saved to {seat_summary_path}")
                    
                    # Assign results to be returned
                    seats_df_results = seats_df
                    seat_summary_results = seat_summary
                    
                except Exception as save_err:
                    print(f"Warning: Failed to save seat prediction results: {save_err}")

            else:
                    print("\nSeat prediction simulation did not produce any valid samples.")
        
        except Exception as simulation_loop_err:
            print(f"Error during seat simulation loop: {simulation_loop_err}")
            if debug: traceback.print_exc()
            # Ensure None is returned if the loop fails
            return None, None
            
    # Return the collected results (or None if simulation failed)
    return seats_df_results, seat_summary_results 