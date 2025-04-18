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
    district_vote_share_posterior: xr.Dataset | xr.DataArray, # Expect district-level shares
    dataset: ElectionDataset,
    num_samples_for_seats: int,
    pred_dir: str,
    prediction_date_mode: str,
    hdi_prob: float = 0.94,
    debug: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Simulates seat allocation based on DIRECT district-level posterior vote shares.

    Args:
        district_vote_share_posterior: Xarray Dataset or DataArray containing posterior samples
                                       of district-level vote shares. Must have dimensions 'chain',
                                       'draw', a district dimension, and a party dimension coordinate.
                                       Values should represent probabilities (summing to 1 across parties
                                       for each district/sample).
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
    print(f"\nStarting seat prediction simulation using DIRECT district shares (up to {num_samples_for_seats} samples)...")
    # Initialize return values
    seats_df_results: Optional[pd.DataFrame] = None
    seat_summary_results: Optional[pd.DataFrame] = None

    # Extract necessary info from dataset
    political_families = dataset.political_families
    election_dates = dataset.election_dates
    last_election_date_for_baseline_votes = None # Renamed for clarity

    # --- Pre-load Data ---
    # We only need district_config and baseline district *total* votes now
    pre_load_successful = False # Initialize flag
    district_config = None
    district_total_votes_baseline = None
    all_district_names = None

    try:
        if debug: print("Pre-loading data for seat simulation (Config & Baseline Totals)...")
        # 1. Find latest election date string (needed for baseline vote totals)
        if election_dates:
            parsed_dates = [pd.to_datetime(d, errors='coerce') for d in election_dates]
            valid_dates = [d for d in parsed_dates if pd.notna(d)]
            if valid_dates:
                    last_election_date_dt = max(valid_dates)
                    # Find the original string representation
                    original_indices = [i for i, dt in enumerate(parsed_dates) if dt == last_election_date_dt]
                    if original_indices:
                        original_index = original_indices[0]
                        if isinstance(election_dates[original_index], str):
                            last_election_date_for_baseline_votes = election_dates[original_index]
                        else:
                            try:
                                last_election_date_for_baseline_votes = pd.to_datetime(election_dates[original_index]).strftime('%Y-%m-%d')
                            except Exception:
                                 last_election_date_for_baseline_votes = last_election_date_dt.strftime('%Y-%m-%d') # Fallback
                    else: # Fallback if original string index not found
                        last_election_date_for_baseline_votes = last_election_date_dt.strftime('%Y-%m-%d')

        if not last_election_date_for_baseline_votes:
                raise ValueError("Could not determine last election date for baseline vote total calculation.")
        if debug: print(f"(Using {last_election_date_for_baseline_votes} election to estimate baseline district total votes)")

        # 2. Load District Config
        district_config = load_district_config()
        if not district_config: raise ValueError("Failed to load district configuration.")
        all_district_names = list(district_config.keys()) # Get list of names
        if debug: print(f"Loaded config for {len(all_district_names)} districts.")

        # 3. Load District Results ONLY for Baseline Total Votes
        district_results_df_all = load_election_results(election_dates, political_families, aggregate_national=False)
        last_election_district_df = district_results_df_all[district_results_df_all['date'] == pd.to_datetime(last_election_date_for_baseline_votes)].copy()
        if last_election_district_df.empty: raise ValueError(f"Could not find district results for baseline date {last_election_date_for_baseline_votes}")
        if 'Circulo' not in last_election_district_df.columns: raise ValueError("'Circulo' column missing in loaded district results.")
        if 'sample_size' not in last_election_district_df.columns: raise ValueError("'sample_size' column missing in loaded district results.")

        # Ensure 'Circulo' is suitable as index
        last_election_district_df['Circulo'] = last_election_district_df['Circulo'].astype(str)
        last_election_district_df = last_election_district_df.set_index('Circulo')

        # Extract baseline total votes per district (sample_size)
        district_total_votes_baseline = pd.to_numeric(last_election_district_df['sample_size'], errors='coerce').fillna(0)
        # Reindex to ensure all configured districts are present, filling missing with 0
        district_total_votes_baseline = district_total_votes_baseline.reindex(all_district_names, fill_value=0)
        if debug: print("Loaded baseline district total votes.")

        # --- Removed baseline share calculations (national and district) ---

        print("Pre-loading complete.")
        pre_load_successful = True

    except Exception as preload_err:
            print(f"Error pre-loading data for seat simulation: {preload_err}")
            if debug: traceback.print_exc()
            return None, None # Return None tuple immediately if pre-load fails
    # --- End Pre-load Data ---

    # Proceed only if pre-loading was successful
    if pre_load_successful:
        # --- Dynamically find coordinate names ---
        party_dim_name = None
        party_coord_values = None
        district_dim_name = None
        district_coord_values = None
        expected_num_parties = len(political_families)
        expected_num_districts = len(all_district_names)

        # Identify target DataArray if input is Dataset
        if isinstance(district_vote_share_posterior, xr.Dataset):
            # Simple approach: use the first data variable. Improve if needed.
            if not district_vote_share_posterior.data_vars:
                 print("Error: Input Dataset has no data variables. Aborting.")
                 return None, None
            data_var_name = list(district_vote_share_posterior.data_vars)[0]
            if debug: print(f"Using data variable '{data_var_name}' from posterior Dataset.")
            target_data_array = district_vote_share_posterior[data_var_name]
        elif isinstance(district_vote_share_posterior, xr.DataArray):
            target_data_array = district_vote_share_posterior
            if debug: print(f"Using input DataArray directly.")
        else:
            print(f"Error: Unsupported type for district_vote_share_posterior: {type(district_vote_share_posterior)}")
            return None, None

        # Check for chain and draw dimensions
        if 'chain' not in target_data_array.dims or 'draw' not in target_data_array.dims:
            print(f"Error: Input posterior missing 'chain' or 'draw' dimension. Found: {list(target_data_array.dims)}")
            return None, None

        # Find party and district dimensions
        for dim_name in target_data_array.dims:
            if dim_name in target_data_array.coords:
                coord = target_data_array[dim_name]
                coord_values_list = coord.values.tolist()
                # Check for party dimension
                if coord.size == expected_num_parties and sorted(coord_values_list) == sorted(political_families):
                    party_dim_name = dim_name
                    party_coord_values = coord_values_list
                    if debug: print(f"Found party dimension: '{party_dim_name}' with values: {party_coord_values}")
                # Check for district dimension
                elif coord.size >= expected_num_districts and set(all_district_names).issubset(set(coord_values_list)):
                    # Allow posterior to have more districts than config (e.g. regions), but ensure all configured are present
                    district_dim_name = dim_name
                    # Use the actual coordinate values from the posterior for selection later
                    district_coord_values = coord_values_list
                    if debug: print(f"Found district dimension: '{district_dim_name}' (matching {len(set(all_district_names) & set(coord_values_list))}/{expected_num_districts} configured districts)")

        # Verify dimensions were found
        if not party_dim_name:
            print(f"Error: Could not find party dimension matching {political_families}. Coords checked: {list(target_data_array.coords.keys())}")
            return None, None
        if not district_dim_name:
            print(f"Error: Could not find district dimension matching configured districts ({expected_num_districts}). Coords checked: {list(target_data_array.coords.keys())}")
            return None, None

        # --- Start Simulation Loop ---
        seat_allocation_samples = []
        try:
            # Stack chain and draw dimensions
            stacked_posterior = target_data_array.stack(sample=("chain", "draw"))
            total_draws = len(stacked_posterior['sample'])
            samples_to_process = min(num_samples_for_seats, total_draws)
            print(f"Processing {samples_to_process} out of {total_draws} available posterior samples...")
            loop_start_time = time.time()

            for i in range(samples_to_process):
                if (i + 1) % 500 == 0: print(f"  Processed {i+1}/{samples_to_process} samples...")
                # Get the vote share sample for this iteration (dims should be district, party)
                sample_district_shares_xr = stacked_posterior.isel(sample=i)

                # Allocate Seats district by district using D'Hondt
                sample_total_seats = {party: 0 for party in political_families}
                districts_processed_this_sample = 0

                for circulo_name, num_seats in district_config.items():
                    # Skip districts with 0 seats assigned
                    if num_seats <= 0: continue

                    # Get baseline total votes for this district
                    baseline_votes = district_total_votes_baseline.get(circulo_name, 0)
                    # Skip if baseline votes are zero (cannot estimate counts)
                    if baseline_votes <= 0:
                        # if debug: print(f"Skipping district '{circulo_name}' for sample {i}: Zero baseline votes.")
                        continue

                    try:
                        # Extract predicted shares for this specific district and sample
                        # Use .sel() with the identified dimension names
                        current_district_shares_xr = sample_district_shares_xr.sel(**{district_dim_name: circulo_name})

                        # Convert shares to pandas Series for easier handling & normalization check
                        current_district_shares_series = current_district_shares_xr.to_series().reindex(party_coord_values)

                        # Normalize shares for robustness (should ideally sum to 1 already)
                        total_share = current_district_shares_series.sum()
                        if total_share > 1e-9:
                            normalized_shares = (current_district_shares_series / total_share).clip(lower=0)
                            # Renormalize after clipping just in case
                            final_total = normalized_shares.sum()
                            if final_total > 1e-9:
                                normalized_shares = normalized_shares / final_total
                            else:
                                normalized_shares.values[:] = 0.0 # Set all to 0 if sum is still ~0
                        else:
                            normalized_shares = pd.Series(0.0, index=current_district_shares_series.index)

                        # Ensure all political families are present before converting to votes
                        normalized_shares = normalized_shares.reindex(political_families, fill_value=0.0)

                        # Convert shares to estimated vote counts
                        estimated_votes = (normalized_shares * baseline_votes).round().astype(int)
                        votes_dict = estimated_votes.to_dict()

                        # Skip D'Hondt if total estimated votes are zero
                        if sum(votes_dict.values()) <= 0:
                            # if debug: print(f"Skipping district '{circulo_name}' for sample {i}: Zero estimated votes after share conversion.")
                            continue

                        # Calculate seat allocation for this district
                        district_seat_allocation = calculate_dhondt(votes_dict, num_seats)

                        # Aggregate seats nationally for this sample
                        for party, seats in district_seat_allocation.items():
                            if party in sample_total_seats:
                                sample_total_seats[party] += seats
                        districts_processed_this_sample += 1

                    except KeyError:
                         if debug: print(f"Warning: District '{circulo_name}' configured but not found in posterior coordinate '{district_dim_name}' for sample {i}. Skipping.")
                         continue # Skip this district if not in posterior
                    except Exception as district_err:
                         print(f"Warning: Error processing district '{circulo_name}' for sample {i}: {district_err}")
                         if debug: traceback.print_exc()
                         continue # Skip to next district on error

                # Store the total seats for this sample if any districts were processed
                if districts_processed_this_sample > 0:
                    sample_total_seats['sample_index'] = i # Add sample index for reference
                    seat_allocation_samples.append(sample_total_seats)
                elif debug:
                    print(f"Note: No districts processed for sample {i}.")


            loop_end_time = time.time()
            print(f"Seat simulation loop finished in {loop_end_time - loop_start_time:.2f} seconds.")
            # --- End Simulation Loop ---

            # --- Process Seat Simulation Results ---
            if seat_allocation_samples:
                seats_df = pd.DataFrame(seat_allocation_samples).fillna(0)
                # Ensure all political families are columns, even if they got 0 seats
                for party in political_families:
                    if party not in seats_df.columns:
                        seats_df[party] = 0
                # Define column order
                party_cols = sorted([p for p in political_families if p in seats_df.columns])
                cols_order = ['sample_index'] + party_cols
                seats_df = seats_df[cols_order]
                seats_df[party_cols] = seats_df[party_cols].astype(int)

                print("\n--- Seat Prediction Simulation Summary (Direct District Method) ---")
                # Calculate summary statistics
                # Convert relevant columns to a dict for arviz
                summary_dict = {col: seats_df[col].values for col in party_cols}
                seat_summary = az.summary(summary_dict,
                                            hdi_prob=hdi_prob,
                                            kind='stats',
                                            round_to=1)
                print(seat_summary.to_string())
                print("------------------------------------------------------------------")

                # Save results to CSV
                try:
                    # Add suffix to distinguish from potential UNS results
                    file_suffix = f"seat_samples_direct_{prediction_date_mode}.csv"
                    summary_suffix = f"seat_summary_direct_{prediction_date_mode}.csv"

                    seats_samples_path = os.path.join(pred_dir, file_suffix)
                    seats_df.to_csv(seats_samples_path, index=False)
                    print(f"Full seat prediction samples saved to {seats_samples_path}")

                    seat_summary_path = os.path.join(pred_dir, summary_suffix)
                    seat_summary.to_csv(seat_summary_path)
                    print(f"Seat prediction summary saved to {seat_summary_path}")

                    # Assign results to be returned
                    seats_df_results = seats_df
                    seat_summary_results = seat_summary

                except Exception as save_err:
                    print(f"Warning: Failed to save seat prediction results: {save_err}")

            else:
                    print("\nSeat prediction simulation (direct district method) did not produce any valid samples.")

        except Exception as simulation_loop_err:
            print(f"Error during seat simulation loop (direct district method): {simulation_loop_err}")
            if debug: traceback.print_exc()
            return None, None # Ensure None is returned if the loop fails

    # Return the collected results (or None if simulation failed)
    return seats_df_results, seat_summary_results 