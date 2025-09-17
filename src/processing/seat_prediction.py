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
import itertools # Added for chain/draw iteration
from collections import defaultdict, Counter # Added Counter for modal scenario
import pandas as pd # Needed for DataFrame manipulation

# Assuming the project structure allows these imports
# Adjust paths if necessary based on final structure
try:
    from src.data.loaders import load_district_config, load_election_results
    from src.processing.forecasting import forecast_district_votes_uns
    from src.processing.electoral_systems import calculate_dhondt, create_electoral_system
    # If DATA_DIR is needed for loaders, import it
    from src.config import DATA_DIR
    from src.data.dataset import ElectionDataset # Needed for type hinting and accessing dataset properties
    from src.visualization.plots import plot_seat_distribution_histograms # Needed if plotting is moved here
except ImportError as e:
    print(f"Error importing dependencies in seat_prediction.py: {e}")
    # Re-raise or handle appropriately depending on desired behavior
    raise

# Define type hints for clarity
XrDataArray = xr.DataArray # Use string forward reference if xarray isn't imported directly
XarrayDataset = xr.Dataset # Use string forward reference for the posterior

def calculate_seat_predictions_with_system(
    national_forecast_shares: Dict[str, float],
    last_election_date: str,
    political_families: List[str],
    election_dates: List[str],
    electoral_system_type: str = 'dhondt',
    **system_kwargs
) -> Dict[str, int]:
    """
    Predicts seat allocation using a configurable electoral system.

    This is the new, flexible version that supports different electoral systems.
    For parliamentary elections, use 'dhondt'. For mayoral elections, use 'mayoral'.

    Args:
        national_forecast_shares (Dict[str, float]): The forecasted national vote shares (MUST be normalized).
        last_election_date (str): The date of the last election (YYYY-MM-DD) to use as baseline for swing.
        political_families (List[str]): List of relevant party names.
        election_dates (List[str]): Full list of historical election dates required by data loaders.
        electoral_system_type (str): Type of electoral system ('dhondt', 'mayoral').
        **system_kwargs: Additional arguments for the electoral system.

    Returns:
        Dict[str, int]: A dictionary mapping party/candidate names to their predicted total seats/positions.
                        Returns None if critical errors occur (e.g., data loading failure).
    """
    print(f"--- Calculating Seat Predictions using {electoral_system_type} system ---")

    # Create the electoral system
    try:
        electoral_system = create_electoral_system(electoral_system_type, **system_kwargs)
        print(f"Using electoral system: {electoral_system.get_system_name()}")
    except ValueError as e:
        print(f"Error: {e}")
        return None

    # 1. Load District Configuration (Seats per District)
    district_config = load_district_config()
    if not district_config:
        print("Error: Failed to load district configuration. Aborting seat prediction run.")
        return None

    # 2. Forecast District Votes using UNS
    forecasted_district_counts = forecast_district_votes_uns(
        national_forecast_shares=national_forecast_shares,
        last_election_date=last_election_date,
        political_families=political_families,
        election_dates=election_dates
    )

    if forecasted_district_counts is None or forecasted_district_counts.empty:
        print("Error: Failed to forecast district votes.")
        return None

    # 3. Allocate Seats per District using the selected electoral system
    total_seats_allocation = {party: 0 for party in political_families}

    if 'Circulo' not in forecasted_district_counts.columns:
        print("Error: Forecasted district counts DataFrame is missing 'Circulo' column.")
        return None

    num_districts_processed = 0
    for _, district_row in forecasted_district_counts.iterrows():
        try:
            circulo_name = district_row['Circulo']
            if circulo_name not in district_config:
                continue

            num_seats = district_config[circulo_name]
            # For mayoral elections, num_seats should be 1
            if electoral_system_type.lower() == 'mayoral':
                num_seats = 1

            votes_dict = district_row[political_families].astype(int).to_dict()

            # Use the new electoral system
            district_seat_allocation_dict = electoral_system.allocate_seats(votes_dict, num_seats)

            for party, seats in district_seat_allocation_dict.items():
                if party in total_seats_allocation:
                    total_seats_allocation[party] += seats
            num_districts_processed += 1

        except Exception as e:
            print(f"Error processing row for district '{district_row.get('Circulo', 'Unknown')}': {e}")
            continue

    print(f"Seat allocation completed for {num_districts_processed} districts using {electoral_system.get_system_name()}.")

    return total_seats_allocation


def calculate_seat_predictions(
    national_forecast_shares: Dict[str, float],
    last_election_date: str,
    political_families: List[str],
    election_dates: List[str]
) -> Dict[str, int]:
    """
    Predicts seat allocation for parliamentary elections using D'Hondt system.

    This is the standard function for Portuguese parliamentary elections. For different
    election types (e.g., municipal mayoral), use calculate_seat_predictions_with_system().

    Args:
        national_forecast_shares (Dict[str, float]): The forecasted national vote shares (MUST be normalized).
        last_election_date (str): The date of the last election (YYYY-MM-DD) to use as baseline for swing.
        political_families (List[str]): List of relevant party names.
        election_dates (List[str]): Full list of historical election dates required by data loaders.

    Returns:
        Dict[str, int]: A dictionary mapping party names to their predicted total seats.
                        Returns None if critical errors occur (e.g., data loading failure).
    """
    # Use the new system internally while maintaining the same API
    return calculate_seat_predictions_with_system(
        national_forecast_shares=national_forecast_shares,
        last_election_date=last_election_date,
        political_families=political_families,
        election_dates=election_dates,
        electoral_system_type='dhondt'
    ) 

def simulate_seat_allocation(
    district_vote_share_posterior: xr.DataArray, # Expect district-level shares (chain, draw, district, party)
    dataset: ElectionDataset,
    num_samples_for_seats: int,
    pred_dir: str,
    prediction_date_mode: str,
    hdi_prob: float = 0.94,
    debug: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Simulates seat allocation based on DIRECT district-level posterior vote shares.
    Generates national seat summaries and detailed district-level seat change profiles.

    Args:
        district_vote_share_posterior: Xarray DataArray containing posterior samples
                                       of district-level vote shares. Must have dimensions 'chain',
                                       'draw', a district dimension, and a party dimension coordinate.
                                       Values should represent probabilities.
        dataset: The ElectionDataset object containing political_families, election_dates, etc.
        num_samples_for_seats: The maximum number of posterior samples to use for the simulation.
        pred_dir: Directory to save prediction outputs (like CSVs and JSONs).
        prediction_date_mode: String indicating the prediction mode ('election_day', etc.) for filenames.
        hdi_prob: The probability mass to include in the highest density interval for summary stats.
        debug: Boolean flag for enabling debug print statements.

    Returns:
        A tuple containing:
        - pd.DataFrame: DataFrame with overall national seat allocation samples (or None if simulation fails).
        - pd.DataFrame: DataFrame with overall national seat allocation summary statistics (or None if simulation fails).
    """
    print(f"\nStarting seat prediction simulation using DIRECT district shares (up to {num_samples_for_seats} samples)...")
    # Initialize return values for national summary
    seats_df_results: Optional[pd.DataFrame] = None
    seat_summary_results: Optional[pd.DataFrame] = None

    # Extract necessary info from dataset
    political_families = dataset.political_families
    election_dates = dataset.election_dates # Needed for baseline vote loading
    last_election_date_for_baseline_votes = None

    # --- Pre-load Data --- (Same as before)
    pre_load_successful = False
    district_config = None
    district_total_votes_baseline = None
    all_district_names = None
    try:
        if debug: print("Pre-loading data for seat simulation (Config & Baseline Totals)...")
        # 1. Find latest election date string
        if election_dates:
            parsed_dates = [pd.to_datetime(d, errors='coerce') for d in election_dates]
            valid_dates = [d for d in parsed_dates if pd.notna(d)]
            if valid_dates:
                last_election_date_dt = max(valid_dates)
                # Find the original string representation (handling potential mixed types)
                original_indices = [i for i, dt in enumerate(parsed_dates) if dt == last_election_date_dt]
                if original_indices:
                    original_index = original_indices[0]
                    if isinstance(election_dates[original_index], str):
                        last_election_date_for_baseline_votes = election_dates[original_index]
                    else:
                        try: 
                            last_election_date_for_baseline_votes = pd.to_datetime(election_dates[original_index]).strftime('%Y-%m-%d')
                        except Exception: 
                            last_election_date_for_baseline_votes = last_election_date_dt.strftime('%Y-%m-%d')
                else: 
                    last_election_date_for_baseline_votes = last_election_date_dt.strftime('%Y-%m-%d')
        if not last_election_date_for_baseline_votes: raise ValueError("Could not determine last election date.")
        if debug: print(f"(Using {last_election_date_for_baseline_votes} election for baseline district totals)")
        # 2. Load District Config
        district_config = load_district_config()
        if not district_config: raise ValueError("Failed to load district configuration.")
        all_district_names = list(district_config.keys())
        if debug: print(f"Loaded config for {len(all_district_names)} districts.")
        # 3. Load Baseline District Total Votes
        district_results_df_all = load_election_results(election_dates, political_families, aggregate_national=False)
        last_election_district_df = district_results_df_all[district_results_df_all['date'] == pd.to_datetime(last_election_date_for_baseline_votes)].copy()
        if last_election_district_df.empty: raise ValueError(f"No district results for baseline {last_election_date_for_baseline_votes}")
        if 'Circulo' not in last_election_district_df.columns: raise ValueError("'Circulo' column missing.")
        if 'sample_size' not in last_election_district_df.columns: raise ValueError("'sample_size' column missing.")
        last_election_district_df['Circulo'] = last_election_district_df['Circulo'].astype(str)
        last_election_district_df = last_election_district_df.set_index('Circulo')
        district_total_votes_baseline = pd.to_numeric(last_election_district_df['sample_size'], errors='coerce').fillna(0)
        district_total_votes_baseline = district_total_votes_baseline.reindex(all_district_names, fill_value=0)
        if debug: print("Loaded baseline district total votes.")
        print("Pre-loading complete.")
        pre_load_successful = True
    except Exception as preload_err:
        print(f"Error pre-loading data: {preload_err}")
        if debug: 
             traceback.print_exc()
        return None, None
    # --- End Pre-load Data ---

    if pre_load_successful:
        # --- Calculate ACTUAL 2024 Seats --- 
        seats2024 = defaultdict(lambda: defaultdict(int))
        try:
            print("Calculating actual seat allocation for 2024 election...")
            results_2024_districts = dataset.results_mult_district[
                dataset.results_mult_district['election_date'] == pd.to_datetime('2024-03-10')
            ]
            if results_2024_districts.empty:
                 print("Warning: 2024 district results not found in dataset. Cannot calculate delta2024.")
            else:
                for circulo_name, num_seats in district_config.items():
                    district_votes_2024 = results_2024_districts[results_2024_districts['Circulo'] == circulo_name]
                    if district_votes_2024.empty:
                        # print(f"Debug: No 2024 results found for district '{circulo_name}'. Assigning 0 seats.")
                        seats2024[circulo_name] = {party: 0 for party in political_families}
                        continue
                    # Ensure we only take the first row if multiple matches (shouldn't happen with date filter)
                    district_votes_2024 = district_votes_2024.iloc[0]
                    votes_dict_2024 = {}
                    for party in political_families:
                        if party in district_votes_2024.index:
                             votes_dict_2024[party] = int(district_votes_2024[party])
                        else:
                             votes_dict_2024[party] = 0
                    # Calculate seats for 2024
                    actual_seats_2024 = calculate_dhondt(votes_dict_2024, num_seats)
                    seats2024[circulo_name] = actual_seats_2024
                print("Actual 2024 seat calculation complete.")
        except Exception as e:
            print(f"Error calculating 2024 seats: {e}")
            if debug: traceback.print_exc()
        # --- End 2024 Seat Calculation --- 

        # --- Initialization for NEW contested summary logic --- 
        # Stores list of seat dicts per district: {district: [ {party:seats}, {party:seats}, ... ]}
        district_samples = defaultdict(list)
        # --- End Initialization ---
        
        # --- Initialization for national summary --- 
        seat_allocation_samples = [] # Still needed for national summary & seat_simulations.json
        # --- End Initialization ---

        # --- Dynamically find coordinate names --- (Same as before)
        party_dim_name = None; party_coord_values = None
        district_dim_name = None; district_coord_values = None
        expected_num_parties = len(political_families)
        expected_num_districts = len(all_district_names)
        target_data_array = None
        if isinstance(district_vote_share_posterior, xr.Dataset):
            if not district_vote_share_posterior.data_vars: print("Error: Input Dataset empty."); return None, None
            data_var_name = list(district_vote_share_posterior.data_vars)[0]
            if debug: print(f"Using data variable '{data_var_name}'")
            target_data_array = district_vote_share_posterior[data_var_name]
        elif isinstance(district_vote_share_posterior, xr.DataArray): target_data_array = district_vote_share_posterior; print("Using input DataArray.")
        else: print(f"Error: Unsupported input type: {type(district_vote_share_posterior)}"); return None, None
        if 'chain' not in target_data_array.dims or 'draw' not in target_data_array.dims: print("Error: Missing chain/draw dims."); return None, None
        for dim_name in target_data_array.dims:
            if dim_name in target_data_array.coords:
                coord = target_data_array[dim_name]; coord_values_list = coord.values.tolist()
                if coord.size == expected_num_parties and sorted(coord_values_list) == sorted(political_families): party_dim_name = dim_name; party_coord_values = coord_values_list; print(f"Found party dim: '{party_dim_name}'")
                elif coord.size >= expected_num_districts and set(all_district_names).issubset(set(coord_values_list)): district_dim_name = dim_name; district_coord_values = coord_values_list; print(f"Found district dim: '{district_dim_name}'")
        if not party_dim_name: print(f"Error: Could not find party dim."); return None, None
        if not district_dim_name: print(f"Error: Could not find district dim."); return None, None
        # --- End Coordinate Finding --- 

        # --- Start Simulation Loop ---
        total_samples_processed = 0
        try:
            stacked_posterior = target_data_array.stack(sample=("chain", "draw"))
            total_draws = len(stacked_posterior['sample'])
            samples_to_process = min(num_samples_for_seats, total_draws)
            print(f"Processing {samples_to_process} out of {total_draws} available posterior samples...")
            loop_start_time = time.time()
            for i in range(samples_to_process):
                if (i + 1) % 500 == 0: print(f"  Processed {i+1}/{samples_to_process} samples...")
                sample_district_shares_xr = stacked_posterior.isel(sample=i)
                sample_total_seats = {party: 0 for party in political_families}
                districts_processed_this_sample = 0
                for circulo_name, num_seats in district_config.items():
                    if num_seats <= 0: continue
                    baseline_votes = district_total_votes_baseline.get(circulo_name, 0)
                    if baseline_votes <= 0: continue
                    try:
                        current_district_shares_xr = sample_district_shares_xr.sel(**{district_dim_name: circulo_name})
                        current_district_shares_series = current_district_shares_xr.to_series().reindex(party_coord_values)
                        total_share = current_district_shares_series.sum()
                        if total_share > 1e-9: normalized_shares = (current_district_shares_series / total_share).clip(lower=0); final_total = normalized_shares.sum(); normalized_shares = normalized_shares / final_total if final_total > 1e-9 else pd.Series(0.0, index=normalized_shares.index)
                        else: normalized_shares = pd.Series(0.0, index=current_district_shares_series.index)
                        normalized_shares = normalized_shares.reindex(political_families, fill_value=0.0)
                        estimated_votes = (normalized_shares * baseline_votes).round().astype(int)
                        votes_dict = estimated_votes.to_dict()
                        if sum(votes_dict.values()) <= 0: continue
                        # Call updated calculate_dhondt, only gets dict
                        district_seat_allocation_dict = calculate_dhondt(votes_dict, num_seats)
                        # STORE district allocation for this sample (NEW)
                        district_samples[circulo_name].append(district_seat_allocation_dict)
                        # AGGREGATE national seats for this sample (Existing - for national summary)
                        for party, seats in district_seat_allocation_dict.items():
                            if party in sample_total_seats: sample_total_seats[party] += seats
                        districts_processed_this_sample += 1
                    except KeyError: continue
                    except Exception as district_err: print(f"Warning: Dist '{circulo_name}' sample {i} err: {district_err}"); continue
                if districts_processed_this_sample > 0: sample_total_seats['sample_index'] = i; seat_allocation_samples.append(sample_total_seats); total_samples_processed += 1
                elif debug: print(f"Note: No districts processed for sample {i}.")
            loop_end_time = time.time(); print(f"Seat simulation loop finished in {loop_end_time - loop_start_time:.2f} seconds.")
        except Exception as simulation_loop_err:
            print(f"Error during simulation loop: {simulation_loop_err}");
            if debug: 
                traceback.print_exc()
            return None, None
        # --- End Simulation Loop ---

        # --- Process National Seat Simulation Results (Existing Logic) ---
        if seat_allocation_samples:
            # --- AUGMENT WITH DIASPORA SCENARIOS ---
            print("\nAugmenting national seat samples with diaspora scenarios...")
            diaspora_scenarios_config = [
                {'name': 'S_2024', 'seats_delta': {'CH': 2, 'PS': 1, 'AD': 1}},
                {'name': 'S_2022', 'seats_delta': {'PS': 3, 'AD': 1}},
                {'name': 'S_2019', 'seats_delta': {'PS': 2, 'AD': 2}},
            ]
            
            # Convert list of dicts to DataFrame for augment_seats_with_diaspora_scenarios
            national_seats_df_temp = pd.DataFrame(seat_allocation_samples).fillna(0)
            for party in political_families: # Ensure all parties are columns before augmentation
                if party not in national_seats_df_temp.columns:
                    national_seats_df_temp[party] = 0
            
            # Call the augmentation function (already defined in this file)
            # Ensure party_names argument is correctly passed using political_families
            augmented_seats_df = augment_seats_with_diaspora_scenarios(
                national_seats_df=national_seats_df_temp,
                diaspora_scenarios=diaspora_scenarios_config,
                party_names=political_families 
            )
            
            if augmented_seats_df.empty:
                print("Warning: Augmentation with diaspora scenarios resulted in an empty DataFrame. Proceeding with national only.")
                seats_df = national_seats_df_temp # Fallback to national only
            else:
                print(f"Augmentation complete. Total samples including diaspora scenarios: {len(augmented_seats_df)}")
                seats_df = augmented_seats_df
            # --- END AUGMENTATION ---

            # Ensure 'sample_index' is not the actual index if it was added by augmentation logic directly
            # The augment_seats_with_diaspora_scenarios creates 'original_sample_id'
            # The main seats_df should be indexed 0 to N-1 for az.summary if not using original_sample_id
            # If 'sample_index' from national samples is problematic, reset index before summary
            # seats_df = seats_df.reset_index(drop=True) # Optional: if index issues arise

            for party in political_families: # Ensure all parties are columns in the final seats_df
                if party not in seats_df.columns: seats_df[party] = 0
            
            # Define party_cols based on political_families to ensure consistent order and inclusion
            party_cols = [p for p in political_families if p in seats_df.columns] # Use sorted if specific order is needed
            
            # Columns to keep for summary and output (excluding utility cols like 'diaspora_scenario_applied' from main stats)
            cols_for_summary = party_cols 
            
            # Ensure 'sample_index' or a similar unique ID per original national draw exists if needed by other parts
            # 'original_sample_id' is now present from the augmentation.
            # For az.summary, we typically pass a dictionary of arrays.
            
            print("\n--- Total Seat Prediction Summary (including Diaspora Scenarios) ---")
            # Prepare data for ArviZ summary: dictionary of numpy arrays
            summary_dict_data = {}
            for col in cols_for_summary:
                if col in seats_df:
                    summary_dict_data[col] = seats_df[col].values.astype(int) # Ensure integer type for seats
                else:
                    summary_dict_data[col] = np.zeros(len(seats_df), dtype=int)

            seat_summary = az.summary(summary_dict_data, hdi_prob=hdi_prob, kind='stats', round_to=1)
            print(seat_summary.to_string()); print("-"*60)
            try:
                # Filenames now reflect that these include diaspora effects
                file_suffix = f"total_seat_samples_direct_{prediction_date_mode}.csv"
                summary_suffix = f"total_seat_summary_direct_{prediction_date_mode}.csv"
                seats_samples_path = os.path.join(pred_dir, file_suffix); seats_df.to_csv(seats_samples_path, index=False); print(f"Total seat samples saved to {seats_samples_path}")
                seat_summary_path = os.path.join(pred_dir, summary_suffix); seat_summary.to_csv(seat_summary_path); print(f"Total seat summary saved to {seat_summary_path}")
                seats_df_results = seats_df; seat_summary_results = seat_summary
                
                # --- Save seat_forecast_simulations.json (Now includes diaspora effects and scenario tag) --- 
                seats_json_path = os.path.join(pred_dir, "seat_forecast_simulations.json")
                # Select relevant columns for JSON output (parties + scenario tag)
                json_output_cols = party_cols + ['diaspora_scenario_applied', 'original_sample_id']
                # Ensure all selected columns exist in seats_df
                json_output_cols = [col for col in json_output_cols if col in seats_df.columns]
                seats_json_data = seats_df[json_output_cols].to_dict(orient='records')
                
                with open(seats_json_path, 'w', encoding='utf-8') as f: json.dump(seats_json_data, f, ensure_ascii=False, indent=2)
                print(f"Seat forecast samples JSON (with diaspora scenarios) saved to {seats_json_path}")
            except Exception as save_err: 
                 print(f"Warning: Failed to save total seat results: {save_err}")
        else: 
             print("\nSeat prediction simulation did not produce valid national samples.")
        # --- End National Results Processing ---

        # --- NEW: Process District Seat Change Profiles --- 
        if total_samples_processed > 0 and district_samples:
             print(f"\nAggregating district seat change profiles...")
             contested_summary_output = {"districts": {}}
             delta_bins = [-2, -1, 0, 1, 2]
             aggregation_start_time = time.time()
             for circulo_name, list_of_sample_dicts in district_samples.items():
                 if not list_of_sample_dicts: continue # Skip if district had no samples
                 
                 # --- Find Modal Scenario --- 
                 # Convert each dict to a frozenset of items (sorted by party) to make it hashable
                 hashable_scenarios = [frozenset(sorted(d.items())) for d in list_of_sample_dicts]
                 # Find the most common hashable scenario
                 if not hashable_scenarios: # Handle empty list case
                     modal_scenario_hashable = frozenset()
                     modal_scenario_dict = {} # Empty scenario
                 else:
                     scenario_counts = Counter(hashable_scenarios)
                     modal_scenario_hashable, _ = scenario_counts.most_common(1)[0]
                     # Convert the modal frozenset back to a dictionary
                     modal_scenario_dict = dict(modal_scenario_hashable)
                 # --- End Find Modal Scenario --- 
                 
                 # Create DataFrame for the district (for individual party mode calculation)
                 district_df = pd.DataFrame(list_of_sample_dicts).fillna(0)
                 # Ensure all political families are columns
                 for party in political_families: 
                      if party not in district_df.columns: district_df[party] = 0
                 district_df = district_df[political_families].astype(int) # Keep only relevant parties

                 district_ENSC = 0.0 # ENSC still calculated based on individual party mode distribution
                 district_parties_output = {} # Individual party mode distributions
                 district_delta2024_scenario = {} # Delta vs modal scenario
                 
                 for party in political_families:
                     if party not in district_df.columns: continue # Should not happen now, but safe check
                     party_seats = district_df[party]
                     
                     # --- Calculate Delta vs Modal Scenario --- 
                     seats_in_modal_scenario = modal_scenario_dict.get(party, 0)
                     seats_actual_2024 = seats2024.get(circulo_name, {}).get(party, 0)
                     delta_vs_2024 = seats_in_modal_scenario - seats_actual_2024
                     district_delta2024_scenario[party] = delta_vs_2024
                     # --- End Delta vs Modal Scenario --- 
                     
                     # --- Keep calculation relative to INDIVIDUAL party mode for ENSC & P(delta) --- 
                     modes = party_seats.mode()
                     modal_p = int(modes[0] if not modes.empty else 0) # Individual mode
                     deltas = party_seats - modal_p # Deltas relative to individual mode
                     bucketed_deltas = deltas.clip(lower=-2, upper=2)
                     delta_counts = bucketed_deltas.value_counts()
                     party_probs = {}
                     total_prob = 0.0
                     for delta_val in delta_bins:
                         prob = delta_counts.get(delta_val, 0) / total_samples_processed
                         party_probs[str(delta_val)] = round(prob, 4) # Use string key for JSON
                         total_prob += prob
                         if delta_val != 0: # Only sum non-zero deltas for ENSC
                             district_ENSC += prob
                     district_parties_output[party] = party_probs
                     # --- End Individual party mode calculations --- 
                     
                 # Store district results
                 contested_summary_output["districts"][circulo_name] = {
                     "ENSC": round(district_ENSC, 4),
                     "delta2024": district_delta2024_scenario, # Use the scenario-based delta here
                     "parties": district_parties_output # Keep the individual party distributions
                 }
             aggregation_duration = time.time() - aggregation_start_time
             print(f"District profile aggregation finished in {aggregation_duration:.2f} seconds.")
             # Save the new contested summary JSON
             try:
                  contested_output_path = os.path.join(pred_dir, "contested_summary.json") # New filename
                  with open(contested_output_path, 'w', encoding='utf-8') as f:
                       json.dump(contested_summary_output, f, ensure_ascii=False, indent=2)
                  print(f"District seat change profiles saved to {contested_output_path}")
             except Exception as save_err:
                  print(f"Warning: Failed to save contested summary JSON: {save_err}")
        else:
             print("\nSkipping district profile aggregation: No samples processed or district data collected.")
        # --- END District Profile Processing --- 

    # Return the national results (same as before)
    return seats_df_results, seat_summary_results

def generate_national_trends_json(
    national_trend_posterior: xr.DataArray,
    pred_dir: str,
    hdi_prob: float = 0.94,
    debug: bool = False
) -> bool:
    """
    Generates the national_trends.json file for the dashboard.

    Args:
        national_trend_posterior (xr.DataArray): DataArray with posterior samples of national vote share
                                                 over time. Expected dims: (chain, draw, calendar_time, parties_complete).
        pred_dir (str): Directory to save the output JSON file.
        hdi_prob (float): Probability for HDI calculation.
        debug (bool): Enable debug printing.

    Returns:
        bool: True if the file was generated successfully, False otherwise.
    """
    print("\\nGenerating national trends JSON...")
    output_path = os.path.join(pred_dir, "national_trends.json")
    
    if not isinstance(national_trend_posterior, xr.DataArray):
        print("Error: national_trend_posterior must be an xarray DataArray.")
        return False
        
    party_dim_name = "parties_complete" # Use the correct dimension name
    time_dim_name = "calendar_time"
    required_dims = {"chain", "draw", time_dim_name, party_dim_name} # Updated expected dims
    if not required_dims.issubset(set(national_trend_posterior.dims)):
         print(f"Error: national_trend_posterior missing required dimensions. Found: {national_trend_posterior.dims}, Expected: {required_dims}")
         return False

    trends_data = []
    try:
        # Calculate mean and HDI
        mean_trend = national_trend_posterior.mean(dim=["chain", "draw"])
        # az.hdi returns a Dataset. We need to access the variable within it.
        hdi_result_dataset = az.hdi(national_trend_posterior, hdi_prob=hdi_prob)
        # Assuming the variable name in the result is the same as the input DataArray's name
        hdi_variable_name = national_trend_posterior.name
        if hdi_variable_name is None:
             # If the input DataArray had no name, try to guess (e.g., from the first variable if it was a dataset)
             # Or assign a default - let's assume 'latent_popularity_national' for now if unnamed
             # A better approach might be to ensure the input DA always has a name. 
             print(f"Warning: Input national_trend_posterior DataArray has no name. Assuming HDI variable name is 'latent_popularity_national'.")
             hdi_variable_name = 'latent_popularity_national' # Default guess
             
        if hdi_variable_name not in hdi_result_dataset:
            print(f"Error: HDI variable '{hdi_variable_name}' not found in az.hdi output dataset. Found: {list(hdi_result_dataset.data_vars)}")
            return False
            
        hdi_bounds = hdi_result_dataset[hdi_variable_name] # Access the DataArray within the Dataset
        
        # Get coordinates using the correct dimension names
        time_coords = national_trend_posterior[time_dim_name].values
        party_coords = national_trend_posterior[party_dim_name].values
        
        for party in party_coords:
            # Select party from the HDI DataArray 
            party_hdi_da = hdi_bounds.sel({party_dim_name: party})
            for i, date_val in enumerate(time_coords):
                # Convert numpy datetime64 to string
                date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
                # Select mean value using correct dimension names
                mean_val = mean_trend.sel({time_dim_name: date_val, party_dim_name: party}).item()
                
                # Select HDI bounds using isel on the HDI DataArray and call .item()
                hdi_low = party_hdi_da.isel({time_dim_name: i, 'hdi': 0}).item()
                hdi_high = party_hdi_da.isel({time_dim_name: i, 'hdi': 1}).item()
                
                trends_data.append({"date": date_str, "party": str(party), "metric": "vote_share_mean", "value": round(mean_val, 4)})
                trends_data.append({"date": date_str, "party": str(party), "metric": "vote_share_low", "value": round(hdi_low, 4)})
                trends_data.append({"date": date_str, "party": str(party), "metric": "vote_share_high", "value": round(hdi_high, 4)})

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trends_data, f, ensure_ascii=False, indent=2)
            
        print(f"National trends JSON saved successfully to {output_path}")
        return True

    except Exception as e:
        print(f"Error generating national trends JSON: {e}")
        if debug:
            traceback.print_exc()
        return False

# === Add back functions previously in dashboard_outputs.py ===

def generate_district_forecast_json(
    mean_district_shares: xr.DataArray,
    district_names: List[str], # Keep for potential validation, but use coords from mean_district_shares
    party_names: List[str], # Keep for potential validation, but use coords from mean_district_shares
    pred_dir: str,
    debug: bool = False
) -> bool:
    """
    Generates the district_forecast.json file for the dashboard.

    Args:
        mean_district_shares (xr.DataArray): DataArray containing the mean posterior vote share
                                             for each party in each district. Expected dims:
                                             (district_dim, party_dim).
        district_names (List[str]): Expected list of district names (primarily for debug checks).
        party_names (List[str]): Expected list of party names (primarily for debug checks).
        pred_dir (str): Directory to save the output JSON file.
        debug (bool): Enable debug printing.

    Returns:
        bool: True if the file was generated successfully, False otherwise.
    """
    print("\nGenerating district forecast JSON...")
    output_path = os.path.join(pred_dir, "district_forecast.json")
    
    # Verify input
    if not isinstance(mean_district_shares, xr.DataArray):
        print("Error: mean_district_shares must be an xarray DataArray.")
        return False
    if not mean_district_shares.dims or len(mean_district_shares.dims) != 2:
        print(f"Error: mean_district_shares has incorrect dimensions: {mean_district_shares.dims}. Expected 2.")
        return False
        
    # Dynamically get dimension names (assuming first is district, second is party)
    # A more robust approach might involve checking coordinate values if names aren't fixed
    district_dim = mean_district_shares.dims[0]
    party_dim = mean_district_shares.dims[1]
    
    actual_district_coords = mean_district_shares[district_dim].values.tolist()
    actual_party_coords = mean_district_shares[party_dim].values.tolist()

    # Use names from the DataArray coordinates as the source of truth
    district_names_to_use = actual_district_coords
    party_names_to_use = actual_party_coords

    # Optional: Check against expected names if needed for debugging
    if debug and sorted(actual_district_coords) != sorted(district_names):
         print(f"Debug Warning: Mismatch between district names in shares ({sorted(actual_district_coords)}) and expected ({sorted(district_names)}). Using names from shares.")
    if debug and sorted(actual_party_coords) != sorted(party_names):
         print(f"Debug Warning: Mismatch between party names in shares ({sorted(actual_party_coords)}) and expected ({sorted(party_names)}). Using names from shares.")

    forecast_data = []
    try:
        for district_name in district_names_to_use:
            district_data = mean_district_shares.sel({district_dim: district_name})
            
            probs_dict = {}
            valid_shares = {}
            for party in party_names_to_use:
                try:
                    share_value = district_data.sel({party_dim: party}).item()
                    if pd.notna(share_value) and isinstance(share_value, (int, float)):
                        share_float = float(share_value)
                        probs_dict[party] = round(share_float, 4) 
                        valid_shares[party] = share_float
                    else:
                         probs_dict[party] = 0.0
                         if debug: print(f"Debug: Invalid share value ({share_value}) for {party} in {district_name}, setting to 0.0")
                except KeyError:
                     probs_dict[party] = 0.0
                     if debug: print(f"Debug: Party '{party}' not found in shares for district '{district_name}', setting prob to 0.0")

            if valid_shares:
                winning_party = max(valid_shares, key=valid_shares.get)
            else:
                winning_party = "N/A"
                if debug: print(f"Debug: No valid shares found for {district_name}, cannot determine winner.")

            forecast_data.append({
                "district_name": district_name,
                "winning_party": winning_party,
                "probs": probs_dict
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(forecast_data, f, ensure_ascii=False, indent=2)
        
        print(f"District forecast JSON saved successfully to {output_path}")
        return True

    except Exception as e:
        print(f"Error generating district forecast JSON: {e}")
        if debug:
            traceback.print_exc()
        return False

def generate_house_effect_json(
    posterior_trace: xr.Dataset, # Expect InferenceData.posterior
    dataset: ElectionDataset,
    pred_dir: str,
    debug: bool = False
) -> bool:
    """
    Generates the house_effects.json file for the dashboard, filtering for pollsters
    active in the current election cycle.

    Args:
        posterior_trace (xr.Dataset): The posterior group from an InferenceData object,
                                        containing 'house_effects'.
        dataset (ElectionDataset): The dataset object containing poll data and election dates.
        pred_dir (str): Directory to save the output JSON file.
        debug (bool): Enable debug printing.

    Returns:
        bool: True if the file was generated successfully, False otherwise.
    """
    print("\nGenerating house effects JSON (filtered for active pollsters)...")
    output_path = os.path.join(pred_dir, "house_effects.json")
    
    variable_name = "house_effects" # Correct variable name
    pollster_dim = "pollsters"     # Expected pollster dimension name
    party_dim = "parties_complete" # Expected party dimension name

    if variable_name not in posterior_trace:
        print(f"Error: '{variable_name}' variable not found in posterior trace.")
        return False
        
    house_effects_da = posterior_trace[variable_name]
    
    # Check for expected dimensions
    required_dims = {"chain", "draw", pollster_dim, party_dim} # Use defined dimension names
    if not required_dims.issubset(set(house_effects_da.dims)):
         print(f"Error: '{variable_name}' missing required dimensions. Found: {house_effects_da.dims}, Expected: {required_dims}")
         return False

    # --- Identify Active Pollsters --- 
    active_pollsters = set()
    try:
        if not hasattr(dataset, 'polls') or dataset.polls is None or dataset.polls.empty:
            print("Warning: dataset.polls is missing or empty. Cannot filter pollsters.")
        elif not hasattr(dataset, 'historical_election_dates') or not dataset.historical_election_dates:
            print("Warning: dataset.historical_election_dates is missing. Cannot determine current cycle.")
        else:
            last_hist_election_date = pd.to_datetime(dataset.historical_election_dates[0]) # Most recent is first
            target_election_date = pd.to_datetime(dataset.election_date)
            
            all_polls_df = dataset.polls.copy()
            # Ensure poll date column is datetime
            if 'date' not in all_polls_df.columns:
                 print("Warning: 'date' column not found in dataset.polls. Cannot filter pollsters.")
            else:
                 all_polls_df['date'] = pd.to_datetime(all_polls_df['date'], errors='coerce')
                 all_polls_df = all_polls_df.dropna(subset=['date'])
                 
                 # Filter polls within the current cycle (after last historical, up to target)
                 current_cycle_polls = all_polls_df[
                     (all_polls_df['date'] > last_hist_election_date) & 
                     (all_polls_df['date'] <= target_election_date) 
                 ]
                 
                 if 'pollster' not in current_cycle_polls.columns:
                      print("Warning: 'pollster' column not found in dataset.polls. Cannot identify active pollsters.")
                 elif not current_cycle_polls.empty:
                      active_pollsters = set(current_cycle_polls['pollster'].unique())
                      print(f"Identified {len(active_pollsters)} pollsters active since {last_hist_election_date.date()}." )
                 else:
                      print(f"No polls found between {last_hist_election_date.date()} and {target_election_date.date()}. Output will be empty.")
                      
    except Exception as e:
        print(f"Error identifying active pollsters: {e}")
        if debug: traceback.print_exc()
        # Proceed without filtering if error occurs? Or return False? Let's proceed for now.
    # --- End Identify Active Pollsters --- 

    house_effects_data = []
    try:
        mean_effects = house_effects_da.mean(dim=["chain", "draw"])
        
        # Get coordinate values from the DataArray itself
        pollster_coords = house_effects_da[pollster_dim].values.tolist()
        party_coords = house_effects_da[party_dim].values.tolist()

        # Optional: Verify coordinates match dataset attributes if needed
        if debug:
            dataset_pollsters = dataset.unique_pollsters.tolist() if hasattr(dataset, 'unique_pollsters') and dataset.unique_pollsters is not None else []
            dataset_parties = dataset.political_families if hasattr(dataset, 'political_families') else []
            if not dataset_pollsters or sorted(pollster_coords) != sorted(dataset_pollsters):
                 print(f"Debug Warning: Mismatch or missing unique_pollsters in dataset. Dataset: {sorted(dataset_pollsters)}. Using coordinates from trace: {sorted(pollster_coords)}")
            if not dataset_parties or sorted(party_coords) != sorted(dataset_parties):
                 print(f"Debug Warning: Mismatch or missing political_families in dataset. Dataset: {sorted(dataset_parties)}. Using coordinates from trace: {sorted(party_coords)}")

        filtered_count = 0
        for pollster_name in pollster_coords:
            # --- Apply Filter --- 
            if active_pollsters and pollster_name not in active_pollsters:
                if debug: print(f"  Skipping pollster: {pollster_name} (not active in current cycle)")
                filtered_count += 1
                continue
            # --- End Filter --- 
                
            for party_name in party_coords:
                try:
                    # Select mean effect using coordinate names (safer than index)
                    mean_effect = mean_effects.sel({pollster_dim: pollster_name, party_dim: party_name}).item()
                except KeyError:
                    # print(f"Warning: Could not find effect for {pollster_name}/{party_name}. Setting to 0.") # Less verbose
                    mean_effect = 0.0 # Default value if specific combo not found

                house_effects_data.append({
                    "pollster": pollster_name,
                    "party": party_name,
                    "house_effect": round(float(mean_effect), 4) # Ensure float before rounding
                })
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} inactive pollsters.")
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(house_effects_data, f, ensure_ascii=False, indent=2)
            
        print(f"House effects JSON saved successfully to {output_path}")
        return True

    except Exception as e:
        print(f"Error generating house effects JSON: {e}")
        if debug:
            traceback.print_exc()
        return False

def generate_poll_bias_json(
    posterior_trace: xr.Dataset, # Expect InferenceData.posterior
    pred_dir: str,
    debug: bool = False
) -> bool:
    """
    Generates the poll_bias.json file for the dashboard.

    Args:
        posterior_trace (xr.Dataset): The posterior group from an InferenceData object,
                                        containing 'poll_bias'.
        pred_dir (str): Directory to save the output JSON file.
        debug (bool): Enable debug printing.

    Returns:
        bool: True if the file was generated successfully, False otherwise.
    """
    print("\nGenerating poll bias JSON...")
    output_path = os.path.join(pred_dir, "poll_bias.json")
    
    variable_name = "poll_bias"
    party_dim = "parties_complete" # Expected party dimension name

    if variable_name not in posterior_trace:
        print(f"Error: '{variable_name}' variable not found in posterior trace.")
        return False
        
    poll_bias_da = posterior_trace[variable_name]
    
    # Check for expected dimensions
    # poll_bias typically has ("chain", "draw", "parties_complete")
    # or after .mean() it would just be ("parties_complete",)
    # For safety, check if party_dim is present.
    if party_dim not in poll_bias_da.dims:
         print(f"Error: '{variable_name}' missing required dimension '{party_dim}'. Found: {poll_bias_da.dims}")
         return False

    poll_bias_data = []
    try:
        # Mean across chain and draw dimensions if they exist
        dims_to_mean = [dim for dim in ["chain", "draw"] if dim in poll_bias_da.dims]
        if dims_to_mean:
            mean_bias_effects = poll_bias_da.mean(dim=dims_to_mean)
        else:
            mean_bias_effects = poll_bias_da # Assume already averaged or single value

        party_coords = poll_bias_da[party_dim].values.tolist()

        for party_name in party_coords:
            try:
                # Select mean effect using coordinate names
                mean_effect = mean_bias_effects.sel({party_dim: party_name}).item()
            except KeyError:
                if debug: print(f"Warning: Could not find bias for {party_name}. Setting to 0.")
                mean_effect = 0.0 # Default value if specific party not found
            except Exception as e:
                if debug: print(f"Warning: Error selecting bias for {party_name}: {e}. Setting to 0.")
                mean_effect = 0.0


            poll_bias_data.append({
                "party": party_name,
                "poll_bias": round(float(mean_effect), 4) # Ensure float before rounding
            })
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(poll_bias_data, f, ensure_ascii=False, indent=2)
            
        print(f"Poll bias JSON saved successfully to {output_path}")
        return True

    except Exception as e:
        print(f"Error generating poll bias JSON: {e}")
        if debug:
            traceback.print_exc()
        return False

def augment_seats_with_diaspora_scenarios(
    national_seats_df: pd.DataFrame,
    diaspora_scenarios: List[Dict[str, any]],
    party_names: List[str]
) -> pd.DataFrame:
    """
    Augments national seat prediction samples with diaspora seat scenarios.

    Args:
        national_seats_df (pd.DataFrame): DataFrame where each row is a sample of national seats
                                          (columns are party names, index is sample_id or range).
        diaspora_scenarios (List[Dict[str, any]]): A list of diaspora scenarios.
                                                  Each scenario is a dictionary with:
                                                  - 'name': str, name of the scenario (e.g., "S_2024")
                                                  - 'seats_delta': Dict[str, int], party-to-seat additions.
        party_names (List[str]): A list of all party codes/names.

    Returns:
        pd.DataFrame: A new DataFrame containing all national samples augmented by each
                      diaspora scenario. Includes a column 'diaspora_scenario_applied'.
    """
    if national_seats_df is None or national_seats_df.empty:
        print("Warning: National seats DataFrame is empty. Cannot augment with diaspora scenarios.")
        return pd.DataFrame()

    augmented_samples_list = []

    for national_sample_index, national_sample_row in national_seats_df.iterrows():
        for scenario in diaspora_scenarios:
            augmented_sample = national_sample_row.copy()
            augmented_sample['original_sample_id'] = national_sample_index 
            augmented_sample['diaspora_scenario_applied'] = scenario['name']
            
            scenario_deltas = scenario['seats_delta']
            for party in party_names:
                # Ensure party column exists in the augmented sample (initialized from national_sample_row)
                if party not in augmented_sample:
                    augmented_sample[party] = 0 # Should not happen if national_seats_df is well-formed
                
                # Add diaspora seats for the current party from the scenario
                augmented_sample[party] += scenario_deltas.get(party, 0)
            
            augmented_samples_list.append(augmented_sample)

    if not augmented_samples_list:
        return pd.DataFrame()
        
    return pd.DataFrame(augmented_samples_list)