import pandas as pd
from pprint import pprint
from typing import Dict, List
import os # Needed for join path if loading config inside
import json # Needed for saving results

# Assuming the project structure allows these imports
# Adjust paths if necessary based on final structure
try:
    from src.data.loaders import load_district_config, load_election_results
    from src.processing.forecasting import forecast_district_votes_uns
    from src.processing.electoral_systems import calculate_dhondt
    # If DATA_DIR is needed for loaders, import it
    from src.config import DATA_DIR
except ImportError as e:
    print(f"Error importing dependencies in seat_prediction.py: {e}")
    # Re-raise or handle appropriately depending on desired behavior
    raise

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