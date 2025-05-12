import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import traceback # Added for exception printing
from typing import List, Dict, Optional

def generate_district_forecast_json(
    mean_district_shares: xr.DataArray,
    district_names: List[str],
    party_names: List[str],
    pred_dir: str,
    debug: bool = False
) -> bool:
    """
    Generates the district_forecast.json file for the dashboard.

    Args:
        mean_district_shares (xr.DataArray): DataArray containing the mean posterior vote share
                                             for each party in each district. Expected dims:
                                             (district_dim, party_dim).
        district_names (List[str]): List of district names corresponding to the district dimension.
        party_names (List[str]): List of party names corresponding to the party dimension.
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
         print(f"Debug Warning: Mismatch between district names in shares ({sorted(actual_district_coords)}) and expected ({sorted(district_names)}).")
    if debug and sorted(actual_party_coords) != sorted(party_names):
         print(f"Debug Warning: Mismatch between party names in shares ({sorted(actual_party_coords)}) and expected ({sorted(party_names)}).")

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
                        # Ensure value is treated as float before rounding
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

# --- Add other generation functions below ---
# def generate_national_trends_json(...):
#     pass

# def generate_house_effect_json(...):
#     pass


