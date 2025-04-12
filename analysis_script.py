import arviz as az
import pandas as pd
import xarray as xr
import numpy as np
import traceback

# Configuration
trace_path = "outputs/dynamic_gp_two_timescale_pen100/trace.zarr"
parties_of_interest = ["IL", "CH", "L", "PAN"]
end_date_exclusive = pd.Timestamp("2018-01-01")
variables_to_analyze = ["latent_mu_calendar", "baseline_effect_calendar", "short_term_effect_calendar"]

print(f"--- Analyzing Posterior from: {trace_path} ---")
print(f"Parties: {parties_of_interest}")
print(f"Period: Before {end_date_exclusive.date()}")
print("-" * 40)

try:
    # Load inference data using from_zarr
    idata = az.from_zarr(trace_path)
    posterior = idata.posterior

    # Get calendar time coordinates and create a boolean mask for the early period
    calendar_time_coords = pd.to_datetime(posterior["calendar_time"].values)
    early_time_mask = calendar_time_coords < end_date_exclusive

    if not np.any(early_time_mask):
        print(f"Error: No time points found before {end_date_exclusive.date()}")
    else:
        print(f"Found {np.sum(early_time_mask)} time points in the early period.")
        print("-" * 40)

        for party in parties_of_interest:
            print(f"Party: {party}")
            try:
                # Select the party
                posterior_party = posterior.sel(parties_complete=party)

                for var_name in variables_to_analyze:
                    if var_name in posterior_party:
                        # Select the variable and the early time period
                        # Flatten samples across chains and draws, then filter by time
                        var_samples_flat = posterior_party[var_name].stack(sample=("chain", "draw"))
                        early_samples = var_samples_flat.sel(calendar_time=early_time_mask)

                        if early_samples.size == 0:
                            print(f"  {var_name}: No samples found for this party in the early period.")
                            continue

                        # Calculate statistics across all time points and samples in the early period
                        # Ensure we handle potential numpy scalar types correctly
                        mean_val = float(early_samples.mean())
                        # Pass numpy array to az.hdi for compatibility
                        hdi_val = az.hdi(early_samples.values, hdi_prob=0.94)

                        # Extract HDI bounds correctly
                        try:
                            hdi_lower = float(hdi_val[0])
                            hdi_upper = float(hdi_val[1])
                            print(f"  {var_name}: Mean={mean_val:.3f}, 94% HDI=({hdi_lower:.3f}, {hdi_upper:.3f})")
                        except Exception as hdi_e:
                             print(f"    Error accessing HDI values for {var_name}: {hdi_e}")
                             print(f"    HDI object type: {type(hdi_val)}")
                             print(f"    HDI object content: {hdi_val}")

                    else:
                        print(f"  Variable '{var_name}' not found in posterior for party {party}.")

            except KeyError:
                print(f"  Party '{party}' not found in coordinates.")
            print("-" * 20)

except FileNotFoundError:
    print(f"Error: Trace file not found at {trace_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    traceback.print_exc()

print(f"--- Analysis Complete ---") 