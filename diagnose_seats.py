# diagnose_seats.py
import os
import argparse
import arviz as az
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import numbers # Import numbers for checking numeric types

# Assuming src is importable from where this script is run
# If not, adjust sys.path or run as a module if placed inside src
try:
    from src.models.elections_facade import ElectionsFacade
    from src.models.dynamic_gp_election_model import DynamicGPElectionModel
    from src.data.dataset import ElectionDataset
    from src.main import load_model # Reuse load_model for consistency
    from src.config import DATA_DIR # Assuming DATA_DIR might be needed by loaders indirectly
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure this script is run from the project root directory or adjust imports.")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Diagnose high seat predictions for specific parties.")
    parser.add_argument(
        "--load-dir",
        default="outputs/latest",
        help="Directory containing the saved model trace and config.",
    )
    parser.add_argument(
        "--parties",
        nargs='+',
        default=["L", "IL"],
        help="Parties to investigate.",
    )
    parser.add_argument(
        "--baseline-party",
        default="PS",
        help="A major party to compare district effects against.",
    )
    parser.add_argument(
        "--output-subdir",
        default="diagnostics/seat_investigation",
        help="Subdirectory within load-dir to save diagnostic plots.",
    )
    # --- Add Prediction Date Mode Argument ---
    parser.add_argument(
        "--prediction-date-mode",
        choices=['election_day', 'last_poll', 'today', 'specific_date'], # Add specific_date for manual override if needed
        default='last_poll', # Default to last_poll for original behavior
        help="Specify which date's latent popularity to use for diagnostics. Default: last_poll"
    )
    parser.add_argument(
        "--specific-date",
        help="Specific date (YYYY-MM-DD) to use if --prediction-date-mode=specific_date",
    )
    # --- End Add Argument ---
    parser.add_argument(
        "--debug", action="store_true", help="Enable debugging output",
    )

    # Dummy args object to pass to load_model, as it expects argparse output
    # We only really care about load_dir and potentially debug/model_type from config
    dummy_args_dict = {
        'load_dir': None, # Will be overridden
        'election_date': None,
        'baseline_timescale': None,
        'election_timescale': None,
        'cutoff_date': None,
        'model_type': None, # Will be read from config
        'baseline_gp_lengthscale': None,
        'baseline_gp_kernel': None,
        'cycle_gp_lengthscale': None,
        'cycle_gp_kernel': None,
        'cycle_gp_max_days': None,
        'hsgp_m': None,
        'hsgp_c': None,
        'hsgp_m_cycle': None,
        'hsgp_c_cycle': None,
        'debug': False,
        # --- Add new args to dummy dict ---
        'prediction_date_mode': None,
        'specific_date': None,
        'seat_prediction_samples': 0, # Not used here, but might be expected by load_model indirectly
    }
    cli_args = parser.parse_args()

    # Update dummy args with actual load_dir and debug flag
    dummy_args_dict['load_dir'] = cli_args.load_dir
    dummy_args_dict['debug'] = cli_args.debug
    dummy_args_dict['prediction_date_mode'] = cli_args.prediction_date_mode # Pass through
    dummy_args_dict['specific_date'] = cli_args.specific_date # Pass through
    dummy_args_ns = argparse.Namespace(**dummy_args_dict) # Convert dict to Namespace

    INVESTIGATE_PARTIES = cli_args.parties
    BASELINE_PARTY = cli_args.baseline_party
    PARTIES_TO_PLOT = INVESTIGATE_PARTIES + [BASELINE_PARTY]

    # --- 1. Load Model and Data ---
    print(f"--- Loading model from {cli_args.load_dir} ---")
    # Use a simplified Namespace for loading, relying on config inside load_dir
    elections_model = load_model(dummy_args_ns, cli_args.load_dir, debug=cli_args.debug)

    if elections_model is None or elections_model.trace is None or elections_model.dataset is None:
        print("Error: Failed to load model, trace, or dataset. Exiting.")
        return

    idata = elections_model.trace
    dataset = elections_model.dataset
    model_instance = elections_model.model_instance
    # Get election_date from facade if available
    target_election_date = elections_model.election_date

    # Ensure it's the Dynamic GP model
    if not isinstance(model_instance, DynamicGPElectionModel):
        print(f"Error: Loaded model is type {type(model_instance)}, not DynamicGPElectionModel.")
        print("This script specifically diagnoses the dynamic district model.")
        return

    # Define output directory
    plot_dir = os.path.join(cli_args.load_dir, cli_args.output_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving diagnostic plots to: {plot_dir}")

    # --- Print Posterior Keys ---
    print("\n--- Variables available in idata.posterior ---")
    if hasattr(idata, 'posterior'):
        print(sorted(list(idata.posterior.keys())))
    else:
        print("idata object does not have a 'posterior' group.")
    # --- End Print Posterior Keys ---

    # Get coordinates needed
    try:
        political_families = idata.posterior.coords['parties_complete'].values.tolist()
        districts = idata.posterior.coords['districts'].values.tolist()
        calendar_time = pd.to_datetime(idata.posterior.coords['calendar_time'].values)
    except KeyError as e:
        print(f"Error: Missing coordinate in idata posterior: {e}. Exiting.")
        return

    # Check if parties exist
    missing_parties = [p for p in PARTIES_TO_PLOT if p not in political_families]
    if missing_parties:
        print(f"Error: Parties {missing_parties} not found in model coordinates ({political_families}). Exiting.")
        return

    # --- Determine Target Date for Analysis ---
    print(f"\n--- Determining Target Date based on mode: {cli_args.prediction_date_mode} ---")
    analysis_target_date = None
    last_poll_date = None # Keep track of this separately if needed
    # Find last poll date from dataset first
    if hasattr(dataset, 'polls_train') and not dataset.polls_train.empty:
        last_poll_date = pd.Timestamp(dataset.polls_train['date'].max()).normalize()

    if cli_args.prediction_date_mode == 'election_day':
        if target_election_date:
            analysis_target_date = pd.Timestamp(target_election_date).normalize()
            print(f"Using election day: {analysis_target_date.date()}")
        else:
            print("Error: Cannot use 'election_day' mode, election date not found in loaded model/config.")
            return
    elif cli_args.prediction_date_mode == 'last_poll':
        if last_poll_date:
            analysis_target_date = last_poll_date
            print(f"Using last poll date: {analysis_target_date.date()}")
        else:
            print("Error: Cannot use 'last_poll' mode, failed to determine last poll date from dataset.")
            return
    elif cli_args.prediction_date_mode == 'today':
        analysis_target_date = pd.Timestamp.now().normalize()
        print(f"Using today's date: {analysis_target_date.date()}")
    elif cli_args.prediction_date_mode == 'specific_date':
        if cli_args.specific_date:
            try:
                analysis_target_date = pd.Timestamp(cli_args.specific_date).normalize()
                print(f"Using specific date: {analysis_target_date.date()}")
            except ValueError:
                print(f"Error: Invalid format for --specific-date '{cli_args.specific_date}'. Use YYYY-MM-DD.")
                return
        else:
            print("Error: --specific-date must be provided when --prediction-date-mode=specific_date.")
            return
    else: # Should not happen due to choices in argparse
        print(f"Internal Error: Unhandled prediction date mode '{cli_args.prediction_date_mode}'.")
        return

    if analysis_target_date is None:
        print("Error: Failed to set analysis_target_date. Exiting.")
        return
    # --- End Determine Target Date ---


    # --- 2. Get District Shares for Analysis Date ---
    print(f"\n--- Calculating District Vote Shares ({cli_args.prediction_date_mode}: {analysis_target_date.date()}) ---")
    district_shares_posterior = None
    try:
        # Use the determined analysis_target_date and mode
        district_shares_posterior = model_instance.get_district_vote_share_posterior(
            idata=idata,
            date_mode=cli_args.prediction_date_mode, # Pass the mode
            target_date=analysis_target_date # Pass the specific date (used only if mode='specific_date', but safe to pass always)
        )
        if district_shares_posterior is None:
            print("Error: Failed to calculate district vote shares.")
            # Proceed with other diagnostics if possible
        else:
            print("District shares calculated.")
    except Exception as e:
        print(f"Error calculating district shares: {e}")
        # Proceed if possible

    # <<< Add Inspection of district_shares_posterior >>>
    if district_shares_posterior is not None:
        try:
            print("\n--- Inspecting Raw Posterior Shares for L/IL (Before Averaging) ---")
            lisboa_shares_L = district_shares_posterior.sel(districts="Lisboa", parties_complete="L")
            lisboa_shares_IL = district_shares_posterior.sel(districts="Lisboa", parties_complete="IL")
            print(f"  Lisboa L Shares (Raw Posterior): Min={lisboa_shares_L.min().item():.6f}, Max={lisboa_shares_L.max().item():.6f}, Mean={lisboa_shares_L.mean().item():.6f}")
            print(f"  Lisboa IL Shares (Raw Posterior): Min={lisboa_shares_IL.min().item():.6f}, Max={lisboa_shares_IL.max().item():.6f}, Mean={lisboa_shares_IL.mean().item():.6f}")
            # Check another district
            aveiro_shares_L = district_shares_posterior.sel(districts="Aveiro", parties_complete="L")
            print(f"  Aveiro L Shares (Raw Posterior): Min={aveiro_shares_L.min().item():.6f}, Max={aveiro_shares_L.max().item():.6f}, Mean={aveiro_shares_L.mean().item():.6f}")
        except Exception as inspect_e:
            print(f"  Error inspecting raw posterior shares: {inspect_e}")
    # <<< End Inspection >>>

    # --- 3. Analyze Predicted Vote Shares ---
    if district_shares_posterior is not None:
        print(f"\n--- Mean Predicted Vote Shares (%) per District ({analysis_target_date.date()}) ---")
        try:
            mean_shares = district_shares_posterior.mean(dim=["chain", "draw"]) * 100
            shares_df = mean_shares.sel(parties_complete=INVESTIGATE_PARTIES).to_pandas()
            # Transpose for better readability if many districts
            if len(districts) > 10:
                shares_df = shares_df.T
            # Sort by a party's share or district name if desired
            # shares_df = shares_df.sort_values(by=INVESTIGATE_PARTIES[0], ascending=False)
            print(shares_df.to_string(float_format="%.2f"))

            # Save to CSV - Include date mode in filename
            shares_csv_path = os.path.join(plot_dir, f"mean_district_shares_{cli_args.prediction_date_mode}_{analysis_target_date.date().isoformat()}.csv")
            shares_df.to_csv(shares_csv_path, float_format="%.4f")
            print(f"Mean district shares saved to {shares_csv_path}")

            # --- Add Plotting of Share Distribution for specific district ---
            debug_district = "Lisboa" # Choose a district to investigate
            if debug_district in districts: # Check if district exists in coords
                 print(f"\n--- Plotting Posterior Share Distribution for {debug_district} ---")
                 plt.figure(figsize=(10, 5))
                 plot_parties = [p for p in INVESTIGATE_PARTIES if p in district_shares_posterior.coords['parties_complete'].values]
                 if plot_parties:
                     # Create a dictionary mapping labels to the DataArray slices
                     data_to_plot = {
                         f'{party} Shares': district_shares_posterior.sel(districts=debug_district, parties_complete=party)
                         for party in plot_parties
                     }
                     az.plot_density(
                         data_to_plot, # Pass the dictionary
                        # var_names and data_labels are inferred from the dictionary keys
                         hdi_prob=None, # Don't need HDI band for this visualization
                         point_estimate=None,
                     )
                     plt.title(f"Posterior Distribution of Vote Shares - {debug_district} ({analysis_target_date.date()})")
                     plt.xlabel("Vote Share Probability")
                     # Include date mode in filename
                     dist_plot_path = os.path.join(plot_dir, f"share_distribution_{debug_district}_{cli_args.prediction_date_mode}_{analysis_target_date.date().isoformat()}.png")
                     plt.savefig(dist_plot_path, bbox_inches='tight')
                     plt.close()
                     print(f"Share distribution plot saved to {dist_plot_path}")
                 else:
                     print(f"    Parties {INVESTIGATE_PARTIES} not found in posterior data for plotting distribution.")
            else:
                 print(f"    Skipping share distribution plot (district '{debug_district}' not found in coordinates)")
            # --- End Added Plotting ---

        except Exception as e:
            print(f"Error analyzing vote shares: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Skipping Vote Share Analysis (Calculation Failed) ---")


    # --- 4. Analyze Mean Latent District Support (Before Softmax) ---
    # Added step to investigate pre-softmax values
    if district_shares_posterior is not None: # Reuse check for district shares calc success
        print(f"\n--- Mean Latent District Support (Pre-Softmax) ({analysis_target_date.date()}) ---")
        try:
            # Get mean posterior values of components
            mean_national_trend_pt = idata.posterior["national_trend_pt"].mean(dim=["chain", "draw"])
            mean_base_offset_p = idata.posterior["base_offset_p"].mean(dim=["chain", "draw"])
            mean_beta_p = idata.posterior["beta_p"].mean(dim=["chain", "draw"])
            # Get national avg trend from posterior if it exists (calculated by get_district_vote_share_posterior potentially)
            if "national_avg_trend_p" in idata.posterior:
                 mean_national_avg_trend_p = idata.posterior["national_avg_trend_p"].mean(dim=["chain", "draw"])
            else:
                 # Recalculate if missing
                 print("DEBUG: Recalculating mean_national_avg_trend_p for latent support analysis.")
                 mean_national_avg_trend_p = mean_national_trend_pt.mean(dim="calendar_time")


            # Ensure calendar_time coordinate is datetime for selection
            mean_national_trend_pt['calendar_time'] = pd.to_datetime(mean_national_trend_pt['calendar_time'].values)

            # Select national trend at the target date
            mean_national_trend_at_date = mean_national_trend_pt.sel(
                calendar_time=analysis_target_date, # Use the determined date
                method="nearest",
                tolerance=pd.Timedelta(days=1)
            )

            # Calculate deviation
            mean_national_dev_at_date = mean_national_trend_at_date - mean_national_avg_trend_p

            # Calculate district adjustment (using mean values)
            # Expand dims for broadcasting
            # Ensure districts coordinate exists in mean_beta_p and mean_base_offset_p before expanding
            if 'districts' not in mean_beta_p.coords or 'districts' not in mean_base_offset_p.coords:
                 raise ValueError("Missing 'districts' coordinate in mean_beta_p or mean_base_offset_p.")
            district_coords_for_expansion = mean_beta_p['districts'].values # Use actual district coords from data

            mean_national_dev_at_date_b = mean_national_dev_at_date.expand_dims(dim={"districts": district_coords_for_expansion}, axis=-1)

            # Check alignment before operation
            if mean_beta_p.dims != mean_national_dev_at_date_b.dims:
                 print(f"DEBUG Latent Support: Aligning dimensions for dynamic adjustment.")
                 # Ensure mean_beta_p and mean_base_offset_p have ('parties_complete', 'districts') order if needed
                 # Assuming mean_beta_p and mean_base_offset_p are (party, district) from .mean()
                 # Assuming mean_national_dev_at_date_b is (party, district) after expansion
                 # If dims mismatch, adjust here. Example:
                 # mean_beta_p = mean_beta_p.transpose('parties_complete', 'districts')
                 pass # Assume dimensions match for now based on previous runs

            mean_dynamic_adjustment = (mean_beta_p - 1) * mean_national_dev_at_date_b
            mean_district_adjustment = mean_base_offset_p + mean_dynamic_adjustment

            # Calculate latent support (using mean values)
            mean_national_trend_at_date_b = mean_national_trend_at_date.expand_dims(dim={"districts": district_coords_for_expansion}, axis=-1)

            # Check alignment before operation
            if mean_national_trend_at_date_b.dims != mean_district_adjustment.dims:
                 print(f"DEBUG Latent Support: Aligning dimensions for final latent support.")
                 # Adjust dimensions if needed, similar to above
                 pass # Assume dimensions match

            mean_latent_district_support = mean_national_trend_at_date_b + mean_district_adjustment

            # Select parties and convert to DataFrame for display
            latent_support_df = mean_latent_district_support.sel(parties_complete=PARTIES_TO_PLOT).to_pandas()
            # Transpose for better readability if many districts
            if len(districts) > 10:
                latent_support_df = latent_support_df.T

            print("Latent support values (logit scale, before softmax):")
            print(latent_support_df.to_string(float_format="%.2f"))

            # Save to CSV - Include date mode in filename
            latent_csv_path = os.path.join(plot_dir, f"mean_latent_support_{cli_args.prediction_date_mode}_{analysis_target_date.date().isoformat()}.csv")
            latent_support_df.to_csv(latent_csv_path, float_format="%.4f")
            print(f"Mean latent support saved to {latent_csv_path}")

            # --- Add Component Breakdown for Specific Party/District --- 
            debug_breakdown_party = 'L'
            debug_breakdown_district = 'Lisboa'
            if debug_breakdown_party in PARTIES_TO_PLOT and debug_breakdown_district in districts:
                 print(f"\n--- Component Breakdown for {debug_breakdown_party} in {debug_breakdown_district} ({analysis_target_date.date()}) ---")
                 try:
                     # Extract mean values for the specific party/district
                     trend_at_date = mean_national_trend_at_date.sel(parties_complete=debug_breakdown_party).item()
                     avg_trend = mean_national_avg_trend_p.sel(parties_complete=debug_breakdown_party).item()
                     base_offset = mean_base_offset_p.sel(parties_complete=debug_breakdown_party, districts=debug_breakdown_district).item()
                     beta = mean_beta_p.sel(parties_complete=debug_breakdown_party, districts=debug_breakdown_district).item()
                     
                     # Calculate derived components
                     nat_dev = trend_at_date - avg_trend
                     dyn_adj = (beta - 1) * nat_dev
                     total_latent_calc = trend_at_date + base_offset + dyn_adj
                     # Get pre-calculated total for comparison
                     total_latent_precalc = mean_latent_district_support.sel(parties_complete=debug_breakdown_party, districts=debug_breakdown_district).item()

                     print(f"  National Trend at Date   : {trend_at_date:.4f}")
                     print(f"  Average National Trend   : {avg_trend:.4f}")
                     print(f"  >> National Deviation    : {nat_dev:.4f}")
                     print(f"  Base Offset (Lisboa)   : {base_offset:.4f}")
                     print(f"  Beta Sensitivity (Lisboa): {beta:.4f}")
                     print(f"  >> Dynamic Adj Term      : {dyn_adj:.4f}  (Beta-1 * NatDev)")
                     print(f"  --------------------------------------")
                     print(f"  Calculated Total Latent: {total_latent_calc:.4f}  (Trend + Base + DynAdj)")
                     print(f"  (Precalculated Total)  : {total_latent_precalc:.4f}") # Should match

                 except KeyError as ke:
                     print(f"    Error extracting components: Missing key {ke}")
                 except Exception as comp_err:
                     print(f"    Error calculating components: {comp_err}")
            else:
                 print(f"\nSkipping component breakdown: Party '{debug_breakdown_party}' or District '{debug_breakdown_district}' not available.")
            # --- End Component Breakdown ---

        except KeyError as e:
            print(f"Error calculating mean latent support: Missing variable {e}")
        except Exception as e:
            print(f"Error calculating mean latent support: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Skipping Mean Latent Support Analysis (District Share Calculation Failed) ---")


    # --- 5. Visualize District Effects --- # Renumbered from 4
    print("\n--- Analyzing District Effects (Base Offset & Beta) ---")
    offset_plot_subdir = os.path.join(plot_dir, "district_offset_plots")
    beta_plot_subdir = os.path.join(plot_dir, "district_beta_plots")
    os.makedirs(offset_plot_subdir, exist_ok=True)
    os.makedirs(beta_plot_subdir, exist_ok=True)

    for district in districts:
        print(f"  Generating district effect plots for: {district}")
        try:
            # --- Base Offset Plot ---
            # Create a dictionary mapping party name to its data for this district
            base_offset_dict = {
                party: idata.posterior['base_offset_p'].sel(districts=district, parties_complete=party)
                for party in PARTIES_TO_PLOT
                if 'base_offset_p' in idata.posterior # Ensure var exists before access
                   and party in idata.posterior['parties_complete'].values # Ensure party exists
                   and district in idata.posterior['districts'].values # Ensure district exists
            }
            # Filter out any empty entries if selection failed
            base_offset_dict = {k: v for k, v in base_offset_dict.items() if v.size > 0}

            if base_offset_dict:
                az.plot_forest(base_offset_dict,
                               # model_names=list(base_offset_dict.keys()), # Explicitly match keys #<<< REMOVED
                               # var_names=['base_offset_p'], # Not needed when passing dict of DataArrays
                               combined=True,
                               hdi_prob=0.94,
                               figsize=(8, 4))
                plt.suptitle(f"District: {district}\nBase Offset (mean & 94% HDI)", y=1.05)
                # Include date mode in filename (though these plots aren't date-specific)
                offset_plot_path = os.path.join(offset_plot_subdir, f"offset_{district.replace(' ','_')}.png")
                plt.savefig(offset_plot_path, bbox_inches='tight')
                plt.close()
            else:
                print(f"    Skipping offset plot for {district} (no valid data found for selected parties).")


            # --- Beta Sensitivity Plot ---
            beta_dict = {
                party: idata.posterior['beta_p'].sel(districts=district, parties_complete=party)
                for party in PARTIES_TO_PLOT
                if 'beta_p' in idata.posterior
                   and party in idata.posterior['parties_complete'].values
                   and district in idata.posterior['districts'].values
            }
            beta_dict = {k: v for k, v in beta_dict.items() if v.size > 0}

            if beta_dict:
                az.plot_forest(beta_dict,
                               # model_names=list(beta_dict.keys()), # <<< REMOVED
                               # var_names=['beta_p'], # Not needed
                               combined=True,
                               hdi_prob=0.94,
                               figsize=(8, 4))
                plt.axvline(1.0, color='grey', linestyle='--', label='Beta = 1')
                plt.legend()
                plt.suptitle(f"District: {district}\nBeta Sensitivity (mean & 94% HDI)", y=1.05)
                # Include date mode in filename (though these plots aren't date-specific)
                beta_plot_path = os.path.join(beta_plot_subdir, f"beta_{district.replace(' ','_')}.png")
                plt.savefig(beta_plot_path, bbox_inches='tight')
                plt.close()
            else:
                 print(f"    Skipping beta plot for {district} (no valid data found for selected parties).")

        except KeyError as e:
             # This might catch missing variables or coordinates during internal selection
             print(f"    Error accessing data for district {district}: {e}")
             continue # Skip to next district
        except Exception as e:
            print(f"    Error generating plots for district {district}: {e}")
            # Print traceback for unexpected errors
            import traceback
            traceback.print_exc()
            continue # Skip to next district

    print(f"District effect plots saved to subdirectories within: {plot_dir}")


    # --- 6. Visualize National Trend --- # Renumbered from 5
    print(f"\n--- Analyzing National Latent Trend around Analysis Date ({analysis_target_date.date()}) ---")
    try:
        # Check if national_trend_pt exists before trying to use it
        if 'national_trend_pt' not in idata.posterior:
            print("Error: 'national_trend_pt' not found in posterior data. Skipping national trend plot.")
        else:
            trend_da = idata.posterior['national_trend_pt'].sel(parties_complete=PARTIES_TO_PLOT)
            window_days = 90
            start_date = analysis_target_date - pd.Timedelta(days=window_days)
            end_date = analysis_target_date + pd.Timedelta(days=window_days)

            # Ensure calendar_time is datetime for comparison
            calendar_time_dt = pd.to_datetime(idata.posterior.coords['calendar_time'].values)
            time_mask = (calendar_time_dt >= start_date) & (calendar_time_dt <= end_date)

            # Check if time_mask has any True values
            if not np.any(time_mask):
                 print(f"Warning: No data found in the time window ({start_date.date()} to {end_date.date()}) for national trend plot.")
            else:
                # Select using the boolean mask
                trend_window = trend_da.sel(calendar_time=time_mask)
                # Get the corresponding time coordinates for plotting
                time_coords_window = calendar_time_dt[time_mask]

                if trend_window.sizes['calendar_time'] == 0: # Double check size after selection
                    print("Warning: Selected trend data window is empty.")
                else:
                    plt.figure(figsize=(12, 6))
                    for party in PARTIES_TO_PLOT:
                        party_trend = trend_window.sel(parties_complete=party)
                        mean_trend = party_trend.mean(dim=['chain', 'draw'])
                        # Access hdi data correctly - check variable name if needed
                        hdi_data = az.hdi(party_trend, hdi_prob=0.94)
                        # Check if the variable name is indeed 'national_trend_pt' inside the hdi result
                        var_name_in_hdi = list(hdi_data.data_vars)[0] # Assume first var is the correct one
                        hdi_values = hdi_data[var_name_in_hdi]

                        plt.plot(time_coords_window, mean_trend, label=f"{party} (Mean)")
                        plt.fill_between(time_coords_window, hdi_values[:,0], hdi_values[:,1], alpha=0.3, label=f"{party} (94% HDI)")

                    plt.axvline(analysis_target_date, color='red', linestyle=':', label=f"Analysis Date ({analysis_target_date.date()})")
                    plt.title(f"National Latent Trend ({window_days} days around Analysis Date)")
                    plt.xlabel("Date")
                    plt.ylabel("Latent Support (Logit Scale)")
                    plt.legend()
                    plt.grid(True, axis='y', linestyle=':')
                    plt.tight_layout()

                    # Include date mode in filename
                    trend_plot_path = os.path.join(plot_dir, f"national_trend_around_{cli_args.prediction_date_mode}_{analysis_target_date.date().isoformat()}.png")
                    plt.savefig(trend_plot_path)
                    plt.close()
                    print(f"National trend plot saved to {trend_plot_path}")

    except KeyError as e:
         print(f"Error: Variable missing for national trend plot: {e}")
    except Exception as e:
        print(f"Error generating national trend plot: {e}")
        import traceback
        traceback.print_exc()

    # --- 7. Summary Text --- # Renumbered from 6
    print("\n--- Summary ---")
    print(f"Analysis for parties: {INVESTIGATE_PARTIES} compared to {BASELINE_PARTY}")
    print(f"Prediction date mode used: {cli_args.prediction_date_mode} ({analysis_target_date.date()})")
    print("\nKey things to check:")
    print("1. Vote Shares Table: Are the mean predicted shares for L/IL unusually high (e.g., >5-10%) in several districts, especially those awarding more seats?")
    print("2. Base Offset Plot: Do L/IL have significantly higher baseline support (mean offset) in many districts compared to the baseline party?")
    print("3. Beta Sensitivity Plot: Are the beta values (mean) for L/IL significantly > 1 in many districts? This would amplify any positive national trend deviation.")
    print("   Are they significantly different from the baseline party's beta values?")
    print("4. National Trend Plot: Was the latent national trend for L/IL estimated to be particularly high or rapidly increasing around the analysis date?")
    print("5. Latent Support Table: Compare relative values *before* softmax. Are L/IL consistently much lower than dominant parties?")
    print("6. Share Distribution Plot: Does the distribution of posterior shares confirm the mean value (e.g., is it truly peaked at zero)?")
    print("\nCombine these observations to understand if high seats are driven by high baseline predictions in districts, high sensitivity to national swings, or an optimistic national trend extrapolation for the specific date analyzed.")

if __name__ == "__main__":
    main() 