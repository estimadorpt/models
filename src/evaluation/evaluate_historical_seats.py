import argparse
import os
import pandas as pd
import numpy as np
import xarray as xr
import json
import traceback
from datetime import datetime

# Use existing project components for loading and processing
from src.main import load_model # Leverage main loading logic
from src.models.elections_facade import ElectionsFacade # To get dataset etc.
from src.data.loaders import load_district_config
from src.processing.electoral_systems import calculate_dhondt
from src.config import DEFAULT_BASELINE_TIMESCALE, DEFAULT_ELECTION_TIMESCALES # Import defaults for load_model


def evaluate_seats(args):
    """
    Main function to evaluate historical seat prediction accuracy using the loaded model.
    Compares model-predicted seats against seats calculated from actual historical vote shares.
    """
    print("--- Starting Historical Seat Evaluation ---")
    load_dir = args.load_dir
    output_dir = args.output_dir if args.output_dir else os.path.join(load_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load Model using the main loading function
        print(f"Loading model from: {load_dir}")
        load_args = argparse.Namespace(
            load_dir=load_dir,
            election_date=None, # Let load_model determine from config or default
            model_type=None, # Let load_model determine from config or default
            baseline_timescale=DEFAULT_BASELINE_TIMESCALE, # Provide defaults
            election_timescale=DEFAULT_ELECTION_TIMESCALES,
            cutoff_date=None,
            baseline_gp_lengthscale=None, baseline_gp_kernel=None,
            cycle_gp_lengthscale=None, cycle_gp_kernel=None, cycle_gp_max_days=None,
            hsgp_m=None, hsgp_c=None, hsgp_m_cycle=None, hsgp_c_cycle=None,
            debug=args.debug # Pass debug flag
        )
        elections_model: ElectionsFacade = load_model(load_args, load_dir, debug=args.debug)

        if elections_model is None or elections_model.trace is None:
            raise RuntimeError(f"Failed to load model or trace from {load_dir}")

        idata = elections_model.trace
        dataset = elections_model.dataset # Access dataset loaded by facade

        # Check if dataset contains the necessary district results
        if not hasattr(dataset, 'results_mult_district') or dataset.results_mult_district.empty:
             raise ValueError("Dataset does not contain 'results_mult_district' needed for actual vote shares.")
        df_actual_results_votes = dataset.results_mult_district
        # Ensure date column is datetime
        if 'election_date' in df_actual_results_votes.columns:
            df_actual_results_votes['election_date'] = pd.to_datetime(df_actual_results_votes['election_date'])
        else:
             raise ValueError("'election_date' column missing from dataset.results_mult_district")

        # Ensure the required variable exists in the posterior
        if "p_district_calendar" not in idata.posterior:
            raise ValueError("Variable 'p_district_calendar' not found in posterior trace. Ensure the model was run with district predictions.")
        posterior_shares_all_time = idata.posterior["p_district_calendar"]

        # <<< Ensure the coordinate is datetime type within the xarray object >>>
        try:
            print("Converting posterior 'calendar_time' coordinate to datetime...")
            posterior_shares_all_time['calendar_time'] = pd.to_datetime(posterior_shares_all_time['calendar_time'].values)
            print("Conversion successful.")
        except Exception as e:
            print(f"Warning: Failed to convert 'calendar_time' coordinate to datetime: {e}")
            # Potentially raise error here if conversion is critical

        # Get coordinate values from the loaded trace
        calendar_times_dt = pd.to_datetime(posterior_shares_all_time['calendar_time'].values)
        districts_model = posterior_shares_all_time['districts'].values.tolist()
        parties_model = posterior_shares_all_time['parties_complete'].values.tolist()
        num_chains = posterior_shares_all_time.sizes['chain']
        num_draws = posterior_shares_all_time.sizes['draw']
        num_samples = num_chains * num_draws

        print(f"Model coordinates - Districts: {len(districts_model)}, Parties: {len(parties_model)}")
        print(f"Posterior samples: {num_samples} ({num_chains} chains x {num_draws} draws)")
        print(f"Parties available in dataset results: {[c for c in df_actual_results_votes.columns if c in parties_model]}")

        # 2. Load District Mandates
        mandates_dict = {}
        if hasattr(dataset, 'district_mandates') and dataset.district_mandates:
             print("Using district mandates from loaded dataset.")
             mandates_dict = dataset.district_mandates # Assuming dataset loads this
        else:
             print(f"Loading district mandates using default loader.")
             mandates_dict = load_district_config() # Use loader
             if not mandates_dict:
                  raise ValueError("Failed to load district mandates using default loader.")
             print(f"Loaded mandates for {len(mandates_dict)} districts.")

        # 3. Find Common Dates
        historical_dates_results = pd.to_datetime(df_actual_results_votes['election_date'].unique())
        common_dates = sorted(list(set(calendar_times_dt) & set(historical_dates_results)))
        if not common_dates:
             print(f"Model calendar times: {calendar_times_dt.min()} to {calendar_times_dt.max()}")
             print(f"Historical result dates: {historical_dates_results.min()} to {historical_dates_results.max()}")
             raise ValueError("No common election dates found between model trace and historical results data in dataset.")
        print(f"Found {len(common_dates)} common historical election dates to evaluate.")

        # 4. Loop through Dates, Samples, Districts and Evaluate
        evaluation_results = []

        for eval_date in common_dates:
            print(f"\nEvaluating election date: {eval_date.strftime('%Y-%m-%d')}")

            # Get posterior vote shares for this date
            try:
                shares_this_date = posterior_shares_all_time.sel(calendar_time=eval_date)
            except KeyError:
                print(f"  Warning: Exact date {eval_date} not in calendar_time coord. Using nearest.")
                shares_this_date = posterior_shares_all_time.sel(calendar_time=eval_date, method='nearest')
                actual_selected_date = pd.Timestamp(shares_this_date['calendar_time'].values).normalize()
                if actual_selected_date != eval_date:
                    print(f"  Selected data for {actual_selected_date.strftime('%Y-%m-%d')} instead.")

            shares_flat = shares_this_date.stack(sample=('chain', 'draw')).transpose('sample', 'districts', 'parties_complete')
            # Get actual results votes/shares for this date from the dataset
            actual_results_this_date_df = df_actual_results_votes[df_actual_results_votes['election_date'] == eval_date].set_index('Circulo')

            for district_model in districts_model:
                print(f"  Processing district: {district_model}...") # Print district start
                if district_model not in mandates_dict:
                    continue
                if district_model not in actual_results_this_date_df.index:
                    continue

                num_mandates = mandates_dict[district_model]
                if num_mandates <= 0:
                     continue

                # A. Calculate ACTUAL seats from historical votes/shares in dataset
                actual_seats_calculated = {}
                actual_data_row = actual_results_this_date_df.loc[[district_model]] # Keep as DataFrame row
                # Check if 'sample_size' column exists for calculating shares from counts
                if 'sample_size' in actual_data_row.columns:
                    actual_total_votes = actual_data_row['sample_size'].iloc[0]
                    if actual_total_votes > 0:
                         actual_shares_dict = {}
                         for party in parties_model:
                             if party in actual_data_row.columns:
                                 actual_shares_dict[party] = actual_data_row[party].iloc[0] / actual_total_votes
                             else:
                                 actual_shares_dict[party] = 0.0 # Assume 0 share if party column missing
                         # Normalize just in case
                         total_share = sum(actual_shares_dict.values())
                         if not np.isclose(total_share, 1.0, atol=1e-4):
                              actual_shares_dict = {p: s / total_share for p, s in actual_shares_dict.items()} if total_share > 0 else actual_shares_dict

                         try:
                             actual_seats_calculated = calculate_dhondt(actual_shares_dict, num_mandates)
                         except Exception as e:
                              print(f"  Error calculating ACTUAL seats for {district_model} on {eval_date.strftime('%Y-%m-%d')}: {e}")
                              actual_seats_calculated = {party: np.nan for party in parties_model} # Mark as NaN if calculation fails
                    else:
                         print(f"  Warning: Actual total votes (sample_size) is {actual_total_votes} for {district_model}. Cannot calculate actual seats.")
                         actual_seats_calculated = {party: 0 for party in parties_model} # Assume 0 seats if no votes
                else:
                    # Assume columns directly represent shares if no sample_size
                    print(f"  Warning: 'sample_size' not found for {district_model}. Assuming columns are shares.")
                    actual_shares_dict = {}
                    for party in parties_model:
                        if party in actual_data_row.columns:
                             actual_shares_dict[party] = actual_data_row[party].iloc[0]
                        else:
                             actual_shares_dict[party] = 0.0
                    # Normalize
                    total_share = sum(actual_shares_dict.values())
                    if not np.isclose(total_share, 1.0, atol=1e-4):
                         actual_shares_dict = {p: s / total_share for p, s in actual_shares_dict.items()} if total_share > 0 else actual_shares_dict
                    try:
                         actual_seats_calculated = calculate_dhondt(actual_shares_dict, num_mandates)
                    except Exception as e:
                         print(f"  Error calculating ACTUAL seats from shares for {district_model}: {e}")
                         actual_seats_calculated = {party: np.nan for party in parties_model}

                # B. Calculate PREDICTED seats from model posterior shares
                pred_shares_district = shares_flat.sel(districts=district_model).values
                pred_seats_district_samples = np.zeros((num_samples, len(parties_model)), dtype=int)

                for i_sample in range(num_samples):
                    # Print progress every 1000 samples
                    if (i_sample + 1) % 1000 == 0:
                        print(f"    ... sample {i_sample + 1}/{num_samples}")

                    sample_shares = pred_shares_district[i_sample, :]
                    sample_shares = np.maximum(sample_shares, 0)
                    share_sum = sample_shares.sum()
                    if not np.isclose(share_sum, 1.0, atol=1e-4):
                         sample_shares = sample_shares / share_sum if share_sum > 0 else sample_shares
                    dhondt_input_shares = {party: sample_shares[j] for j, party in enumerate(parties_model)}
                    try:
                         allocated_seats_dict = calculate_dhondt(dhondt_input_shares, num_mandates)
                         for j, party in enumerate(parties_model):
                              pred_seats_district_samples[i_sample, j] = allocated_seats_dict.get(party, 0)
                    except Exception as dhondt_err:
                         # print(f"    Error during D'Hondt prediction for sample {i_sample}, district {district_model}: {dhondt_err}")
                         pred_seats_district_samples[i_sample, :] = -1 # Mark sample as invalid

                # Aggregate predicted seats (mean) across valid samples
                valid_samples_mask = ~np.any(pred_seats_district_samples == -1, axis=1)
                if np.sum(valid_samples_mask) == 0:
                    # print(f"  Error: No valid D'Hondt predictions for {district_model} on {eval_date.strftime('%Y-%m-%d')}.")
                    mean_pred_seats = {party: np.nan for party in parties_model}
                else:
                    mean_pred_seats_array = np.mean(pred_seats_district_samples[valid_samples_mask, :], axis=0)
                    mean_pred_seats = {party: mean_pred_seats_array[j] for j, party in enumerate(parties_model)}

                # C. Store results comparing predicted vs calculated actual
                for party in parties_model:
                    actual = actual_seats_calculated.get(party, 0) # Get calculated actual seats
                    # Ensure actual is numeric before appending
                    actual = int(actual) if pd.notna(actual) else 0

                    pred_mean = mean_pred_seats.get(party, np.nan)
                    # Calculate error only if both actual and predicted are valid numbers
                    if pd.notna(actual) and pd.notna(pred_mean):
                         error = pred_mean - actual
                    else:
                         error = np.nan

                    evaluation_results.append({
                        "election_date": eval_date.strftime('%Y-%m-%d'),
                        "district": district_model,
                        "party": party,
                        "actual_seats": actual,
                        "predicted_seats_mean": pred_mean,
                        "prediction_error": error
                    })

        # 5. Summarize and Save Results
        if not evaluation_results:
             print("\nNo evaluation results generated.")
             return

        df_eval = pd.DataFrame(evaluation_results)
        output_file = os.path.join(output_dir, "historical_seat_evaluation.csv")
        df_eval.to_csv(output_file, index=False, float_format='%.3f')
        print(f"\nSaved detailed evaluation results to: {output_file}")

        df_valid = df_eval.dropna(subset=['prediction_error', 'actual_seats'])
        if not df_valid.empty:
            # Ensure actual_seats is numeric for calculations
            df_valid['actual_seats'] = pd.to_numeric(df_valid['actual_seats'])
            overall_mae = np.mean(np.abs(df_valid['prediction_error']))
            print(f"\nOverall Mean Absolute Error (Seats): {overall_mae:.3f}")

            print("\nError Summary (Predicted - Actual Seats):")
            # Filter out rows where actual seats are 0 for the summary table, focus on parties that won seats
            df_errors = df_valid[(np.abs(df_valid['prediction_error']) > 0.01) & (df_valid['actual_seats'] > 0)]
            df_errors = df_errors.sort_values(by='prediction_error', key=abs, ascending=False)

            # Aggregate errors by party (considering only non-zero actual seats)
            party_error_summary = df_errors.groupby('party')['prediction_error'].agg(['sum', 'mean', 'count']).sort_values('sum', key=abs, ascending=False)
            print("\nTotal Seat Prediction Error by Party (where party won seats):")
            print(party_error_summary.to_string(float_format='%.2f'))

            print("\nLargest Individual Errors (where party won seats):")
            print(df_errors.to_string(max_rows=50, float_format='%.2f'))
        else:
             print("\nCould not calculate overall MAE (no valid predictions/actuals).")

        print("\n--- Historical Seat Evaluation Finished ---")

    except FileNotFoundError as e:
        print(f"ERROR: Input file not found. {e}")
    except ValueError as e:
        print(f"ERROR: Data validation or configuration error. {e}")
    except RuntimeError as e:
         print(f"ERROR: Model loading or execution error. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate historical seat prediction accuracy of a trained model.")
    parser.add_argument("--load-dir", required=True, help="Directory containing the saved model trace and config.")
    parser.add_argument("--output-dir", default=None, help="Directory to save the evaluation results CSV. Defaults to load_dir/evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")

    args = parser.parse_args()

    evaluate_seats(args) 