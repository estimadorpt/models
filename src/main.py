import os
import argparse
import arviz as az
import requests
import matplotlib.pyplot as plt
import pandas as pd
import time
import traceback
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import pymc as pm
import json
import numbers
import re
from pprint import pprint
import pytensor
pytensor.config.cxx = '/usr/bin/clang++'
from src.models.elections_facade import ElectionsFacade
from src.models.static_baseline_election_model import StaticBaselineElectionModel
from src.models.dynamic_gp_election_model import DynamicGPElectionModel

# # <<< Print loaded class file path >>>
# print(f"DEBUG MAIN: ElectionsFacade class loaded from: {ElectionsFacade.__file__}")
# # <<< End print >>>

from src.visualization.plots import (
    plot_election_data, 
    plot_latent_popularity_vs_polls, 
    plot_latent_component_contributions, 
    plot_recent_polls,
    plot_house_effects_heatmap,
    plot_reliability_diagram,
    plot_latent_trend_since_last_election,
    plot_forecasted_election_distribution,
    plot_seat_distribution_histograms,
    plot_poll_bias_forest
)
from src.config import DEFAULT_BASELINE_TIMESCALE, DEFAULT_ELECTION_TIMESCALES
from src.data.dataset import ElectionDataset
from src.data.loaders import load_election_results, load_district_config
from src.processing.electoral_systems import calculate_dhondt
from src.processing.seat_prediction import simulate_seat_allocation

def get_model_class(model_type_str: str):
    if model_type_str == "static":
        return StaticBaselineElectionModel
    elif model_type_str == "dynamic_gp":
        return DynamicGPElectionModel
    else:
        raise ValueError(f"Unknown model type: {model_type_str}")

def fit_model(args):
    """Fit a new model with the specified parameters"""
    try:
        # --- Select Model Class ---
        model_class_to_use = get_model_class(args.model_type)
        print(f"Using model type: {args.model_type}")
        # --- End Select Model Class ---
        
        if args.debug:
            print(f"Fitting model for election date {args.election_date}")
            if args.model_type == "static":
                 print(f"Using baseline timescales: {args.baseline_timescale}")
                 print(f"Using election timescales: {args.election_timescale}")
            elif args.model_type == "dynamic_gp":
                 # Print new args for dynamic_gp
                 print(f"Using Baseline GP lengthscale: {args.baseline_gp_lengthscale}")
                 print(f"Using Baseline GP kernel: {args.baseline_gp_kernel}")
                 print(f"Using Cycle GP lengthscale: {args.cycle_gp_lengthscale}")
                 print(f"Using Cycle GP kernel: {args.cycle_gp_kernel}")
                 print(f"Using Cycle GP max days: {args.cycle_gp_max_days}")
                 print(f"Using HSGP m (baseline): {args.hsgp_m}")
                 print(f"Using HSGP c (baseline): {args.hsgp_c}")
                 print(f"Using HSGP m (cycle): {args.hsgp_m_cycle}")
                 print(f"Using HSGP c (cycle): {args.hsgp_c_cycle}")

        # --- Determine Output Directory --- #
        # Check if the user provided a specific output dir or if it's the default
        # We need the actual default value from argparse setup
        # Assuming the default is 'outputs/' based on argparse definition below
        default_output_dir = "outputs/"
        if args.output_dir == default_output_dir:
            # Default case: Create a timestamped subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{args.model_type}_run_{timestamp}" # Include model type for clarity
            output_dir = os.path.join(default_output_dir.rstrip('/'), run_name)
            print(f"No specific output directory provided. Creating timestamped directory: {output_dir}")
        else:
            # User provided a specific path: Use it directly
            output_dir = args.output_dir
            print(f"Using specified output directory: {output_dir}")
        # --- End Determine Output Directory --- #

        # Make sure the final output directory exists
        os.makedirs(output_dir, exist_ok=True)

        if args.debug:
            print(f"Using output directory: {output_dir}")
        
        # --- Prepare Model Kwargs ---
        model_kwargs = {
             "debug": args.debug
        }
        # Store base config shared by multiple modes
        config_to_save = {
            "model_type": args.model_type,
            "election_date": args.election_date,
            "cutoff_date": args.cutoff_date,
            "debug": args.debug,
            # Add sampling params for reproducibility info
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "target_accept": args.target_accept,
            "seed": args.seed,
        }

        if args.model_type == "static":
             model_kwargs["baseline_lengthscale"] = args.baseline_timescale
             model_kwargs["election_lengthscale"] = args.election_timescale
             config_to_save["baseline_timescale"] = args.baseline_timescale
             config_to_save["election_timescale"] = args.election_timescale

        elif args.model_type == "dynamic_gp":
             model_kwargs["baseline_gp_lengthscale"] = args.baseline_gp_lengthscale
             model_kwargs["baseline_gp_kernel"] = args.baseline_gp_kernel
             model_kwargs["cycle_gp_lengthscale"] = args.cycle_gp_lengthscale
             model_kwargs["cycle_gp_kernel"] = args.cycle_gp_kernel
             model_kwargs["cycle_gp_max_days"] = args.cycle_gp_max_days
             model_kwargs["hsgp_m"] = args.hsgp_m
             model_kwargs["hsgp_c"] = args.hsgp_c
             model_kwargs["hsgp_m_cycle"] = args.hsgp_m_cycle
             model_kwargs["hsgp_c_cycle"] = args.hsgp_c_cycle
             config_to_save["baseline_gp_lengthscale"] = args.baseline_gp_lengthscale
             config_to_save["baseline_gp_kernel"] = args.baseline_gp_kernel
             config_to_save["cycle_gp_lengthscale"] = args.cycle_gp_lengthscale
             config_to_save["cycle_gp_kernel"] = args.cycle_gp_kernel
             config_to_save["cycle_gp_max_days"] = args.cycle_gp_max_days
             config_to_save["hsgp_m"] = args.hsgp_m
             config_to_save["hsgp_c"] = args.hsgp_c
             config_to_save["hsgp_m_cycle"] = args.hsgp_m_cycle
             config_to_save["hsgp_c_cycle"] = args.hsgp_c_cycle
        # --- End Prepare Model Kwargs ---
        
        # Initialize the elections model
        elections_model = ElectionsFacade(
            election_date=args.election_date,
            model_class=model_class_to_use, # Use selected class
            test_cutoff=pd.Timedelta(args.cutoff_date) if args.cutoff_date else None,
            # Pass model-specific kwargs using dictionary unpacking
            **model_kwargs 
        )
        
        # Access the specific model instance for model-specific details if needed
        model_instance = elections_model.model_instance
        
        # Start time measurement
        start_time = time.time()
        
        # Run inference
        prior, trace, posterior = elections_model.run_inference(
            draws=args.draws, 
            tune=args.tune, 
            target_accept=args.target_accept, 
            max_treedepth=12 # Increase max_treedepth
        )
        
        # Calculate elapsed time
        end_time = time.time()
        fitting_duration = end_time - start_time
        print(f"Model fitting completed in {fitting_duration:.2f} seconds")
        
        # Send notification if requested
        print("DEBUG FIT: Checking for notification...")
        if args.notify:
            print("DEBUG FIT: Sending notification...")
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Model fit completed in {fitting_duration:.2f} seconds".encode(encoding='utf-8'))
        
        # Generate diagnostic plots
        print("DEBUG FIT: Checking if trace exists for diagnostics...")
        if elections_model.trace is not None:
            diag_plot_dir = os.path.join(output_dir, "diagnostics")
            print(f"DEBUG FIT: Creating diagnostic plot directory: {diag_plot_dir}...")
            os.makedirs(diag_plot_dir, exist_ok=True)
            print("DEBUG FIT: Calling generate_diagnostic_plots...")
            elections_model.generate_diagnostic_plots(diag_plot_dir)
            print("DEBUG FIT: Finished generate_diagnostic_plots.")

            # --- Call model-specific swing plots ---
            if isinstance(model_instance, DynamicGPElectionModel) and hasattr(model_instance, 'generate_swing_diagnostic_plots'):
                 try:
                      print("DEBUG FIT: Calling generate_swing_diagnostic_plots...")
                      model_instance.generate_swing_diagnostic_plots(elections_model.trace, diag_plot_dir)
                      print("DEBUG FIT: Finished generate_swing_diagnostic_plots.")
                 except Exception as swing_plot_err:
                      print(f"Warning: Failed to generate swing diagnostic plots: {swing_plot_err}")
            # --- End model-specific swing plots ---
        else:
            print("Skipping diagnostic plot generation as trace is not available.")
        
        # --- Calculate and Save Fit Metrics ---
        print("DEBUG FIT: Checking if trace and calculate_fit_metrics exist...")
        metrics_dict = {}
        if elections_model.trace is not None and hasattr(model_instance, 'calculate_fit_metrics'):
            print("DEBUG FIT: Preparing to calculate fit metrics...")
            try:
                print("\nCalculating fit metrics...")
                # Ensure the trace object is available in the model instance if needed by methods
                if not hasattr(model_instance, 'trace') or model_instance.trace is None:
                     model_instance.trace = elections_model.trace # Assign trace if needed
                
                # Capture both metrics and the potentially updated idata
                metrics_dict, updated_idata = model_instance.calculate_fit_metrics(elections_model.trace)
                
                # Update the facade's trace object with the potentially updated one
                if updated_idata is not None:
                     elections_model.trace = updated_idata
                     print("Updated facade trace with results from calculate_fit_metrics.")
                
                print("Fit Metrics:")
                for key, value in metrics_dict.items():
                    # Check if value is a number (int, float, numpy number)
                    if isinstance(value, numbers.Number):
                        # Handle potential NaN values
                        if np.isnan(value):
                             print(f"  {key}: NaN")
                        else:
                             print(f"  {key}: {value:.4f}")
                    # Check if it's one of the calibration dictionaries
                    elif isinstance(value, dict) and key.endswith('_calibration'):
                         print(f"  {key}: [Calibration Data - see plots]")
                    # Otherwise, print the value as a string
                    else:
                        print(f"  {key}: {value}")
                
                metrics_path = os.path.join(output_dir, "fit_metrics.json")
                # --- Convert numpy arrays in calibration dicts to lists AND handle NaN for JSON ---
                metrics_serializable = {}
                for key, value in metrics_dict.items():
                    if isinstance(value, dict) and key.endswith('_calibration'):
                         calib_serializable = {}
                         for cal_key, cal_value in value.items():
                             # Check for NaN in calibration values (though less likely here)
                             if isinstance(cal_value, float) and np.isnan(cal_value):
                                 calib_serializable[cal_key] = None
                             elif isinstance(cal_value, np.ndarray):
                                 # Replace NaN within numpy arrays before converting to list
                                 calib_serializable[cal_key] = np.where(np.isnan(cal_value), None, cal_value).tolist()
                             else:
                                 calib_serializable[cal_key] = cal_value
                         metrics_serializable[key] = calib_serializable
                    # Explicitly check top-level values for NaN
                    elif isinstance(value, float) and np.isnan(value):
                        metrics_serializable[key] = None
                    elif isinstance(value, np.number) and np.isnan(value): # Catch numpy float types
                         metrics_serializable[key] = None
                    else:
                         metrics_serializable[key] = value
                # --- End conversion ---
                
                with open(metrics_path, 'w') as f:
                    # Save the serializable version
                    json.dump(metrics_serializable, f, indent=4) 
                print(f"Fit metrics saved to {metrics_path}")

                # --- Plot Calibration --- #
                viz_dir = os.path.join(output_dir, "visualizations") # Use same dir as other plots
                os.makedirs(viz_dir, exist_ok=True)
                if "poll_calibration" in metrics_dict: # Use original metrics dict for plotting
                     plot_reliability_diagram(
                         metrics_dict["poll_calibration"],
                         title="Poll Calibration",
                         filename=os.path.join(viz_dir, "calibration_polls.png")
                     )
                if "result_district_calibration" in metrics_dict: # Check for district calibration key
                     plot_reliability_diagram(
                         metrics_dict["result_district_calibration"],
                         title="District Result Calibration",
                         filename=os.path.join(viz_dir, "calibration_results_district.png")
                     )
                # --- End Plot Calibration --- #

            except Exception as metrics_err:
                print(f"ERROR during fit metrics calculation or saving: {metrics_err}")
                # Optional: Add traceback print here if needed
                # import traceback
                # traceback.print_exc()
        else:
            print("DEBUG FIT: Skipping fit metrics calculation (trace or method missing).")

        # Save model configuration
        print("DEBUG FIT: Saving model configuration...")
        config_path = os.path.join(output_dir, "model_config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            print(f"Model config saved to {config_path}")
        except Exception as config_save_err:
            print(f"ERROR saving model config: {config_save_err}")
        
        # Save the trace and model - this now returns True/False
        save_successful = elections_model.save_inference_results(output_dir)
        
        # --- RE-ADD and ADAPT automatic 'latest' symlink creation block ---
        # Create/update the 'latest' symlink only if saving was successful
        if save_successful:
            # Determine the parent directory to place the 'latest' link
            base_output_dir = os.path.dirname(output_dir)
            # Handle case where output_dir might be the root (e.g., './my_run') or just a name
            if not base_output_dir:
                 base_output_dir = '.' # Assume current directory if no parent path given

            latest_link_path = os.path.join(base_output_dir, "latest")
            target_path_absolute = os.path.abspath(output_dir)

            print(f"Updating symbolic link '{latest_link_path}' to point to '{target_path_absolute}'")

            try:
                # Remove existing link/file if it exists
                if os.path.islink(latest_link_path) or os.path.exists(latest_link_path):
                    os.remove(latest_link_path)

                # Create the new symlink
                # Check if the target is actually a directory (should be)
                target_is_dir = os.path.isdir(target_path_absolute)
                os.symlink(target_path_absolute, latest_link_path, target_is_directory=target_is_dir)
                print(f"Symbolic link 'latest' updated successfully.")
            except Exception as symlink_err:
                print(f"ERROR updating 'latest' symlink: {symlink_err}")
        else:
            print("Skipping 'latest' symlink creation as no inference results were saved.")
        # --- End RE-ADD symlink logic ---

        print(f"Training process complete for {args.election_date}.")
        return elections_model
        
    except Exception as e:
        print(f"ERROR during model fitting: {e}")
        # Ensure traceback is printed on outer error
        if args.notify:
            try:
                requests.post("https://ntfy.sh/bc-estimador",
                    data=f"Model fit FAILED: {e}".encode(encoding='utf-8'))
            except Exception as notify_err:
                print(f"Failed to send error notification: {notify_err}")
        if args.debug:
             import traceback
             traceback.print_exc()
        return None

def load_model(args, directory, election_date=None, baseline_timescales=None, election_timescales=None, debug=False):
    """
    Load a saved model
    
    Parameters:
    -----------
    directory : str
        Directory where the model is saved
    election_date : str, optional
        Election date context for loading dataset.
    baseline_timescales : list, optional
        Baseline timescales used for the model.
    election_timescales : list, optional
        Election timescales used for the model.
    debug : bool, optional
        Enable debug output.
    """
    try:
        if debug:
            print(f"Loading model from {directory}")
        
        # --- Load Configuration --- 
        loaded_config = {}
        config_path = os.path.join(directory, "model_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                print(f"Loaded configuration from {config_path}")
                if debug:
                    print(f"Loaded config: {loaded_config}")
            except Exception as config_err:
                print(f"Warning: Failed to load model configuration from {config_path}: {config_err}")
        else:
            print(f"Warning: model_config.json not found in {directory}. Using defaults or command-line args.")
        # --- End Load Configuration ---

        # --- Determine Configuration Priority --- 
        # Priority: Command-line args > Loaded config > Defaults
        final_election_date = args.election_date if args.election_date else loaded_config.get('election_date', '2026-01-01')
        # Handle timescale args (which might be defaults from argparse)
        # If the command-line arg is different from the argparse default, use it. Otherwise, check loaded config.
        bs_timescale_arg = args.baseline_timescale if args.baseline_timescale != DEFAULT_BASELINE_TIMESCALE else None
        el_timescale_arg = args.election_timescale if args.election_timescale != DEFAULT_ELECTION_TIMESCALES else None

        final_baseline_timescales = bs_timescale_arg if bs_timescale_arg is not None else loaded_config.get('baseline_timescales', DEFAULT_BASELINE_TIMESCALE)
        final_election_timescales = el_timescale_arg if el_timescale_arg is not None else loaded_config.get('election_timescales', DEFAULT_ELECTION_TIMESCALES)

        final_cutoff_date = args.cutoff_date if args.cutoff_date else loaded_config.get('cutoff_date', None)

        # Ensure timescales are lists
        if not isinstance(final_baseline_timescales, list):
            final_baseline_timescales = [final_baseline_timescales]
        if not isinstance(final_election_timescales, list):
            final_election_timescales = [final_election_timescales]

        # --- Determine Model Type and Class --- #
        # Priority: Command-line args > Loaded config
        # Use the literal default value "static" for comparison
        # final_model_type = args.model_type if args.model_type != "static" else loaded_config.get('model_type', None)
        # if not final_model_type:
        #     print("Warning: Model type not found in args or config, defaulting to 'static'.")
        #     final_model_type = 'static' # Default if still not found

        # Corrected logic:
        loaded_type = loaded_config.get('model_type', None) # Get type from config first
        # Use command-line arg only if it's different from the default 'static'
        if hasattr(args, 'model_type') and args.model_type is not None and args.model_type != "static":
             final_model_type = args.model_type
             if debug: print(f"DEBUG LOAD: Using model_type '{final_model_type}' from command-line args.")
        elif loaded_type: # If command-line wasn't overriding, use loaded config type
             final_model_type = loaded_type
             if debug: print(f"DEBUG LOAD: Using model_type '{final_model_type}' from loaded config.")
        else: # Otherwise, default to static
             final_model_type = 'static'
             if debug: print("DEBUG LOAD: Using default model_type 'static'.")

        print(f"DEBUG load_model: Determined final_model_type = {final_model_type}")
        try:
            model_class_to_load = get_model_class(final_model_type)
        except ValueError as e:
            print(f"Error determining model class: {e}. Falling back to StaticBaselineElectionModel.")
            model_class_to_load = StaticBaselineElectionModel
            final_model_type = 'static' # Ensure consistency
        # --- End Determine Model Type --- #

        # --- Prepare Model Kwargs (based on final_model_type) --- #
        model_kwargs = { "debug": debug }
        if final_model_type == "static":
            # Ensure timescales are lists
            if not isinstance(final_baseline_timescales, list): final_baseline_timescales = [final_baseline_timescales]
            if not isinstance(final_election_timescales, list): final_election_timescales = [final_election_timescales]
            model_kwargs["baseline_lengthscale"] = final_baseline_timescales
            model_kwargs["election_lengthscale"] = final_election_timescales
        elif final_model_type == "dynamic_gp":
            # Helper to get value: arg -> config -> default (None here, handled by model init)
            def get_gp_param(arg_name, config_key, default_val=None):
                arg_val = getattr(args, arg_name, None)
                # Check if arg_val is the default argparse value (might need specific check)
                # For now, assume if arg_val is not None, it was specified or is the default we want
                if arg_val is not None:
                    return arg_val
                return loaded_config.get(config_key, default_val)

            # Get all dynamic_gp params using helper or direct access
            model_kwargs["baseline_gp_lengthscale"] = get_gp_param("baseline_gp_lengthscale", "baseline_gp_lengthscale")
            model_kwargs["baseline_gp_kernel"] = get_gp_param("baseline_gp_kernel", "baseline_gp_kernel")
            model_kwargs["cycle_gp_lengthscale"] = get_gp_param("cycle_gp_lengthscale", "cycle_gp_lengthscale")
            model_kwargs["cycle_gp_kernel"] = get_gp_param("cycle_gp_kernel", "cycle_gp_kernel")
            model_kwargs["cycle_gp_max_days"] = get_gp_param("cycle_gp_max_days", "cycle_gp_max_days")
            model_kwargs["hsgp_m"] = get_gp_param("hsgp_m", "hsgp_m")
            model_kwargs["hsgp_c"] = get_gp_param("hsgp_c", "hsgp_c")
            model_kwargs["hsgp_m_cycle"] = get_gp_param("hsgp_m_cycle", "hsgp_m_cycle")
            model_kwargs["hsgp_c_cycle"] = get_gp_param("hsgp_c_cycle", "hsgp_c_cycle")

            # Remove None values so model defaults take over if nothing was specified/loaded
            model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        # --- End Prepare Model Kwargs --- #

        if debug:
            print(f"Initializing Facade with -> election_date: {final_election_date}")
            print(f"Initializing Facade with -> model_type: {final_model_type}")
            print(f"Initializing Facade with -> cutoff_date: {final_cutoff_date}")
            print(f"Initializing Facade with -> model_kwargs: {model_kwargs}")

        # <<< Add debug print here >>>
        print(f"DEBUG load_model: Instantiating facade with model_class = {model_class_to_load.__name__}")

        elections_model = ElectionsFacade(
            election_date=final_election_date,
            model_class=model_class_to_load,
            test_cutoff=pd.Timedelta(final_cutoff_date) if final_cutoff_date else None,
            **model_kwargs # Pass determined kwargs
        )
        
        # <<< Debug Check AFTER Facade Instantiation >>>
        print(f"DEBUG LOAD: Instantiated facade type: {type(elections_model)}")
        if elections_model:
            print(f"DEBUG LOAD: Facade has get_latent_popularity right after init? {hasattr(elections_model, 'get_latent_popularity')}")
            # <<< Explicitly set election_date on the model instance AFTER facade init >>>
            if hasattr(elections_model, 'model_instance') and elections_model.model_instance is not None:
                 if hasattr(elections_model.model_instance, 'election_date'): # Check if attr exists
                      print(f"DEBUG LOAD: Setting election_date '{final_election_date}' on model instance.")
                      elections_model.model_instance.election_date = final_election_date
                 else:
                      print("DEBUG LOAD: Warning - model instance does not have 'election_date' attribute to set.")
        # <<< End Debug Check >>>
        
        # Load the saved trace
        elections_model.load_inference_results(directory)
        
        # --- Debug loaded idata --- 
        if elections_model.trace is not None and hasattr(elections_model.trace, 'coords') and elections_model.trace.coords is not None:
            if 'calendar_time' in elections_model.trace.coords:
                 loaded_cal_time = elections_model.trace.coords['calendar_time'].values
                 print(f"DEBUG LOAD: Type of loaded calendar_time: {type(loaded_cal_time)}")
                 if hasattr(loaded_cal_time, 'dtype'): print(f"DEBUG LOAD: Dtype of loaded calendar_time: {loaded_cal_time.dtype}")
                 try:
                      print(f"DEBUG LOAD: Last 5 loaded calendar_time coords: {loaded_cal_time[-5:]}")
                 except Exception as e:
                      print(f"DEBUG LOAD: Error printing loaded coords: {e}")
            else:
                 print("DEBUG LOAD: 'calendar_time' not found in loaded coords.")
        else:
             print("DEBUG LOAD: No trace or coords found after loading.")
        # --- End debug --- 
        
        # Rebuild the model structure (necessary for posterior predictive checks etc.)
        # print("Rebuilding model structure...")
        # elections_model.build_model() # <<< COMMENTED OUT: Should rely on loaded idata coords for prediction/viz
        # print("Model structure rebuilt.")
        
        return elections_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        if debug:
            traceback.print_exc()
        return None

def visualize(args):
    """Visualize the results of a trained model"""
    try:
        print(f"Visualizing model from {args.load_dir}")
        
        # Load the model using the modified load_model function signature
        elections_model = load_model(args, args.load_dir, debug=args.debug)
        
        if elections_model is None:
            print("Exiting visualization due to loading error.")
            return
        
        # Rebuild the underlying PyMC model structure after loading trace/config
        # We don't need to rebuild here if plotting functions only need the trace and dataset?
        # print("Rebuilding model structure...")
        # elections_model.build_model()
        # print("Model structure rebuilt successfully.")

        print("\nGenerating historical model diagnostics and visualization plots...")
        viz_dir = os.path.join(args.load_dir, "visualizations") # Define viz_dir based on load_dir
        os.makedirs(viz_dir, exist_ok=True)

        # Pass the facade instance (elections_model) directly to plotting functions
        # These use the posterior trace directly
        # Ensure the plotting functions use elections_model.trace and elections_model.dataset
        plot_latent_popularity_vs_polls(elections_model, viz_dir, include_target_date=True)
        plot_latent_component_contributions(elections_model, viz_dir)
        plot_recent_polls(elections_model, viz_dir)
        plot_house_effects_heatmap(elections_model, viz_dir)
        plot_poll_bias_forest(elections_model, viz_dir)

        print(f"Historical visualizations saved to {viz_dir}")
        
        # --- Generate Forecast Distribution Plot ---
        # print("\nGenerating election outcome forecast distribution...")
        # plot_forecasted_election_distribution(elections_model, args.output_dir)
        # --- End Forecast Plot Generation ---
        
        print(f"\nVisualizations generation step complete.") # Adjusted print statement
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        if args.debug:
            traceback.print_exc()
        return None

def cross_validate(args):
    """Perform cross-validation on historical elections"""
    try:
        if args.debug:
            print("Starting cross-validation...")
        
        # Create cross-validation directory
        cv_dir = os.path.join(args.output_dir, "cross_validation")
        os.makedirs(cv_dir, exist_ok=True)
        
        # Get all historical elections from the dataset
        all_elections = ElectionDataset.historical_election_dates
        if args.debug:
            print(f"Found {len(all_elections)} historical elections: {all_elections}")
        
        # Store results for each election
        cv_results = []
        
        # Process each election
        for election_date in all_elections:
            print(f"\nCross-validating election: {election_date}")
            
            # Create directory for this election's results
            election_dir = os.path.join(cv_dir, election_date.replace("-", ""))
            os.makedirs(election_dir, exist_ok=True)
            
            # Initialize model for this election
            elections_model = ElectionsFacade(
                election_date=election_date,
                model_class=StaticBaselineElectionModel,
                baseline_timescales=args.baseline_timescales,
                election_timescales=args.election_timescales,
                debug=args.debug
            )
            
            # Fit the model
            try:
                elections_model.run_inference(
                    draws=args.draws,
                    tune=args.tune,
                    target_accept=0.9
                )
                
                # Save the inference results
                elections_model.save_inference_results(election_dir)
                
                # Store results
                cv_results.append({
                    'election_date': election_date,
                    'status': 'completed'
                })
            except Exception as e:
                print(f"Error processing election {election_date}: {e}")
                import traceback
                traceback.print_exc()
                
                # Add a placeholder result
                cv_results.append({
                    'election_date': election_date,
                    'error': str(e),
                    'status': 'failed'
                })
                
            # Save plots - Comment out the call
            if not args.fast:
                # save_plots(elections_model, election_dir)
                print(f"Plot generation skipped for {election_date} (plots commented out).")
        
        # Summarize cross-validation results
        if cv_results:
            cv_df = pd.DataFrame(cv_results)
            print("\nCROSS-VALIDATION SUMMARY")
            print("="*50)
            print(cv_df)
            print("="*50)
            
            # Save results to CSV
            cv_df.to_csv(os.path.join(cv_dir, "cross_validation_results.csv"), index=False)
        
        return cv_results
    
    except Exception as e:
        print(f"Error in cross-validation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Error in cross-validation: {e}".encode(encoding='utf-8'))
        return []

def visualize_data(args):
    """Visualize the raw input poll and election data."""
    try:
        print("Visualizing raw input data...")
        if not args.election_date:
            print("Warning: --election-date not specified. Using a default future date for dataset context.")
            # Use a sensible default, e.g., start of next year if not provided
            election_date = (datetime.now().replace(month=1, day=1) + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
            print(f"Using default date: {election_date}")
        else:
            election_date = args.election_date
            print(f"Using specified election date context: {election_date}")

        print(f"Using baseline timescales: {args.baseline_timescale}")
        print(f"Using election timescales: {args.election_timescale}")

        # Create the dataset instance
        # Note: We might not need test_cutoff here, as we're visualizing all data
        dataset = ElectionDataset(
            election_date=election_date,
            baseline_timescales=args.baseline_timescale,
            election_timescales=args.election_timescale,
            test_cutoff=None # Visualize all data by default
        )

        # Call the plotting function
        plot_election_data(dataset, args.output_dir)

        print("Data visualization complete.")

    except Exception as e:
        print(f"Error generating data visualizations: {e}")
        if args.debug:
            traceback.print_exc()
        return None

def diagnose_model(args):
    """Load a saved model and generate diagnostic plots."""
    try:
        if not args.load_dir or not os.path.exists(args.load_dir):
            raise ValueError(f"--load-dir must be provided and exist for diagnose mode. Path: {args.load_dir}")
            
        model_dir = os.path.abspath(args.load_dir)
        if args.debug:
            print(f"Diagnosing model from directory: {model_dir}")

        # Load the model and its trace
        elections_model = load_model(args, model_dir, debug=args.debug)

        if elections_model is None:
            print(f"Failed to load model from {model_dir}")
            return
            
        if elections_model.trace is None:
            print(f"No trace found in {model_dir}. Cannot generate diagnostics.")
            return

        # Define output directory for plots within the loaded model's directory
        diag_plot_dir = os.path.join(model_dir, "diagnostics")
        os.makedirs(diag_plot_dir, exist_ok=True)
        
        print(f"Generating diagnostic plots for trace loaded from {model_dir}")
        elections_model.generate_diagnostic_plots(diag_plot_dir)

        # --- Call model-specific swing plots ---
        model_instance = elections_model.model_instance # Get the instance
        if isinstance(model_instance, DynamicGPElectionModel) and hasattr(model_instance, 'generate_swing_diagnostic_plots'):
             try:
                  print("DEBUG DIAGNOSE: Calling generate_swing_diagnostic_plots...")
                  model_instance.generate_swing_diagnostic_plots(elections_model.trace, diag_plot_dir)
                  print("DEBUG DIAGNOSE: Finished generate_swing_diagnostic_plots.")
             except Exception as swing_plot_err:
                  print(f"Warning: Failed to generate swing diagnostic plots: {swing_plot_err}")
        # --- End model-specific swing plots ---

        print(f"Diagnostic plots saved to {diag_plot_dir}")

        # --- Calculate and Display Fit Metrics (from loaded model) ---
        metrics_dict = {}
        if elections_model.trace is not None and hasattr(elections_model.model_instance, 'calculate_fit_metrics'):
            try:
                print("\nCalculating fit metrics for the loaded model...")
                # Ensure the trace object is available in the model instance
                if not hasattr(elections_model.model_instance, 'trace') or elections_model.model_instance.trace is None:
                     elections_model.model_instance.trace = elections_model.trace # Assign trace if needed
                metrics_dict = elections_model.model_instance.calculate_fit_metrics(elections_model.trace)
                print("Fit Metrics:")
                for key, value in metrics_dict.items():
                    # Check if value is a number (int, float, numpy number)
                    if isinstance(value, numbers.Number):
                        # Handle potential NaN values
                        if np.isnan(value):
                             print(f"  {key}: NaN")
                        else:
                             print(f"  {key}: {value:.4f}")
                    # Check if it's one of the calibration dictionaries
                    elif isinstance(value, dict) and key.endswith('_calibration'):
                         print(f"  {key}: [Calibration Data - see plots]")
                    # Otherwise, print the value as a string
                    else:
                        print(f"  {key}: {value}")

                # Optionally save metrics again or just display
                # metrics_path = os.path.join(model_dir, "fit_metrics_diagnose.json")
                # with open(metrics_path, 'w') as f:
                #     json.dump(metrics_dict, f, indent=4)
                # print(f"Fit metrics saved to {metrics_path}")

                # --- Plot Calibration --- #
                viz_dir = os.path.join(model_dir, "visualizations") # Use same dir as other plots
                os.makedirs(viz_dir, exist_ok=True)
                if "poll_calibration" in metrics_dict:
                     plot_reliability_diagram(
                         metrics_dict["poll_calibration"],
                         title="Poll Calibration",
                         filename=os.path.join(viz_dir, "calibration_polls.png")
                     )
                if "result_district_calibration" in metrics_dict: # Check for district calibration key
                     plot_reliability_diagram(
                         metrics_dict["result_district_calibration"],
                         title="District Result Calibration",
                         filename=os.path.join(viz_dir, "calibration_results_district.png")
                     )
                # --- End Plot Calibration --- #

            except ValueError as ve:
                 print(f"Warning: Could not calculate fit metrics: {ve}") # Often due to missing data in idata
            except Exception as metrics_err:
                 print(f"Warning: Failed to calculate fit metrics: {metrics_err}")
        else:
            print("Skipping fit metric calculation (no trace or method unavailable).")
        # --- End Calculate and Display Fit Metrics ---

    except Exception as e:
        print(f"Error during model diagnosis: {e}")
        if args.debug:
            traceback.print_exc()
        return None

def predict(args):
    """Loads a trained model and generates election outcome forecasts (vote shares and seats)."""
    print(f"Generating election outcome forecast for model in {args.load_dir}")
    num_samples_for_seats = args.seat_prediction_samples
    print(f"Will use up to {num_samples_for_seats} posterior samples for seat prediction simulation.")

    try:
        elections_model = load_model(args, args.load_dir, debug=args.debug)

        # <<< Add Debug Check Here >>>
        print(f"DEBUG PREDICT: Loaded elections_model type: {type(elections_model)}")
        if elections_model:
            # Determine the model instance and its type
            model_instance = elections_model.model_instance
            model_instance_type = type(model_instance) if model_instance else None
            print(f"DEBUG PREDICT: Model instance type: {model_instance_type}")
            print(f"DEBUG PREDICT: Has get_latent_popularity method (facade)? {hasattr(elections_model, 'get_latent_popularity')}")
            if model_instance:
                 print(f"DEBUG PREDICT: Has get_district_vote_share_posterior method (instance)? {hasattr(model_instance, 'get_district_vote_share_posterior')}")
        # <<< End Debug Check >>>

        if elections_model is None: print("Exiting prediction due to loading error."); return
        if elections_model.trace is None: print(f"No trace found in {args.load_dir}. Cannot generate predictions."); return
        if elections_model.dataset is None: print("Dataset not loaded in model. Cannot generate predictions."); return

        pred_dir = os.path.join(args.load_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)

        # --- Get Posterior Vote Shares (National or District based on model) ---
        target_pop_posterior = None # For national/static model
        district_shares_posterior = None # For dynamic_gp model
        model_instance = elections_model.model_instance # Get the underlying model instance
        hdi_prob = 0.94 # Define hdi_prob before the conditional blocks

        # Determine which method to call based on the actual model instance type
        if isinstance(model_instance, DynamicGPElectionModel) and hasattr(model_instance, 'get_district_vote_share_posterior'):
            print(f"\nCalculating District Vote Share Posterior (mode: {args.prediction_date_mode})...")
            district_shares_posterior = model_instance.get_district_vote_share_posterior(
                idata=elections_model.trace,
                date_mode=args.prediction_date_mode
            )
            if district_shares_posterior is None:
                 print("Could not retrieve district vote shares. Skipping seat predictions.")
            else:
                 print("District vote share posterior calculated successfully.")
                 # Optional: Display summary for a specific district or national average if needed
                 # For now, we proceed directly to seat simulation

        elif hasattr(elections_model, 'get_latent_popularity'): # Fallback or for static model
             print(f"\nCalculating Latent Popularity Prediction (mode: {args.prediction_date_mode})...")
             # This likely gets national popularity for static models or if district method failed/unavailable
             target_pop_posterior = elections_model.get_latent_popularity(date_mode=args.prediction_date_mode)
             if target_pop_posterior is None:
                  print("Could not retrieve latent popularity. Skipping predictions.")
                  # We might want to exit or handle this case differently
             else:
                 # --- Display Vote Share Summary (for National/Static) --- 
                 pred_summary = None # Initialize pred_summary before the try block
                 try: # Start try block for summary calculation
                     pred_summary = az.summary(target_pop_posterior, hdi_prob=hdi_prob, kind='stats', round_to=None)
                     pred_summary = pred_summary.rename(columns={
                         'mean': 'Mean', 'sd': 'SD',
                         f'hdi_{100*(1-hdi_prob)/2:.1f}%': f'HDI {100*(1-hdi_prob)/2:.0f}%',
                         f'hdi_{100*(1-(1-hdi_prob)/2):.1f}%': f'HDI {100*(1-(1-hdi_prob)/2):.0f}%'
                     })
                     output_cols_display = ['Mean', f'HDI {100*(1-hdi_prob)/2:.0f}%', f'HDI {100*(1-(1-hdi_prob)/2):.0f}%']
                     output_cols_display = [col for col in output_cols_display if col in pred_summary.columns]
                     if output_cols_display:
                          pred_summary_display = pred_summary[output_cols_display]
                          # Ensure formatting handles potential non-numeric gracefully if needed
                          pred_summary_pct = pred_summary_display.applymap(lambda x: f"{x*100:.1f}%" if pd.notnull(x) and isinstance(x, numbers.Number) else "N/A")
                          print(f"\n--- Latent Popularity Prediction ({args.prediction_date_mode}) --- (National Vote Shares) ---")
                          print(pred_summary_pct.to_string())
                          print("----------------------------------------------------------------")
                     # Use clear filename based on mode
                     output_path = os.path.join(pred_dir, f"vote_share_summary_{args.prediction_date_mode}.csv")
                     pred_summary.to_csv(output_path)
                     print(f"National vote share prediction summary saved to {output_path}")
                 except Exception as summary_err:
                     print(f"Warning: Failed to calculate or display vote share summary statistics: {summary_err}")
                 # --- End Display Vote Share Summary ---
        else:
            print("Error: Cannot determine how to calculate vote share posteriors for the loaded model.")
            return # Exit if no way to get shares
        # --- End Posterior Share Calculation ---


        # --- Call SEAT PREDICTION SIMULATION --- 
        seats_df = None
        seat_summary = None
        # Check if we have the necessary inputs for seat simulation
        posterior_data_for_seats = district_shares_posterior if district_shares_posterior is not None else target_pop_posterior

        if num_samples_for_seats > 0 and elections_model.dataset and posterior_data_for_seats is not None:
             # Check if we are using district shares (indicating DynamicGP model)
             using_district_shares = district_shares_posterior is not None
             if using_district_shares:
                  print("\nStarting seat simulation using DIRECT DISTRICT shares...")
             else:
                  print("\nStarting seat simulation using NATIONAL shares (UNS implied)...")

             try:
                 # Call the simulation function - it now expects district shares if available
                 seats_df, seat_summary = simulate_seat_allocation(
                     # Pass the appropriate posterior data
                     district_vote_share_posterior=posterior_data_for_seats, 
                     dataset=elections_model.dataset,
                     num_samples_for_seats=num_samples_for_seats,
                     pred_dir=pred_dir,
                     prediction_date_mode=args.prediction_date_mode,
                     hdi_prob=hdi_prob, # Reuse hdi_prob from vote share summary
                     debug=args.debug
                 )

                 # --- Plot Seat Distribution Histograms (if simulation succeeded) --- 
                 if seats_df is not None:
                     try:
                         print("\nGenerating seat distribution histograms...")
                         plot_seat_distribution_histograms(
                             seats_df,
                             pred_dir,
                             date_mode=args.prediction_date_mode,
                             # Use appropriate filename based on simulation type
                             filename=f"seat_histograms_{'direct_' if using_district_shares else 'UNS_'}{args.prediction_date_mode}.png"
                         )
                     except Exception as plot_err:
                         print(f"Warning: Failed to generate seat distribution histograms: {plot_err}")
                         if args.debug: 
                             import traceback # Import locally
                             traceback.print_exc()
                 else:
                     print("\nSkipping seat histogram plot as simulation did not produce results.")

             except Exception as simulation_err:
                 print(f"Error during seat prediction simulation call: {simulation_err}")
                 if args.debug: 
                     import traceback # Import locally
                     traceback.print_exc()

        elif num_samples_for_seats <= 0:
             print("\nSeat prediction simulation skipped (num_samples_for_seats <= 0).")
        elif posterior_data_for_seats is None:
             print("\nSeat prediction simulation skipped (posterior vote share data not available).")
        else: # elections_model.dataset is None
             print("\nSeat prediction simulation skipped (dataset not available).")
        # --- END SEAT PREDICTION SIMULATION CALL ---

        # --- Generate Latent Trend Plot --- # Moved outside the 'else' block to run even if popularity fails
        try:
             print("\nGenerating plot for latent trend since last election...")
             # Define plot_dir robustly (might be same as pred_dir)
             plot_dir = os.path.join(args.load_dir, "visualizations") 
             os.makedirs(plot_dir, exist_ok=True) 
             # Assuming the plot function can handle None trace gracefully or we check before calling
             if elections_model and elections_model.trace is not None:
                  plot_latent_trend_since_last_election(elections_model, plot_dir)
             else:
                  print("Skipping latent trend plot: model or trace not available.")
        except Exception as trend_plot_err:
             print(f"Warning: Failed to generate latent trend plot: {trend_plot_err}")
             if args.debug:
                  import traceback # Import locally
                  traceback.print_exc()
        # --- End Latent Trend Plot ---

        # --- Forecast Distribution Plotting --- # Moved outside the 'else' block
        # This plot currently relies on national latent popularity
        if elections_model and hasattr(elections_model, 'model_instance') and elections_model.model_instance:
            # Check if the necessary NATIONAL posterior exists before plotting
            if target_pop_posterior is not None:
                try:
                     print("\nAttempting to generate forecast distribution plot (based on national trend)...")
                     # Ensure pred_dir exists
                     os.makedirs(pred_dir, exist_ok=True)
                     plot_forecasted_election_distribution(
                         elections_model, 
                         pred_dir, 
                         date_mode=args.prediction_date_mode, 
                         filename=f"forecast_distribution_{args.prediction_date_mode}.png"
                     )
                except Exception as plot_err:
                     print(f"Warning: Failed to generate forecast distribution plot: {plot_err}")
                     if args.debug:
                          import traceback # Import locally
                          traceback.print_exc()
            else:
                 print("Skipping forecast distribution plot: national target popularity not calculated.")
        else:
             print("Warning: Model instance not available for plotting forecast distribution.")

        print(f"\nPrediction generation step complete. Results saved to {pred_dir}")

    except Exception as e:
        print(f"Error during prediction mode: {e}")
        # <<< Import traceback locally here >>>
        if args.debug: 
            import traceback 
            traceback.print_exc()

def main(args=None):
    parser = argparse.ArgumentParser(description="Election Model CLI")
    
    # --- Mode Selection ---
    parser.add_argument(
        "--mode",
        choices=["train", "viz", "visualize-data", "diagnose", "cross-validate", "predict"],
        required=True,
        help="Operation mode",
    )
    parser.add_argument(
        "--model-type",
        choices=["static", "dynamic_gp"],
        default="static", # Default to the original static model
        help="Type of election model to use ('static' or 'dynamic_gp')",
    )
    
    # --- Common Arguments ---
    parser.add_argument(
        "--dataset",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--election-date",
        help="The target election date (YYYY-MM-DD). Required for 'train'. Used for context in others.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/", # Default to base 'outputs' directory
        help="Base directory to save outputs (timestamped/latest subdir will be created)",
    )
    parser.add_argument(
        "--load-dir",
        help="Directory to load model from for 'viz', 'diagnose', 'predict' modes.",
    )
    parser.add_argument(
        "--cutoff-date",
        help="Exclude data after this number of days before election_date (e.g., '30d') for retrodictive testing",
    )
    parser.add_argument(
        "--notify", action="store_true", help="Send ntfy.sh notification on completion/error",
    )
    parser.add_argument(
        "--seed", type=int, default=8675309, help="Random seed",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debugging output",
    )
    
    # --- Sampling Arguments ---
    sampling_group = parser.add_argument_group('Sampling Parameters (for train, cross-validate)')
    sampling_group.add_argument(
        "--draws", type=int, default=1500, help="Number of posterior draws per chain",
    )
    sampling_group.add_argument(
        "--chains", type=int, default=4, help="Number of MCMC chains",
    )
    sampling_group.add_argument(
        "--tune", type=int, default=1500, help="Number of tuning samples per chain",
    )
    sampling_group.add_argument(
        "--target-accept", type=float, default=0.98, help="Target acceptance rate for NUTS",
    )

    # --- Static Model Specific Arguments ---
    static_group = parser.add_argument_group('Static Model Parameters')
    static_group.add_argument(
        "--baseline-timescale",
        type=float,
        nargs="+", # Allow multiple, though model uses first
        default=[DEFAULT_BASELINE_TIMESCALE], # Make default a list
        help="Baseline timescale(s) in days for static model GP kernel",
    )
    static_group.add_argument(
        "--election-timescale",
        type=float,
        nargs="+", # Allow multiple, though model uses first
        default=[DEFAULT_ELECTION_TIMESCALES], # Make default a list with the int value
        help="Election timescale(s) in days for static model GP kernel",
    )
    
    # --- Dynamic GP Model Specific Arguments ---
    dynamic_gp_group = parser.add_argument_group('Dynamic GP Model Parameters')
    dynamic_gp_group.add_argument(
        "--baseline-gp-lengthscale", type=float, default=365.0, help="Baseline GP lengthscale (days) for dynamic_gp"
    )
    dynamic_gp_group.add_argument(
        "--baseline-gp-kernel", choices=["Matern52", "ExpQuad"], default="Matern52", help="Baseline GP kernel for dynamic_gp"
    )
    dynamic_gp_group.add_argument(
        "--cycle-gp-lengthscale", type=float, default=45.0, help="Cycle GP lengthscale (days) for dynamic_gp"
    )
    dynamic_gp_group.add_argument(
        "--cycle-gp-kernel", choices=["Matern52", "ExpQuad"], default="Matern52", help="Cycle GP kernel for dynamic_gp"
    )
    dynamic_gp_group.add_argument(
        "--cycle-gp-max-days", type=int, default=180, help="Maximum days before election for Cycle GP"
    )
    dynamic_gp_group.add_argument(
        "--hsgp-m", type=int, nargs=1, default=[100], help="Number of basis functions (m) for Baseline HSGP"
    )
    dynamic_gp_group.add_argument(
        "--hsgp-c", type=float, default=2.0, help="Expansion factor (c) for Baseline HSGP"
    )
    dynamic_gp_group.add_argument(
        "--hsgp-m-cycle", type=int, nargs=1, default=[50], help="Number of basis functions (m) for Cycle HSGP"
    )
    dynamic_gp_group.add_argument(
        "--hsgp-c-cycle", type=float, default=1.5, help="Expansion factor (c) for Cycle HSGP"
    )

    # --- Prediction Specific Arguments ---
    pred_group = parser.add_argument_group('Prediction Parameters')
    pred_group.add_argument(
        "--seat-prediction-samples", 
        type=int, 
        default=1000, 
        help="Number of posterior samples to use for seat prediction simulation (0 to disable)"
    )

    # --- Prediction Date Mode Argument --- 
    pred_group.add_argument(
        "--prediction-date-mode",
        choices=['election_day', 'last_poll', 'today'],
        default='election_day',
        help="Specify which date's latent popularity to use for prediction (election day, last poll, or today). Default: election_day"
    )

    # --- Cross-validation Specific Arguments ---
    cv_group = parser.add_argument_group('Cross-validation Parameters')
    # cv_group.add_argument(
    #     "--fast", action="store_true", help="Skip plot generation during cross-validation"
    # ) # Example if needed

    args = parser.parse_args(args)
    
    # --- Argument Validation ---
    np.random.seed(args.seed)

    # Mode-specific requirements
    if args.mode == "train" and not args.election_date:
        parser.error("--election-date is required for train mode")
        
    if args.mode in ["predict", "viz", "diagnose"] and not args.load_dir:
        parser.error(f"--load-dir is required for {args.mode} mode")
        
    if args.load_dir and not os.path.isdir(args.load_dir):
        parser.error(f"--load-dir path '{args.load_dir}' does not exist or is not a directory.")

    # Clean up output_dir path if it ends with 'latest' but isn't the default
    if args.output_dir.endswith('/latest') and args.output_dir != "outputs/latest":
         args.output_dir = os.path.dirname(args.output_dir)
         print(f"Adjusted output directory base to: {args.output_dir}")
    elif not args.output_dir.endswith('/'):
         args.output_dir += '/'
         
    # Ensure output directory exists (base path)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Execute Mode ---
    start_main_time = time.time()
    try:
        if args.mode == "train":
            fit_model(args)
        elif args.mode == "predict":
            predict(args)
        elif args.mode == "cross-validate":
            cross_validate(args)
        elif args.mode == "viz":
            visualize(args)
        elif args.mode == "visualize-data":
            visualize_data(args)
        elif args.mode == "diagnose":
            diagnose_model(args)
            
        end_main_time = time.time()
        print(f"\n'{args.mode}' mode finished in {end_main_time - start_main_time:.2f} seconds.")
            
        # Send success notification if requested
        if args.notify:
            try:
                requests.post("https://ntfy.sh/bc-estimador",
                    data=f"{args.mode.capitalize()} ({args.model_type}) completed successfully".encode(encoding='utf-8'))
            except Exception as notify_err:
                print(f"Failed to send success notification: {notify_err}")
                
        return 0 # Success exit code

    except Exception as e:
        end_main_time = time.time()
        print(f"\nError during '{args.mode}' mode after {end_main_time - start_main_time:.2f} seconds: {e}")
        if args.debug:
            traceback.print_exc()
            
        # Send error notification if requested
        if args.notify:
            try:
                requests.post("https://ntfy.sh/bc-estimador",
                    data=f"Error in {args.mode} mode ({args.model_type}): {e}".encode(encoding='utf-8'))
            except Exception as notify_err:
                print(f"Failed to send error notification: {notify_err}")
                
        return 1 # Error exit code

if __name__ == "__main__":
    import sys
    sys.exit(main())