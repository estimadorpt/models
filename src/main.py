import os
import argparse
import arviz
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

from src.models.elections_facade import ElectionsFacade
from src.models.static_baseline_election_model import StaticBaselineElectionModel
from src.models.dynamic_gp_election_model import DynamicGPElectionModel
from src.visualization.plots import (
    plot_election_data, 
    plot_latent_popularity_vs_polls, 
    plot_latent_component_contributions, 
    plot_recent_polls,
    plot_house_effects_heatmap,
    # plot_forecasted_election_distribution # Moved to predict function
    # plot_nowcast_latent_vs_polls_combined # Removed import
)
from src.config import DEFAULT_BASELINE_TIMESCALE, DEFAULT_ELECTION_TIMESCALES
from src.data.dataset import ElectionDataset

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
                 print(f"Using GP lengthscale: {args.gp_lengthscale}") # Use new argument

        # Create the model output directory with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
        
        # Create output directory with timestamp as a subdirectory
        base_output_dir = args.output_dir.rstrip('/')
        if base_output_dir.endswith('/latest'):
            base_output_dir = os.path.dirname(base_output_dir)
            
        output_dir = os.path.join(base_output_dir, timestamp)
        
        # Make sure all parent directories exist
        os.makedirs(output_dir, exist_ok=True)
        
        if args.debug:
            print(f"Using output directory: {output_dir}")
        
        # --- Prepare Model Kwargs ---
        model_kwargs = {
             "debug": args.debug
        }
        if args.model_type == "static":
             model_kwargs["baseline_lengthscale"] = args.baseline_timescale
             model_kwargs["election_lengthscale"] = args.election_timescale
        elif args.model_type == "dynamic_gp":
             model_kwargs["gp_lengthscale"] = args.gp_lengthscale # Pass new arg
             model_kwargs["gp_kernel"] = args.gp_kernel # Pass kernel type
             model_kwargs["hsgp_m"] = args.hsgp_m # Pass hsgp m
             model_kwargs["hsgp_c"] = args.hsgp_c # Pass hsgp c
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
            max_treedepth=10 # Keeping lower tree depth for faster run if needed
        )
        
        # Calculate elapsed time
        end_time = time.time()
        fitting_duration = end_time - start_time
        print(f"Model fitting completed in {fitting_duration:.2f} seconds")
        
        # Send notification if requested
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Model fit completed in {fitting_duration:.2f} seconds".encode(encoding='utf-8'))
        
        # Generate diagnostic plots
        if elections_model.trace is not None:
            diag_plot_dir = os.path.join(output_dir, "diagnostics")
            elections_model.generate_diagnostic_plots(diag_plot_dir)
            # If there are model-specific diagnostic plots, call them here:
            # if hasattr(model_instance, 'generate_specific_diagnostics'):
            #    model_instance.generate_specific_diagnostics(diag_plot_dir)
        else:
            print("Skipping diagnostic plot generation as trace is not available.")
        
        # Save model configuration
        print(f"Saving model configuration to {output_dir}/model_config.json")
        config_to_save = {
            "election_date": args.election_date,
            "baseline_timescales": args.baseline_timescale,
            "election_timescales": args.election_timescale,
            "cutoff_date": args.cutoff_date,
        }
        config_path = os.path.join(output_dir, "model_config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            print("Model configuration saved successfully.")
        except Exception as config_err:
            print(f"Warning: Failed to save model configuration: {config_err}")
        
        # Debug: Check trace before saving
        # print(f"DEBUG: Before saving - elections_model.trace is None? {elections_model.trace is None}")
        # if elections_model.trace is not None:
        #      print(f"DEBUG: Before saving - elections_model.trace type: {type(elections_model.trace)}")
        # else:
        #      print(f"DEBUG: Before saving - trace object does not exist or is None.")
              
        # Save the trace and model - this now returns True/False
        # print(f"DEBUG: Type of elections_model just before save call: {type(elections_model)}")
        # try:
        #     print(f"DEBUG: elections_model.save_inference_results method: {elections_model.save_inference_results}")
        # except AttributeError:
        #     print("DEBUG: elections_model does NOT have attribute save_inference_results")
              
        save_successful = elections_model.save_inference_results(output_dir)
        
        # Create/update the 'latest' symlink only if saving was successful
        if save_successful:
            latest_link_path = os.path.join(base_output_dir, "latest")
            target_path_absolute = os.path.abspath(output_dir)
            
            print(f"Updating symbolic link '{latest_link_path}' to point to '{target_path_absolute}'")
            
            try:
                # Remove existing link/file if it exists
                if os.path.islink(latest_link_path) or os.path.exists(latest_link_path):
                    os.remove(latest_link_path)
                
                # Create the new symlink
                os.symlink(target_path_absolute, latest_link_path, target_is_directory=True)
                print(f"Symbolic link 'latest' updated successfully.")
            except Exception as symlink_err:
                print(f"Warning: Failed to create/update symbolic link: {symlink_err}")
        else:
            print("Skipping 'latest' symlink creation as no inference results were saved.")
        
        print(f"Training process complete for {args.election_date}.")
        return elections_model
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        if args.debug:
            traceback.print_exc()
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Error fitting model: {e}".encode(encoding='utf-8'))
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
        final_model_type = args.model_type if args.model_type != "static" else loaded_config.get('model_type', None)
        if not final_model_type:
            print("Warning: Model type not found in args or config, defaulting to 'static'.")
            final_model_type = 'static' # Default if still not found

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
            # Priority: args -> loaded_config -> default (handled by model __init__)
            gp_len_arg = args.gp_lengthscale if hasattr(args, 'gp_lengthscale') and args.gp_lengthscale is not None else None
            gp_ker_arg = args.gp_kernel if hasattr(args, 'gp_kernel') and args.gp_kernel is not None else None
            hsgp_m_arg = args.hsgp_m if hasattr(args, 'hsgp_m') and args.hsgp_m is not None else None
            hsgp_c_arg = args.hsgp_c if hasattr(args, 'hsgp_c') and args.hsgp_c is not None else None

            final_gp_lengthscale = gp_len_arg if gp_len_arg is not None else loaded_config.get('gp_lengthscale', None)
            final_gp_kernel = gp_ker_arg if gp_ker_arg is not None else loaded_config.get('gp_kernel', None)
            final_hsgp_m = hsgp_m_arg if hsgp_m_arg is not None else loaded_config.get('hsgp_m', None)
            final_hsgp_c = hsgp_c_arg if hsgp_c_arg is not None else loaded_config.get('hsgp_c', None)

            # Only add to kwargs if a value was determined (otherwise let model default handle it)
            if final_gp_lengthscale is not None: model_kwargs["gp_lengthscale"] = final_gp_lengthscale
            if final_gp_kernel is not None: model_kwargs["gp_kernel"] = final_gp_kernel
            if final_hsgp_m is not None: model_kwargs["hsgp_m"] = final_hsgp_m
            if final_hsgp_c is not None: model_kwargs["hsgp_c"] = final_hsgp_c
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
        
        # Load the saved trace
        elections_model.load_inference_results(directory)
        
        # Rebuild the model structure (necessary for posterior predictive checks etc.)
        # This uses the parameters passed to ElectionsFacade, ensuring consistency
        print("Rebuilding model structure...")
        elections_model.build_model()
        print("Model structure rebuilt.")
        
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

        print(f"Diagnostic plots saved to {diag_plot_dir}")

    except Exception as e:
        print(f"Error during model diagnosis: {e}")
        if args.debug:
            traceback.print_exc()
        return None

def predict(args):
    """Loads a trained model and generates the election outcome forecast distribution."""
    print(f"Generating election outcome forecast for model in {args.load_dir}")
    
    try:
        # 1. Read election_date from config first
        config_path = os.path.join(args.load_dir, "model_config.json")
        if not os.path.exists(config_path):
            print(f"Error: model_config.json not found in {args.load_dir}")
            return
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            loaded_election_date = config.get('election_date')
            if not loaded_election_date:
                 raise ValueError("election_date not found or is empty in model_config.json")
        except (json.JSONDecodeError, ValueError, Exception) as e:
             print(f"Error reading or parsing election_date from {config_path}: {e}")
             return

        # 2. Instantiate Facade with the loaded date
        print(f"Instantiating model for election date: {loaded_election_date}")
        elections_model = ElectionsFacade(election_date=loaded_election_date, debug=args.debug)
        
        # 3. Load the inference results (trace, etc.)
        load_status = elections_model.load_inference_results(args.load_dir) # DEBUG
        if not load_status:
             print(f"Error: Failed to load model results from {args.load_dir}")
             return

        # Rebuild model structure if needed (though prediction might not need full rebuild)
        # predict_election_outcome primarily needs trace data.
        # Let's keep it simple for now and assume predict_election_outcome handles needed data.
        
        # Define output directory for predictions
        pred_dir = os.path.join(args.load_dir, "predictions") # Changed from visualizations
        os.makedirs(pred_dir, exist_ok=True)
        
        # Generate and plot the forecast distribution
        # Import the specific plot function here or ensure it's imported globally
        from src.visualization.plots import plot_forecasted_election_distribution
        if elections_model.model_instance:
            plot_forecasted_election_distribution(elections_model.model_instance, pred_dir)
        else:
             print("Warning: Model instance not available for plotting forecast distribution.")
        
        print(f"\nPrediction generation step complete. Results saved to {pred_dir}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
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
        "--draws", type=int, default=1000, help="Number of posterior draws per chain",
    )
    sampling_group.add_argument(
        "--chains", type=int, default=4, help="Number of MCMC chains",
    )
    sampling_group.add_argument(
        "--tune", type=int, default=1000, help="Number of tuning samples per chain",
    )
    sampling_group.add_argument(
        "--target-accept", type=float, default=0.95, help="Target acceptance rate for NUTS",
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
        "--gp-lengthscale", type=float, default=180.0, help="GP lengthscale in days for dynamic_gp model"
    )
    dynamic_gp_group.add_argument(
        "--gp-kernel", choices=["Matern52", "ExpQuad"], default="Matern52", help="GP kernel type for dynamic_gp model"
    )
    dynamic_gp_group.add_argument(
        "--hsgp-m", type=int, nargs=1, default=[100], help="Number of basis functions (m) for HSGP in dynamic_gp model"
    )
    dynamic_gp_group.add_argument(
        "--hsgp-c", type=float, default=2.0, help="Expansion factor (c) for HSGP in dynamic_gp model"
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