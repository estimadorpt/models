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
from src.models.election_model import ElectionModel
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

def fit_model(args):
    """Fit a new model with the specified parameters"""
    try:
        if args.debug:
            print(f"Fitting model for election date {args.election_date}")
            print(f"Using baseline timescales: {args.baseline_timescale}")
            print(f"Using election timescales: {args.election_timescale}")
        
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
        
        # Initialize the elections model
        elections_model = ElectionsFacade(
            election_date=args.election_date,
            model_class=ElectionModel,
            baseline_timescales=args.baseline_timescale,
            election_timescales=args.election_timescale,
            test_cutoff=pd.Timedelta(args.cutoff_date) if args.cutoff_date else None,
            debug=args.debug
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

        # TODO: Handle model selection based on config/args if multiple models exist
        # For now, assume ElectionModel
        model_class_to_load = ElectionModel

        if debug:
            print(f"Initializing Facade with -> election_date: {final_election_date}")
            print(f"Initializing Facade with -> baseline_timescales: {final_baseline_timescales}")
            print(f"Initializing Facade with -> election_timescales: {final_election_timescales}")
            print(f"Initializing Facade with -> cutoff_date: {final_cutoff_date}")

        elections_model = ElectionsFacade(
            election_date=final_election_date,
            model_class=model_class_to_load,
            baseline_timescales=final_baseline_timescales,
            election_timescales=final_election_timescales,
            test_cutoff=pd.Timedelta(final_cutoff_date) if final_cutoff_date else None, # Pass loaded/default cutoff
            debug=debug
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
                model_class=ElectionModel,
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
    parser.add_argument(
        "--mode",
        choices=["train", "viz", "visualize-data", "diagnose", "cross-validate", "predict"],
        required=True,
        help="Operation mode",
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--election-date",
        help="The election date to predict (required for train/predict modes)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/latest",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--train-timescales", 
        action='store_true', 
        help="Train kernel timescales"
    )
    parser.add_argument(
        "--baseline-timescale",
        type=float,
        nargs="+",
        default=DEFAULT_BASELINE_TIMESCALE,
        help="Baseline timescale(s) in days for GP kernel",
    )
    parser.add_argument(
        "--election-timescale",
        type=float,
        nargs="+",
        default=DEFAULT_ELECTION_TIMESCALES,
        help="Election timescale(s) in days for GP kernel",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="Number of posterior draws",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="Number of tuning samples per chain",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.95,
        help="Target acceptance rate",
    )
    parser.add_argument(
        "--load-dir",
        help="Load model from directory for prediction",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send notification on completion",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8675309,
        help="Random seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging output",
    )
    parser.add_argument(
        "--cutoff-date",
        help="Exclude data after this date for retrodictive testing",
    )
    
    args = parser.parse_args(args)
    
    # Set random seed
    np.random.seed(args.seed)

    # Add specific argument requirements based on mode
    if args.mode == "train" and not args.election_date:
        parser.error("--election-date is required for train mode")
        
    # Prediction mode needs a load directory
    if args.mode == "predict" and not args.load_dir:
        parser.error("--load-dir is required for predict mode")
        
    # Visualization modes need a load directory
    if args.mode in ["viz", "diagnose"] and not args.load_dir:
        parser.error("--load-dir is required for viz/diagnose modes")

    # Run the selected mode
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
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            traceback.print_exc()
        if args.notify:
            try:
                requests.post("https://ntfy.sh/bc-estimador",
                    data=f"Error in {args.mode} mode: {e}".encode(encoding='utf-8'))
            except Exception as notify_err:
                print(f"Failed to send notification: {notify_err}")
        return 1
    
    # Send notification if requested
    if args.notify:
        try:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"{args.mode.capitalize()} completed successfully".encode(encoding='utf-8'))
        except Exception as notify_err:
            print(f"Failed to send notification: {notify_err}")
    
    return 0

if __name__ == "__main__":
    main()