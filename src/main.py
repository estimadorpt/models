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
            baseline_timescales=args.baseline_timescale,
            election_timescales=args.election_timescale,
            test_cutoff=pd.Timedelta(args.cutoff_date) if args.cutoff_date else None,
            debug=args.debug
        )
        
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

def load_model(directory, election_date=None, baseline_timescales=None, election_timescales=None, debug=False):
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

        # Use loaded config values if available, otherwise use provided args or defaults
        final_election_date = loaded_config.get('election_date', election_date if election_date else '2026-01-01')
        # Ensure loaded values override function args if config exists
        final_baseline_timescales = loaded_config.get('baseline_timescales', baseline_timescales if baseline_timescales is not None else DEFAULT_BASELINE_TIMESCALE)
        final_election_timescales = loaded_config.get('election_timescales', election_timescales if election_timescales is not None else DEFAULT_ELECTION_TIMESCALES)
        # Add other parameters like cutoff_date if needed by Facade init
        final_cutoff_date = loaded_config.get('cutoff_date', None) # Assuming Facade handles None

        # Ensure timescales are lists (might be redundant if config saves them correctly, but safe)
        if not isinstance(final_baseline_timescales, list):
            final_baseline_timescales = [final_baseline_timescales]
        if not isinstance(final_election_timescales, list):
            final_election_timescales = [final_election_timescales]
            
        if debug:
            print(f"Initializing Facade with -> election_date: {final_election_date}")
            print(f"Initializing Facade with -> baseline_timescales: {final_baseline_timescales}")
            print(f"Initializing Facade with -> election_timescales: {final_election_timescales}")
            print(f"Initializing Facade with -> cutoff_date: {final_cutoff_date}")

        elections_model = ElectionsFacade(
            election_date=final_election_date,
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
        elections_model.model.build_model()
        print("Model structure rebuilt.")
        
        return elections_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        if debug:
            traceback.print_exc()
        return None

def nowcast(args):
    """Run nowcasting by refitting the model with all available data up to the present."""
    print("Running nowcast by refitting model with all available data...")
    start_time = time.time()

    # --- 1. Prepare Output Directory --- 
    output_dir = os.path.join(args.output_dir, "nowcast_refit") # New subdir name
    os.makedirs(output_dir, exist_ok=True)
    latest_base_dir = os.path.join(args.output_dir, "latest")
    os.makedirs(latest_base_dir, exist_ok=True)
    latest_nowcast_dir = os.path.join(latest_base_dir, "nowcast_refit") # Link name
    if os.path.exists(latest_nowcast_dir):
        if os.path.islink(latest_nowcast_dir):
            os.unlink(latest_nowcast_dir)
        else:
            import shutil
            shutil.rmtree(latest_nowcast_dir)
    try:
        abs_output_dir = os.path.abspath(output_dir)
        abs_latest_nowcast_dir = os.path.abspath(latest_nowcast_dir)
        os.symlink(abs_output_dir, abs_latest_nowcast_dir, target_is_directory=True)
        if args.debug:
            print(f"Created symbolic link from {abs_latest_nowcast_dir} to {abs_output_dir}")
    except OSError as e:
        print(f"Warning: Could not create symbolic link: {e}")

    # --- 2. Load Configuration from Training Run --- 
    loaded_config = {}
    config_path = os.path.join(args.load_dir, "model_config.json") # Load from the specified training run
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            print(f"Loaded configuration from {config_path}: {loaded_config}")
        except Exception as config_err:
            print(f"ERROR: Failed to load model configuration from {config_path}: {config_err}")
            return # Cannot proceed without config
    else:
        print(f"ERROR: model_config.json not found in {args.load_dir}. Cannot proceed.")
        return

    # Use loaded config or defaults for Facade initialization
    baseline_timescales = loaded_config.get('baseline_timescales', args.baseline_timescale)
    election_timescales = loaded_config.get('election_timescales', args.election_timescale)
    # Ensure they are lists
    if not isinstance(baseline_timescales, list): baseline_timescales = [baseline_timescales]
    if not isinstance(election_timescales, list): election_timescales = [election_timescales]

    # --- 3. Initialize Facade & Determine Nowcast Date --- 
    try:
        # Temporarily instantiate dataset just to get the latest poll date
        temp_dataset = ElectionDataset(election_date="2099-01-01", test_cutoff=None) # Use future date context
        latest_poll_date = pd.to_datetime(temp_dataset.polls['date']).max()
        nowcast_election_date = (latest_poll_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Latest poll date: {latest_poll_date.strftime('%Y-%m-%d')}. Setting nowcast context date to: {nowcast_election_date}")

        # Now instantiate the actual Facade for the refit
        # CRUCIALLY set test_cutoff=None to use ALL polls for training
        elections_model = ElectionsFacade(
            election_date=nowcast_election_date, # Use date for context endpoint
            baseline_timescales=baseline_timescales,
            election_timescales=election_timescales,
            test_cutoff=None, 
            debug=args.debug
        )
        print(f"Facade initialized. Dataset contains {len(elections_model.dataset.polls)} total polls.")
        # Verify polls_train contains all polls
        if len(elections_model.dataset.polls_train) != len(elections_model.dataset.polls):
             print(f"WARNING: polls_train ({len(elections_model.dataset.polls_train)}) != total polls ({len(elections_model.dataset.polls)}). Check dataset init logic.")
        else:
             print("Confirmed: polls_train contains all loaded polls.")
             
    except Exception as e:
        print(f"Error initializing ElectionsFacade: {e}")
        if args.debug:
            traceback.print_exc()
        return

    # --- 4. Run Inference (Refit) --- 
    try:
        print("\nStarting model refit using all data...")
        # Use potentially fewer draws/tune for speed in nowcast setting
        # TODO: Make draws/tune configurable via args for nowcast mode?
        nowcast_draws = args.draws // 2 if args.draws > 500 else 500 # Example: half draws, min 500
        nowcast_tune = args.tune // 2 if args.tune > 500 else 500   # Example: half tune, min 500
        print(f"Using draws={nowcast_draws}, tune={nowcast_tune}")
        
        elections_model.run_inference(
            draws=nowcast_draws,
            tune=nowcast_tune,
            target_accept=args.target_accept # Use target_accept from args
        )
    except Exception as e:
        print(f"Error during model refitting: {e}")
        if args.debug:
            traceback.print_exc()
        return
        
    # --- 5. Save Results --- 
    try:
        # Save the new trace and the config used for THIS run
        current_config = {
            "election_date": nowcast_election_date,
            "baseline_timescales": baseline_timescales,
            "election_timescales": election_timescales,
            "cutoff_date": None, # Explicitly None for refit
            "refit_draws": nowcast_draws,
            "refit_tune": nowcast_tune
        }
        config_path = os.path.join(output_dir, "nowcast_config.json")
        with open(config_path, 'w') as f:
            json.dump(current_config, f, indent=4)
            
        elections_model.save_inference_results(output_dir) # Saves trace etc.
        print(f"Refit results (trace, config) saved to {output_dir}")
    except Exception as e:
        print(f"Error saving refit results: {e}")
        # Continue to plotting if possible

    # --- 6. Plot Results --- 
    try:
        print("\nGenerating plots from refit results...")
        # Use the existing plot function designed for a fitted model
        plot_latent_popularity_vs_polls(elections_model, output_dir)
        # Add other relevant plots if desired, e.g.:
        # plot_latent_component_contributions(elections_model, output_dir)
        plot_house_effects_heatmap(elections_model, output_dir)
        print(f"Plots saved to {output_dir}")
    except Exception as plot_err:
        print(f"Warning: Failed to generate plots from refit results: {plot_err}")
        if args.debug:
            traceback.print_exc()

    print(f"\nNowcast refit completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Results saved to {output_dir}")
    print("==========================================\n")

def visualize(args):
    """Generate visualizations for a saved model"""
    try:
        if args.debug:
            print(f"Generating visualizations for model in {args.load_dir}")
        
        model_dir = os.path.abspath(args.load_dir)
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")
            
        viz_dir = os.path.join(model_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load the model (which now includes the full refit trace)
        elections_model = load_model(
            directory=model_dir,
            debug=args.debug,
            # Pass arguments explicitly or let load_model use config
            election_date=args.election_date, 
            baseline_timescales=args.baseline_timescale, 
            election_timescales=args.election_timescale 
        )
        
        if elections_model is None:
            raise ValueError(f"Failed to load model from {model_dir}")
        
        # Rebuild the underlying PyMC model structure after loading trace/config
        print("\nRebuilding model structure...")
        try:
            elections_model.build_model()
            print("Model structure rebuilt successfully.")
        except Exception as build_err:
            print(f"ERROR: Failed to rebuild model structure after loading: {build_err}")
            # Decide if we can proceed without the full structure for some plots
            # For now, let's exit if build fails as prediction needs it.
            return

        # --- Generate Standard Historical Plots --- 
        print("\nGenerating historical model diagnostics and visualization plots...")
        # These use the posterior trace directly
        plot_latent_popularity_vs_polls(elections_model, viz_dir, include_target_date=True)
        plot_latent_component_contributions(elections_model, viz_dir)
        plot_recent_polls(elections_model, viz_dir) # Might need adjustment if it expects specific test set
        plot_house_effects_heatmap(elections_model, viz_dir)
        print(f"Historical visualizations saved to {viz_dir}")
        
        # --- Generate Forecast Distribution Plot ---
        # print("\nGenerating election outcome forecast distribution...")
        # plot_forecasted_election_distribution(elections_model, viz_dir)
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
        plot_election_data(dataset)

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
        elections_model = load_model(
            directory=model_dir,
            debug=args.debug,
            # Pass arguments explicitly
            election_date=args.election_date, # Pass command-line args or default handled inside load_model
            baseline_timescales=args.baseline_timescale, 
            election_timescales=args.election_timescale
        )

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
        plot_forecasted_election_distribution(elections_model, pred_dir)
        
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