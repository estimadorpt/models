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

from src.models.elections_facade import ElectionsFacade
from src.visualization.plots import plot_election_data, plot_latent_popularity_vs_polls, plot_latent_component_contributions, plot_recent_polls
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
        elections_model.run_inference(
            draws=args.draws,
            tune=args.tune,
            target_accept=0.9
        )
        
        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Model fitting completed in {elapsed_time:.2f} seconds")
        
        # Send notification if requested
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Model fit completed in {elapsed_time:.2f} seconds".encode(encoding='utf-8'))
        
        # Generate diagnostic plots
        if elections_model.trace is not None:
            diag_plot_dir = os.path.join(output_dir, "diagnostics")
            elections_model.generate_diagnostic_plots(diag_plot_dir)
        else:
            print("Skipping diagnostic plot generation as trace is not available.")
        
        # Save the trace and model
        elections_model.save_inference_results(output_dir)
        
        # Create a symbolic link to the latest run
        latest_dir = os.path.join(base_output_dir, "latest")
        if os.path.exists(latest_dir):
            if os.path.islink(latest_dir):
                os.unlink(latest_dir)
            else:
                import shutil
                shutil.rmtree(latest_dir)
        
        try:
            # Convert to absolute paths for better macOS compatibility
            abs_output_dir = os.path.abspath(output_dir)
            abs_latest_dir = os.path.abspath(latest_dir)
            
            # On macOS, os.symlink order is: link_target, link_name
            os.symlink(abs_output_dir, abs_latest_dir, target_is_directory=True)
            if args.debug:
                print(f"Created symbolic link from {abs_latest_dir} to {abs_output_dir}")
        except OSError as e:
            print(f"Warning: Could not create symbolic link: {e}")
        
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
        
        # Use provided args or defaults
        final_election_date = election_date if election_date else '2026-01-01'
        final_baseline_timescales = baseline_timescales if baseline_timescales is not None else DEFAULT_BASELINE_TIMESCALE
        final_election_timescales = election_timescales if election_timescales is not None else DEFAULT_ELECTION_TIMESCALES

        # Ensure timescales are lists
        if not isinstance(final_baseline_timescales, list):
            final_baseline_timescales = [final_baseline_timescales]
        if not isinstance(final_election_timescales, list):
            final_election_timescales = [final_election_timescales]
            
        if debug:
            print(f"Using election_date: {final_election_date}")
            print(f"Using baseline_timescales: {final_baseline_timescales}")
            print(f"Using election_timescales: {final_election_timescales}")
        
        elections_model = ElectionsFacade(
            election_date=final_election_date,
            baseline_timescales=final_baseline_timescales,
            election_timescales=final_election_timescales,
            debug=debug
        )
        
        # Load the saved trace
        elections_model.load_inference_results(directory)
        
        # Rebuild the model structure (necessary for posterior predictive checks etc.)
        # This uses the parameters passed to ElectionsFacade, ensuring consistency
        print("Rebuilding model structure...")
        elections_model.model.model = elections_model.model.build_model()
        print("Model structure rebuilt.")
        
        return elections_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        if debug:
            traceback.print_exc()
        return None

def nowcast(args):
    """Run nowcasting to estimate current party support."""
    print("Running nowcast based on pre-trained model...")
    start_time = time.time()
    
    # Load the pre-trained model
    output_dir = os.path.join(args.output_dir, "nowcast")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create latest directory if it doesn't exist
    latest_base_dir = os.path.join(args.output_dir, "latest")
    os.makedirs(latest_base_dir, exist_ok=True)
    
    # Create a symbolic link to the latest nowcast run
    latest_nowcast_dir = os.path.join(latest_base_dir, "nowcast")
    if os.path.exists(latest_nowcast_dir):
        if os.path.islink(latest_nowcast_dir):
            os.unlink(latest_nowcast_dir)
        else:
            import shutil
            shutil.rmtree(latest_nowcast_dir)
    
    try:
        # Convert to absolute paths for better macOS compatibility
        abs_output_dir = os.path.abspath(output_dir)
        abs_latest_nowcast_dir = os.path.abspath(latest_nowcast_dir)
        
        # On macOS, os.symlink order is: link_target, link_name
        os.symlink(abs_output_dir, abs_latest_nowcast_dir, target_is_directory=True)
        if args.debug:
            print(f"Created symbolic link from {abs_latest_nowcast_dir} to {abs_output_dir}")
    except OSError as e:
        print(f"Warning: Could not create symbolic link: {e}")
    
    try:
        elections_model = load_model(
            directory=args.load_dir, 
            debug=args.debug,
            # Pass arguments explicitly
            election_date=elections_model.election_date if hasattr(elections_model, 'election_date') else args.election_date,
            baseline_timescales=args.baseline_timescale, # Pass command-line args
            election_timescales=args.election_timescale  # Pass command-line args
        )
        
        if elections_model is None:
            raise ValueError(f"Failed to load model from {args.load_dir}")
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Fallback: use tomorrow as election date
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        if args.debug:
            print(f"Using fallback election date: {tomorrow}")
        return

    # Get access to all polls regardless of how the dataset was split
    try:
        # Use the original polls data that hasn't been filtered for historical elections
        current_polls = elections_model.dataset.polls.copy()
        
        # Convert percentages to counts for DirichletMultinomial model
        current_polls = elections_model.dataset.cast_as_multinomial(current_polls)
        
        if args.debug:
            print(f"Successfully loaded all polls (including future election polls)")
    except AttributeError as e:
        print(f"Error accessing polls: {e}")
        # If 'polls' doesn't exist, combine train and test
        try:
            current_polls = pd.concat([
                elections_model.dataset.polls_train.copy(), 
                elections_model.dataset.polls_test.copy()
            ]).drop_duplicates()
        except (AttributeError, ValueError):
            # If train/test are not available, just use train
            current_polls = elections_model.dataset.polls_train.copy()

    # Get the latest election date from historical elections
    historical_dates = sorted([pd.to_datetime(date) for date in elections_model.dataset.historical_election_dates])
    latest_election_date = historical_dates[-1].strftime('%Y-%m-%d')  # Get the last (most recent) date
    
    if args.debug:
        print(f"Using latest historical election date: {latest_election_date}")
        print(f"Total polls in dataset: {len(current_polls)}")
        print(f"Poll date range: {pd.to_datetime(current_polls['date']).min().strftime('%Y-%m-%d')} to {pd.to_datetime(current_polls['date']).max().strftime('%Y-%m-%d')}")
        print("First 5 poll dates:")
        for date in pd.to_datetime(current_polls['date']).sort_values().head().tolist():
            print(f"  - {date.strftime('%Y-%m-%d')}")
    
    # Filter polls to only include those after the latest election
    polls_after_election = current_polls[pd.to_datetime(current_polls['date']) > pd.to_datetime(latest_election_date)]
    
    if len(polls_after_election) == 0:
        print(f"Error: No polls found after the latest election date ({latest_election_date})")
        print("Check that your poll dates are correctly formatted and newer than the latest election.")
        print("You may also need to update the historical_election_dates in the dataset class.")
        return
    else:
        # Use the polls after the election
        current_polls = polls_after_election
    
    if args.debug:
        print(f"Using {len(current_polls)} polls for nowcasting")
        print(f"Poll dates range: {pd.to_datetime(current_polls['date']).min().strftime('%Y-%m-%d')} to {pd.to_datetime(current_polls['date']).max().strftime('%Y-%m-%d')}")
    
    # Set election date to one day after the most recent poll
    latest_poll_date = pd.to_datetime(current_polls['date']).max()
    election_date = (latest_poll_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    if args.debug:
        print(f"Setting election date to one day after latest poll: {election_date}")
    
    # Run the nowcast using the updated data
    nowcast_ppc, coords, dims = elections_model.nowcast_party_support(
        current_polls=current_polls,
        latest_election_date=latest_election_date
    )
    
    # Plot the results - Comment out the call
    print("Generating plots...")
    # plot_nowcast_results(
    #     nowcast_ppc=nowcast_ppc, 
    #     current_polls=current_polls, 
    #     election_date=latest_election_date,
    #     title=f"Nowcast of Party Support Since {latest_election_date}",
    #     filename=os.path.join(output_dir, "nowcast_results.png")
    # )
    
    print(f"Nowcast completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Results saved to {output_dir}")
    print("==========================================\n")

def visualize(args):
    """Generate visualizations for a saved model"""
    try:
        if args.debug:
            print(f"Generating visualizations for model in {args.load_dir}")
        
        # Create visualization output directory
        # Make sure the parent directory exists
        model_dir = os.path.abspath(args.load_dir)
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")
            
        viz_dir = os.path.join(model_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load the model
        elections_model = load_model(
            directory=model_dir,
            debug=args.debug,
            # Pass arguments explicitly
            election_date=args.election_date, # Use command-line args
            baseline_timescales=args.baseline_timescale, 
            election_timescales=args.election_timescale 
        )
        
        if elections_model is None:
            raise ValueError(f"Failed to load model from {model_dir}")
        
        # Generate and save plots
        print("Generating model diagnostics and visualization plots...")
        plot_latent_popularity_vs_polls(elections_model, viz_dir)
        plot_latent_component_contributions(elections_model, viz_dir)
        plot_recent_polls(elections_model, viz_dir)

        print(f"Visualizations generation step complete. Results saved to {viz_dir}") # Adjusted print statement
        
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

def main(args=None):
    parser = argparse.ArgumentParser(description="Election Model CLI")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "predict-history", "retrodictive", "nowcast", "cross-validate", "viz", "visualize-data", "diagnose"],
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

    # Validate arguments
    if args.mode in ["train", "predict"]:
        if not args.election_date:
            parser.error("--election-date is required for train/predict modes")
        # Remove dataset requirement for train mode since data is loaded internally
        # if not args.dataset and args.mode == "train":
        #    parser.error("--dataset is required for train mode")
    
    # Add validation for visualize-data if needed (e.g., require election date)
    # if args.mode == "visualize-data" and not args.election_date:
    #     parser.error("--election-date is recommended for visualize-data mode to set context")

    if args.mode == "predict" and not args.load_dir:
        parser.error("--load-dir is required for predict mode")
        
    if args.mode == "viz" and not args.load_dir:
        parser.error("--load-dir is required for viz mode")
        
    if args.mode == "diagnose" and not args.load_dir:
        parser.error("--load-dir is required for diagnose mode")
    
    # Run the selected mode
    try:
        if args.mode == "train":
            fit_model(args)
        elif args.mode == "predict":
            predict(args)
        elif args.mode == "predict-history":
            predict_history(args)
        elif args.mode == "retrodictive":
            retrodictive(args)
        elif args.mode == "nowcast":
            nowcast(args)
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