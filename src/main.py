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
from src.evaluation.retrodictive import evaluate_retrodictive_accuracy
from src.visualization.plots import save_plots, plot_model_components, plot_nowcast_results
from src.config import DEFAULT_BASELINE_TIMESCALE, DEFAULT_ELECTION_TIMESCALES

def fit_model(args):
    """Fit a new model with the specified parameters"""
    try:
        if args.debug:
            print(f"Fitting model for election date {args.election_date}")
            print(f"Using baseline timescales: {args.baseline_timescales}")
            print(f"Using election timescales: {args.election_timescales}")
        
        # Create the model output directory with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the elections model
        elections_model = ElectionsFacade(
            election_date=args.election_date,
            baseline_timescales=args.baseline_timescales,
            election_timescales=args.election_timescales,
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
        
        # Save the trace and model
        elections_model.save_inference_results(output_dir)
        
        # Create a symbolic link to the latest run
        latest_dir = os.path.join(os.path.dirname(args.output_dir), "latest")
        if os.path.exists(latest_dir):
            if os.path.islink(latest_dir):
                os.unlink(latest_dir)
            else:
                import shutil
                shutil.rmtree(latest_dir)
        
        try:
            os.symlink(output_dir, latest_dir, target_is_directory=True)
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

def load_model(directory, **kwargs):
    """
    Load a saved model
    
    Parameters:
    -----------
    directory : str
        Directory where the model is saved
    **kwargs : 
        Additional arguments to pass to ElectionsFacade
    """
    try:
        debug = kwargs.get('debug', False)
        if debug:
            print(f"Loading model from {directory}")
        
        # Initialize the elections model with the same parameters
        # Default values for timescales if not provided
        baseline_timescales = kwargs.get('baseline_timescales', [365])
        election_timescales = kwargs.get('election_timescales', [30, 15])
        election_date = kwargs.get('election_date', '2026-01-01')  # Default date
        
        # Ensure timescales are lists
        if not isinstance(baseline_timescales, list):
            baseline_timescales = [baseline_timescales]
        if not isinstance(election_timescales, list):
            election_timescales = [election_timescales]
            
        if debug:
            print(f"Using baseline_timescales: {baseline_timescales}")
            print(f"Using election_timescales: {election_timescales}")
        
        elections_model = ElectionsFacade(
            election_date=election_date,
            baseline_timescales=baseline_timescales,
            election_timescales=election_timescales,
            debug=debug
        )
        
        # Load the saved trace
        elections_model.load_inference_results(directory)
        
        # Build the model
        elections_model.model.model = elections_model.model.build_model()
        
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
    
    try:
        elections_model = load_model(
            directory=args.load_dir, 
            debug=args.debug,
            baseline_timescales=args.baseline_timescale if hasattr(args, 'baseline_timescale') else [365],
            election_timescales=args.election_timescale if hasattr(args, 'election_timescale') else [30, 15]
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
    
    # Plot the results
    print("Generating plots...")
    plot_nowcast_results(
        nowcast_ppc=nowcast_ppc, 
        current_polls=current_polls, 
        election_date=latest_election_date,
        title=f"Nowcast of Party Support Since {latest_election_date}",
        filename=os.path.join(output_dir, "nowcast_results.png")
    )
    
    print(f"Nowcast completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Results saved to {output_dir}")
    print("==========================================\n")

def cross_validate(args):
    """Perform cross-validation on historical elections"""
    try:
        if args.debug:
            print("Starting cross-validation...")
        
        # Create cross-validation directory
        cv_dir = os.path.join(args.output_dir, "cross_validation")
        os.makedirs(cv_dir, exist_ok=True)
        
        # Get all historical elections from the dataset
        from src.data.dataset import ElectionDataset
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
                
                # Evaluate accuracy
                accuracy_metrics = evaluate_retrodictive_accuracy(elections_model)
                
                # Store results
                cv_results.append({
                    'election_date': election_date,
                    **accuracy_metrics
                })
            except Exception as e:
                print(f"Error evaluating accuracy for election {election_date}: {e}")
                import traceback
                traceback.print_exc()
                
                # Add a placeholder result with NaN values
                cv_results.append({
                    'election_date': election_date,
                    'mae': np.nan,
                    'rmse': np.nan,
                    'mape': np.nan,
                    'error': str(e)
                })
                
            # Save plots
            if not args.fast:
                save_plots(elections_model, election_dir)
        
        # Summarize cross-validation results
        if cv_results:
            cv_df = pd.DataFrame(cv_results)
            print("\nCROSS-VALIDATION SUMMARY")
            print("="*50)
            print(cv_df)
            print("="*50)
            print(f"Mean absolute error: {cv_df['mae'].mean():.3f}")
            print(f"Root mean squared error: {cv_df['rmse'].mean():.3f}")
            
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

def main(args=None):
    parser = argparse.ArgumentParser(description="Election Model CLI")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "predict-history", "retrodictive", "nowcast", "cross-validate", "viz"],
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
    
    args = parser.parse_args(args)
    
    # Set random seed
    np.random.seed(args.seed)

    # Validate arguments
    if args.mode in ["train", "predict"]:
        if not args.election_date:
            parser.error("--election-date is required for train/predict modes")
        if not args.dataset and args.mode == "train":
            parser.error("--dataset is required for train mode")
    
    if args.mode == "predict" and not args.load_dir:
        parser.error("--load-dir is required for predict mode")
        
    # Run the selected mode
    try:
        if args.mode == "train":
            train(args)
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
            if not args.load_dir:
                parser.error("--load-dir is required for viz mode")
            visualize(args)
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