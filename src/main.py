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

from src.models.elections_facade import ElectionsFacade
from src.evaluation.retrodictive import evaluate_retrodictive_accuracy
from src.visualization.plots import save_plots, plot_model_components
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

def load_model(args):
    """Load a saved model"""
    try:
        if args.debug:
            print(f"Loading model from {args.load_dir}")
        
        # Initialize the elections model with the same parameters
        elections_model = ElectionsFacade(
            election_date=args.election_date,
            baseline_timescales=args.baseline_timescales,
            election_timescales=args.election_timescales,
            debug=args.debug
        )
        
        # Load the saved trace
        elections_model.load_inference_results(args.load_dir)
        
        # Build the model
        elections_model.model.model = elections_model.model.build_model()
        
        return elections_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        if args.debug:
            traceback.print_exc()
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Error loading model: {e}".encode(encoding='utf-8'))
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

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Election Analysis Model")
    
    # Required arguments
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['fit', 'load', 'cross-validate'],
                      help='Mode to run the model in')
    
    # File and directory arguments
    parser.add_argument('--output-dir', type=str, default='outputs/latest',
                      help='Directory to save outputs')
    parser.add_argument('--load-dir', type=str, default=None,
                      help='Directory to load saved model from')
    
    # Model parameters
    parser.add_argument('--election-date', type=str, required=True,
                      help='Target election date (YYYY-MM-DD)')
    parser.add_argument('--baseline-timescales', type=float, nargs='+', default=[DEFAULT_BASELINE_TIMESCALE],
                      help='Baseline timescale for annual cycle (in days)')
    parser.add_argument('--election-timescales', type=float, nargs='+', default=DEFAULT_ELECTION_TIMESCALES,
                      help='Election timescales for pre-campaign (30 days) and official campaign (15 days)')
    
    # MCMC parameters
    parser.add_argument('--draws', type=int, default=2000,
                      help='Number of samples to draw')
    parser.add_argument('--tune', type=int, default=1000,
                      help='Number of tuning steps')
    
    # Cross-validation parameters
    parser.add_argument('--cutoff-date', type=str, default=None,
                      help='Cutoff date for retrodictive testing (YYYY-MM-DD)')
    
    # Flags
    parser.add_argument('--fast', action='store_true',
                      help='Run in fast mode (skip plots and diagnostics)')
    parser.add_argument('--notify', action='store_true',
                      help='Send notifications on completion')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    
    args = parser.parse_args()
    
    # Execute the requested mode
    if args.mode == "fit":
        if args.debug:
            print("Running in fit mode")
            
        # Fit the model with the specified parameters
        elections_model = fit_model(args)
        if elections_model is not None:
            # Save plots (skip in fast mode)
            if not args.fast:
                save_plots(elections_model, args.output_dir)
                if args.notify:
                    requests.post("https://ntfy.sh/bc-estimador",
                        data="Finished analysis and saved plots".encode(encoding='utf-8'))
        
    elif args.mode == "load":
        if args.debug:
            print("Running in load mode")
            
        # Load the model from saved results
        elections_model = load_model(args)
        
        # Save plots (skip in fast mode)
        if not args.fast:
            save_plots(elections_model, args.output_dir)
            if args.notify:
                requests.post("https://ntfy.sh/bc-estimador",
                    data="Finished analysis and saved plots".encode(encoding='utf-8'))
    
    elif args.mode == "cross-validate":
        if args.debug:
            print("Running in cross-validate mode")
            
        # Perform cross-validation
        cross_validate(args)


if __name__ == "__main__":
    main()