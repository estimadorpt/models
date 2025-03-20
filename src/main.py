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

def print_forecast_summary(elections_model):
    """Print a summary of the forecast results"""
    try:
        # Extract the forecast data for the election date
        if hasattr(elections_model, "prediction") and elections_model.prediction is not None:
            latent_pop = elections_model.prediction.get('latent_popularity')
            if latent_pop is not None:
                # Get forecast for the last date (election day)
                election_day_forecast = latent_pop[:, :, -1, :]
                
                # Calculate mean and credible intervals
                mean_forecast = np.mean(election_day_forecast, axis=(0, 1))
                lower_bound = np.percentile(election_day_forecast, 2.5, axis=(0, 1))
                upper_bound = np.percentile(election_day_forecast, 97.5, axis=(0, 1))
                
                # Get party names
                parties = elections_model.prediction_coords['parties_complete']
                
                # Print forecast summary
                print("\nELECTION FORECAST SUMMARY FOR", elections_model.dataset.election_date)
                print("="*50)
                for i, party in enumerate(parties):
                    print(f"{party:15s}: {mean_forecast[i]:.1%} [{lower_bound[i]:.1%} - {upper_bound[i]:.1%}]")
                print("="*50)
    except Exception as e:
        print(f"Error printing forecast summary: {e}")


def fit_model(args):
    """Fit a model with the specified parameters"""
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Determine if we're doing a cutoff for retrodictive testing
    test_cutoff = None
    if args.cutoff_date:
        cutoff_date = pd.to_datetime(args.cutoff_date)
        target_date = pd.to_datetime(args.election_date)
        test_cutoff = target_date - cutoff_date

    # Create an instance of the elections model
    elections_model = ElectionsFacade(
        election_date=args.election_date,
        baseline_timescales=args.baseline_timescales,
        election_timescales=args.election_timescales,
        test_cutoff=test_cutoff,
        debug=args.debug,
    )
    
    # Store the output directory in the ElectionsFacade instance
    elections_model.output_dir = args.output_dir
    
    # Run inference
    try:
        if args.debug:
            print(f"Running inference with {args.draws} draws and {args.tune} tuning steps")
        
        elections_model.run_inference(draws=args.draws, tune=args.tune)
        
        # Analyze trace quality after inference
        if args.debug:
            print("\n=== TRACE QUALITY ANALYSIS ===")
            elections_model._analyze_trace_quality(elections_model.trace)
        
        # Save inference results to output directory
        print(f"Saving model to {args.output_dir}")
        elections_model.save_inference_results(directory=args.output_dir)
        
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data="Finished sampling".encode(encoding='utf-8'))
    except Exception as e:
        print(f"Error running inference: {e}")
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Error running inference: {e}".encode(encoding='utf-8'))
        return None
    
    return elections_model


def load_model(args):
    """Load a previously fitted model"""
    # Create an instance of the elections model
    elections_model = ElectionsFacade(
        election_date=args.election_date,
        baseline_timescales=args.baseline_timescales,
        election_timescales=args.election_timescales,
        debug=args.debug,
    )
    
    # Store the output directory in the ElectionsFacade instance
    elections_model.output_dir = args.output_dir
    
    try:
        # If a specific directory is provided, load from there
        if args.load_dir:
            load_path = args.load_dir
        else:
            load_path = args.output_dir
        
        # Use the load_inference_results method which will also call _analyze_trace_quality
        if args.debug:
            print(f"Loading inference results from {load_path}")
        elections_model.load_inference_results(directory=load_path)
        
        # Verify that trace was loaded properly
        if elections_model.trace is None:
            raise ValueError(f"Failed to load trace. Make sure the directory contains a valid trace.zarr file.")
        
        # Explicitly build the model - this is needed for generating forecasts after loading
        elections_model.model.model = elections_model.model.build_model()
        if args.debug:
            print("Model successfully built after loading!")
        
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data="Loaded saved inference results".encode(encoding='utf-8'))
        
        return elections_model
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")


def generate_forecast(elections_model, args):
    """Generate forecasts using the fitted model"""
    try:
        if args.debug:
            print(f"Generating forecast...")
        start_time = time.time()
        
        # Make sure the model is built properly
        if not hasattr(elections_model.model, 'model') or elections_model.model.model is None:
            print("Model not built yet, building model now...")
            elections_model.model.model = elections_model.model.build_model()
        
        # Generate forecast for the election date
        elections_model.generate_forecast(election_date=args.election_date)
        
        if args.debug:
            end_time = time.time()
            print(f"Forecast generation took {end_time - start_time:.2f} seconds")
        
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data="Generated forecast".encode(encoding='utf-8'))
        
        # Print forecast summary
        print_forecast_summary(elections_model)
        
    except Exception as e:
        print(f"Error generating forecast: {e}")
        if args.debug:
            traceback.print_exc()
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Error generating forecast: {e}".encode(encoding='utf-8'))


def cross_validate(args):
    """
    Perform cross-validation by fitting models for each past election,
    using only data available before that election
    """
    if args.debug:
        print("SAVE_MARKER: Entering cross_validate function")
        
    # Create output directory for cross-validation results
    cv_dir = os.path.join(args.output_dir, "cross_validation")
    if not os.path.exists(cv_dir):
        os.makedirs(cv_dir)
    
    # Get list of elections to validate
    # Create a dummy model to access the historical_election_dates
    temp_model = ElectionsFacade(
        election_date=args.election_date,
        baseline_timescales=args.baseline_timescales,
        election_timescales=args.election_timescales
    )
    
    # Skip the most recent election from cross-validation
    elections_to_validate = temp_model.dataset.historical_election_dates[1:]
    print(f"\nPerforming cross-validation for {len(elections_to_validate)} elections: {elections_to_validate}")
    
    # Store results for each election
    cv_results = []
    
    # For each election, fit a model using only data before that election
    for election_date in elections_to_validate:
        print(f"\n{'='*50}")
        print(f"Cross-validating for election: {election_date}")
        print(f"{'='*50}")
        
        # Create an election-specific output directory
        election_dir = os.path.join(cv_dir, election_date)
        if not os.path.exists(election_dir):
            os.makedirs(election_dir)
        
        # Set the args for this specific election
        this_args = argparse.Namespace(
            mode="fit",  # Explicitly set mode to fit
            election_date=election_date,
            cutoff_date=None,  # Don't apply extra cutoff
            baseline_timescales=args.baseline_timescales,
            election_timescales=args.election_timescales,
            output_dir=election_dir,
            debug=args.debug,
            draws=args.draws, 
            tune=args.tune,
            notify=False,
            fast=True  # Skip some expensive operations
        )
        
        if args.debug:
            print("SAVE_MARKER: cross_validate - before calling fit_model")
        
        # Fit model
        elections_model = fit_model(this_args)
        if elections_model is None:
            print(f"Failed to fit model for election {election_date}, skipping")
            continue
        
        if args.debug:
            print("SAVE_MARKER: cross_validate - before calling generate_forecast")
        
        # Generate forecast
        generate_forecast(elections_model, this_args)
        
        # Evaluate accuracy against actual results
        try:
            accuracy_metrics = evaluate_retrodictive_accuracy(elections_model, election_date)
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


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Election Forecast Model")
    
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
            # Generate forecast
            generate_forecast(elections_model, args)
            
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
        
        # Generate forecast
        generate_forecast(elections_model, args)
        
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