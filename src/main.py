import os
import argparse
import arviz
import requests
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import numpy as np
import pandas as pd
import time
import traceback
import sys

# Debug module loading
print(f"DEBUG: Main module {__name__} is being loaded")

from src.models.elections_facade import ElectionsFacade
from src.evaluation.retrodictive import evaluate_retrodictive_accuracy

# Global flag to track if main has already run
_MAIN_HAS_RUN = False

def save_plots(elections_model, output_dir):
    """
    Save various plots from the model
    
    Parameters:
    -----------
    elections_model : ElectionsFacade
        The elections model to generate plots from
    output_dir : str
        Directory to save plots in
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Save retrodictive check plot
    try:
        retro_fig = elections_model.plot_retrodictive_check()
        retro_fig.savefig(os.path.join(plots_dir, "latent_popularity_evolution_with_observed_and_results.png"))
        plt.close(retro_fig)
    except Exception as e:
        print(f"Error saving retrodictive plot: {e}")
    
    # Save forecast plot - our new implementation returns a list of figures
    try:
        forecast_figs = elections_model.plot_forecast()
        # Check if we got a list or a single figure
        if isinstance(forecast_figs, list):
            for i, fig in enumerate(forecast_figs):
                fig.savefig(os.path.join(plots_dir, f"forecast_plot_{i}.png"))
                plt.close(fig)
        else:
            # In case it returns a single figure
            forecast_figs.savefig(os.path.join(plots_dir, "latent_popularity_evolution_last_year.png"))
            plt.close(forecast_figs)
    except Exception as e:
        print(f"Error saving forecast plots: {e}")
    
    # Save party correlations
    try:
        corr_fig = elections_model.plot_party_correlations()
        corr_fig.savefig(os.path.join(plots_dir, "party_correlations.png"))
        plt.close(corr_fig)
    except Exception as e:
        print(f"Error saving party correlations plot: {e}")
    
    # Save predictive accuracy
    try:
        acc_fig = elections_model.plot_predictive_accuracy()
        acc_fig.savefig(os.path.join(plots_dir, "polling_accuracy.png"))
        plt.close(acc_fig)
    except Exception as e:
        print(f"Error saving predictive accuracy plot: {e}")
    
    # Save house effects for each pollster
    house_effects_dir = os.path.join(plots_dir, "house_effects")
    if not os.path.exists(house_effects_dir):
        os.makedirs(house_effects_dir)
        
    for pollster in elections_model.dataset.unique_pollsters:
        try:
            house_fig = elections_model.plot_house_effects(pollster)
            house_fig.savefig(os.path.join(house_effects_dir, f"house_effects_{pollster.replace('/', '_')}.png"))
            plt.close(house_fig)
        except Exception as e:
            print(f"Error plotting house effects for {pollster}: {e}")
    
    # Plot individual model components
    components_dir = os.path.join(plots_dir, "components")
    if not os.path.exists(components_dir):
        os.makedirs(components_dir)
        
    try:
        plot_model_components(elections_model, components_dir)
    except Exception as e:
        print(f"Error plotting model components: {e}")


def plot_model_components(elections_model, output_dir):
    """
    Plot various model components
    
    Parameters:
    -----------
    elections_model : ElectionsFacade
        The elections model to generate plots from
    output_dir : str
        Directory to save plots in
    """
    trace = elections_model.trace
    
    # Party baseline
    try:
        party_baseline = trace.posterior.party_baseline.mean(("chain", "draw"))
        plt.figure(figsize=(10, 6))
        # Convert xarray DataArray to pandas DataFrame for bar plotting
        party_baseline_df = party_baseline.to_dataframe(name="value").reset_index()
        plt.bar(party_baseline_df["parties_complete"], party_baseline_df["value"])
        plt.title("Party Baseline by Party")
        plt.xticks(rotation=45)
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "party_baseline_by_party.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting party baseline: {e}")
    
    # Election party baseline
    try:
        election_party_baseline = trace.posterior.election_party_baseline.mean(("chain", "draw"))
        plt.figure(figsize=(12, 8))
        # Convert to long-form DataFrame for easier plotting
        df = election_party_baseline.to_dataframe(name="value").reset_index()
        
        # Plot as line plot with hue for parties
        sns.lineplot(data=df, x="elections", y="value", hue="parties_complete", marker="o")
        plt.title("Election Party Baseline by Party and Election")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "election_party_baseline_by_party_and_election.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting election party baseline: {e}")
    
    # Poll bias
    try:
        poll_bias = trace.posterior.poll_bias.mean(("chain", "draw"))
        plt.figure(figsize=(10, 6))
        # Convert xarray DataArray to pandas DataFrame for bar plotting
        poll_bias_df = poll_bias.to_dataframe(name="value").reset_index()
        plt.bar(poll_bias_df["parties_complete"], poll_bias_df["value"])
        plt.title("Poll Bias by Party")
        plt.xticks(rotation=45)
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "poll_bias_by_party.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting poll bias: {e}")
    
    # House effects
    try:
        house_effects = trace.posterior.house_effects.mean(("chain", "draw"))
        plt.figure(figsize=(12, 8))
        # Convert to long-form DataFrame
        df = house_effects.to_dataframe(name="value").reset_index()
        
        # Plot as heatmap which is better for this type of data
        pivot_df = df.pivot(index="pollsters", columns="parties_complete", values="value")
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0)
        plt.title("House Effects by Party and Pollsters")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "house_effects_by_party_and_pollsters.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting house effects: {e}")
    
    # House election effects
    try:
        house_election_effects = trace.posterior.house_election_effects.mean(("chain", "draw"))
        # For 3D data, let's create multiple plots, one for each election
        df = house_election_effects.to_dataframe(name="value").reset_index()
        
        # Group by election and create separate plots
        for election in df["elections"].unique():
            election_df = df[df["elections"] == election]
            pivot_df = election_df.pivot(index="pollsters", columns="parties_complete", values="value")
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0)
            plt.title(f"House Election Effects - Election {election}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"house_election_effects_election_{election}.png"))
            plt.close()
        
        # Also create a summary plot
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x="elections", y="value", hue="parties_complete")
        plt.title("House Election Effects by Party and Election")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "house_election_effects_by_party_and_election.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting house election effects: {e}")
    
    # Party time effect weighted
    try:
        party_time_effect_weighted = trace.posterior.party_time_effect_weighted.mean(("chain", "draw"))
        plt.figure(figsize=(12, 8))
        # Convert to long-form DataFrame
        df = party_time_effect_weighted.to_dataframe(name="value").reset_index()
        
        # Plot as line plot
        sns.lineplot(data=df, x="countdown", y="value", hue="parties_complete")
        plt.title("Party Time Effect Weighted by Party and Countdown")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "party_time_effect_weighted_by_party_and_countdown.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting party time effect weighted: {e}")
    
    # Latent popularity
    try:
        latent_popularity = trace.posterior.latent_popularity.mean(("chain", "draw"))
        plt.figure(figsize=(12, 8))
        # Convert to long-form DataFrame
        df = latent_popularity.to_dataframe(name="value").reset_index()
        
        # Plot as line plot
        sns.lineplot(data=df, x="observations", y="value", hue="parties_complete")
        plt.title("Latent Popularity by Party over Time")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latent_popularity_by_party_over_time.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting latent popularity: {e}")
    
    # Noisy popularity
    try:
        noisy_popularity = trace.posterior.noisy_popularity.mean(("chain", "draw"))
        plt.figure(figsize=(12, 8))
        # Convert to long-form DataFrame
        df = noisy_popularity.to_dataframe(name="value").reset_index()
        
        # Plot as line plot
        sns.lineplot(data=df, x="observations", y="value", hue="parties_complete")
        plt.title("Noisy Popularity by Party over Time")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "noisy_popularity_by_party_over_time.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting noisy popularity: {e}")
    
    # Election party time effect weighted
    try:
        election_party_time_effect_weighted = trace.posterior.election_party_time_effect_weighted.mean(("chain", "draw"))
        # For this 3D array, let's select a specific countdown value (0)
        eptew_at_zero = election_party_time_effect_weighted.isel(countdown=0)
        
        # Convert to long-form DataFrame
        df = eptew_at_zero.to_dataframe(name="value").reset_index()
        
        # Plot as line plot
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x="elections", y="value", hue="parties_complete", marker="o")
        plt.title("Election Party Time Effect Weighted by Party and Election (Countdown=0)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "election_party_time_effect_weighted_by_party_and_election.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting election party time effect weighted: {e}")


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
        weights=args.weights,
        test_cutoff=test_cutoff,
        debug=args.debug,
    )
    
    # Store the output directory in the ElectionsFacade instance
    elections_model.output_dir = args.output_dir
    
    # Run inference
    try:
        elections_model.run_inference(draws=args.draws, tune=args.tune)
        
        # Save inference results to output directory - always save regardless of fast mode
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
        weights=args.weights,
        debug=args.debug,
    )
    
    # Store the output directory in the ElectionsFacade instance
    elections_model.output_dir = args.output_dir
    
    # If a specific directory is provided, load from there
    if args.load_dir:
        # Load only the trace, prior, and posterior
        try:
            trace_path = os.path.join(args.load_dir, "trace.zarr")
            if os.path.exists(trace_path):
                print(f"Loading trace from {trace_path}")
                elections_model.trace = arviz.from_zarr(trace_path)
            else:
                raise ValueError(f"Trace file not found at {trace_path}")
                
            # Optional: load prior and posterior if they exist
            prior_path = os.path.join(args.load_dir, "prior_check.zarr")
            old_prior_path = os.path.join(args.load_dir, "prior.zarr")
            if os.path.exists(prior_path):
                elections_model.prior = arviz.from_zarr(prior_path)
            elif os.path.exists(old_prior_path):
                elections_model.prior = arviz.from_zarr(old_prior_path)
                
            posterior_path = os.path.join(args.load_dir, "posterior_check.zarr")
            old_posterior_path = os.path.join(args.load_dir, "posterior.zarr")
            if os.path.exists(posterior_path):
                elections_model.posterior = arviz.from_zarr(posterior_path)
            elif os.path.exists(old_posterior_path):
                elections_model.posterior = arviz.from_zarr(old_posterior_path)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    else:
        elections_model.load_inference_results()
    
    # Verify that trace was loaded properly
    if elections_model.trace is None:
        raise ValueError(f"Failed to load trace from {args.load_dir}. Make sure the directory contains a valid trace.zarr file.")
    
    # Explicitly build the model - this is needed for generating forecasts after loading
    elections_model.model.model = elections_model.model.build_model()
    print("Model successfully built after loading!")
    
    if args.notify:
        requests.post("https://ntfy.sh/bc-estimador",
            data="Loaded saved inference results".encode(encoding='utf-8'))
    
    return elections_model


def generate_forecast(elections_model, args):
    """Generate forecasts using the fitted model"""
    try:
        print(f"Generating forecast in {args.mode} mode...")
        start_time = time.time()
        
        # Make sure the model is built properly
        if not hasattr(elections_model.model, 'model') or elections_model.model.model is None:
            print("Model not built yet, building model now...")
            elections_model.model.model = elections_model.model.build_model()
        
        # Use the specified election date or the default from args
        elections_model.generate_forecast(election_date=args.election_date)
        
        end_time = time.time()
        print(f"Forecast generation took {end_time - start_time:.2f} seconds")
        
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data="Generated forecast".encode(encoding='utf-8'))
        
        # Print forecast summary
        print_forecast_summary(elections_model)
        
    except Exception as e:
        print(f"Error generating forecast: {e}")
        import traceback
        traceback.print_exc()
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Error generating forecast: {e}".encode(encoding='utf-8'))


def cross_validate(args):
    """
    Perform cross-validation by fitting models for each past election,
    using only data available before that election
    """
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
        
        # Fit a model using data up to a few days before this election
        election_datetime = pd.to_datetime(election_date)
        cutoff_date = (election_datetime - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Set the args for this specific election
        this_args = argparse.Namespace(
            mode="fit",  # Explicitly set mode to fit
            election_date=election_date,
            cutoff_date=None,  # Don't apply extra cutoff
            baseline_timescales=args.baseline_timescales,
            election_timescales=args.election_timescales,
            weights=args.weights,
            output_dir=election_dir,
            debug=args.debug,
            draws=args.draws, 
            tune=args.tune,
            notify=False,
            fast=True  # Skip some expensive operations
        )
        
        print("SAVE_MARKER: cross_validate - before calling fit_model")
        # Fit model
        elections_model = fit_model(this_args)
        if elections_model is None:
            print(f"Failed to fit model for election {election_date}, skipping")
            continue
        
        print("SAVE_MARKER: cross_validate - before calling generate_forecast")
        # Generate forecast
        generate_forecast(elections_model, this_args)
        
        # Evaluate accuracy against actual results
        accuracy_metrics = evaluate_retrodictive_accuracy(elections_model, election_date)
        cv_results.append({
            'election_date': election_date,
            **accuracy_metrics
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
    print("\nSAVE_MARKER: Entering main function")
    
    # Debug to see where main is being called from
    print("DEBUG: main() call stack:")
    current_stack = traceback.extract_stack()
    for frame in current_stack[:-1]:  # Skip the current frame
        print(f"  File {frame.filename}, line {frame.lineno}, in {frame.name}")
    
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
    parser.add_argument('--baseline-timescales', type=float, nargs='+', default=[180, 365, 730],
                      help='Baseline timescales for Gaussian process')
    parser.add_argument('--election-timescales', type=float, nargs='+', default=[60, 30, 15],
                      help='Election timescales for Gaussian process')
    parser.add_argument('--weights', type=float, nargs='+', default=[0.5, 0.3, 0.2],
                      help='Weights for each timescale component')
    
    # MCMC parameters
    parser.add_argument('--draws', type=int, default=1000,
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
    
    # Store mode in args to pass it to generate_forecast
    args.mode = args.mode
    
    # Execute the requested mode
    if args.mode == "fit":
        print("SAVE_MARKER: main function - fit branch")
        # Fit the model with the specified parameters
        # Note: fit_model already saves the model, no need to save again
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
        print("SAVE_MARKER: main function - load branch")
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
        print("SAVE_MARKER: main function - cross-validate branch")
        # Perform cross-validation
        cv_results = cross_validate(args)


if __name__ == "__main__":
    main()