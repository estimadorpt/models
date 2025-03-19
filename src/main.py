import os
import argparse
import arviz
import requests
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import numpy as np

from src.models.elections_facade import ElectionsFacade


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


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Election Model")
    parser.add_argument(
        "--draws", type=int, default=1000, help="Number of posterior samples to draw"
    )
    parser.add_argument(
        "--tune", type=int, default=1000, help="Number of tuning steps for NUTS sampler"
    )
    parser.add_argument(
        "--election-date", type=str, default="2024-03-10", help="Target election date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--baseline-timescales", type=int, nargs="+", default=[365], 
        help="Timescales for baseline GP in days"
    )
    parser.add_argument(
        "--election-timescales", type=int, nargs="+", default=[60], 
        help="Timescales for election-specific GP in days"
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None, 
        help="Optional weights for GP components"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, 
        help="Directory to save outputs (default: timestamped directory in 'outputs/')"
    )
    parser.add_argument(
        "--load-results", action="store_true", default=False, 
        help="Load previously saved inference results instead of running inference"
    )
    parser.add_argument(
        "--load-dir", type=str, default=None, 
        help="Directory to load results from (if --load-results is specified)"
    )
    parser.add_argument(
        "--notify", action="store_true", default=False,
        help="Send notifications via ntfy.sh"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable detailed diagnostic output"
    )
    parser.add_argument(
        "--fast", action="store_true", default=False,
        help="Skip plots and other time-consuming operations for maximum speed"
    )
    
    args = parser.parse_args()
    
    # Configure output directory
    if args.output_dir is None:
        # Use a timestamped directory if not specified
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        args.output_dir = os.path.join("outputs", timestamp)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
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
    
    # Step 1: Either load results or run inference
    if args.load_results:
        # If a specific directory is provided, load from there
        if args.load_dir:
            # Store load directory to handle prediction.zarr correctly
            elections_model.load_dir = args.load_dir
            elections_model.load_inference_results(directory=args.load_dir)
        else:
            elections_model.load_inference_results()
        
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data="Loaded saved inference results".encode(encoding='utf-8'))
    else:
        # Run inference
        try:
            elections_model.run_inference(draws=args.draws, tune=args.tune)
            # Save inference results to output directory
            if not args.fast:
                elections_model.save_inference_results(directory=args.output_dir)
            if args.notify:
                requests.post("https://ntfy.sh/bc-estimador",
                    data="Finished sampling".encode(encoding='utf-8'))
        except Exception as e:
            print(f"Error running inference: {e}")
            if args.notify:
                requests.post("https://ntfy.sh/bc-estimador",
                    data=f"Error running inference: {e}".encode(encoding='utf-8'))
            return
    
    # Step 2: Generate forecast once
    try:
        print("Generating forecast...")
        elections_model.generate_forecast()
        
        # Save inference results including prediction to output directory
        if not args.fast:
            elections_model.save_inference_results(directory=args.output_dir)
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data="Generated forecast".encode(encoding='utf-8'))
    except Exception as e:
        print(f"Error generating forecast: {e}")
        if args.notify:
            requests.post("https://ntfy.sh/bc-estimador",
                data=f"Error generating forecast: {e}".encode(encoding='utf-8'))
    
    # Step 3: Save plots (skip in fast mode)
    if not args.fast:
        try:
            save_plots(elections_model, args.output_dir)
            if args.notify:
                requests.post("https://ntfy.sh/bc-estimador",
                    data="Finished analysis and saved plots".encode(encoding='utf-8'))
        except Exception as e:
            print(f"Error saving plots: {e}")
            if args.notify:
                requests.post("https://ntfy.sh/bc-estimador",
                    data=f"Error saving plots: {e}".encode(encoding='utf-8'))
    else:
        print("Fast mode: Skipped plot generation")
        
    # Step 4: Print forecast summary
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


if __name__ == "__main__":
    main()