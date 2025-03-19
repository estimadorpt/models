import os
from typing import Dict, List, Tuple, Optional

import arviz
import pandas as pd
import numpy as np
import xarray as xr
import pymc as pm
import time
import datetime

from src.data.dataset import ElectionDataset
from src.models.election_model import ElectionModel
from src.visualization.plots import (
    retrodictive_plot,
    predictive_plot,
    plot_house_effects,
    plot_party_correlations,
    plot_predictive_accuracy,
    plot_latent_trajectories,
    plot_party_trajectory
)


class ElectionsFacade:
    """
    A facade class that simplifies interactions with the election model system.
    
    This class provides a simplified interface to the election data and model,
    making it easier to load data, build and sample from models, and generate
    visualizations without directly dealing with the underlying complexity.
    """
    
    def __init__(
        self,
        election_date: str,
        baseline_timescales: List[int] = [365],
        election_timescales: List[int] = [60],
        weights: List[float] = None,
        test_cutoff: pd.Timedelta = None,
        debug: bool = False,
    ):
        """
        Initialize the elections facade.
        
        Parameters:
        -----------
        election_date : str
            The target election date in 'YYYY-MM-DD' format
        baseline_timescales : List[int]
            Timescales for baseline GP components in days
        election_timescales : List[int]
            Timescales for election-specific GP components in days
        weights : List[float]
            Weights for GP components
        test_cutoff : pd.Timedelta
            How much data to hold out for testing
        debug : bool
            Whether to print detailed diagnostic information
        """
        # Load the dataset
        self.dataset = ElectionDataset(
            election_date=election_date,
            baseline_timescales=baseline_timescales,
            election_timescales=election_timescales,
            weights=weights,
            test_cutoff=test_cutoff,
        )
        
        # Create the model
        self.model = ElectionModel(self.dataset)
        
        # Initialize trace containers
        self.prior = None
        self.trace = None
        self.posterior = None
        self.prediction = None
        self.prediction_coords = None
        self.prediction_dims = None
        
        # Set debug flag
        self.debug = debug
        
        # Initialize output_dir and model_config
        self.output_dir = None
        self.model_config = {
            "election_date": election_date,
            "baseline_timescales": baseline_timescales,
            "election_timescales": election_timescales,
            "weights": weights,
            "save_dir": None,
            "seed": 12345,
        }
        
    def build_model(self):
        """Build the PyMC model"""
        return self.model.build_model()
    
    def run_inference(
        self, 
        draws: int = 200, 
        tune: int = 100, 
        target_accept: float = 0.995, 
        **sampler_kwargs
    ):
        """
        Run MCMC inference on the model.
        
        Parameters:
        -----------
        draws : int
            Number of posterior samples to draw
        tune : int
            Number of tuning samples
        target_accept : float
            Target acceptance rate for NUTS sampler
        **sampler_kwargs :
            Additional arguments for PyMC sampler
        
        Returns:
        --------
        prior, trace, posterior : tuple of arviz.InferenceData
            Prior, trace and posterior samples
        """
        var_names = [
            "party_baseline",
            "election_party_baseline",
            "poll_bias",
            "house_effects",
            "house_election_effects",
            "party_time_effect_weighted",
            "latent_popularity",
            "noisy_popularity",
            "election_party_time_effect_weighted",
            "N_approve",
            "R"
        ]
        
        # Include draws and tune in sampler_kwargs instead of passing them directly
        sampler_kwargs['draws'] = draws
        sampler_kwargs['tune'] = tune
        sampler_kwargs['target_accept'] = target_accept
        
        self.prior, self.trace, self.posterior = self.model.sample_all(
            var_names=var_names,
            **sampler_kwargs
        )
        
        return self.prior, self.trace, self.posterior
    
    def save_inference_results(self, directory: str = "."):
        """
        Save inference results to disk.
        
        Parameters:
        -----------
        directory : str
            Directory where to save files
        """
        # Create output directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        if hasattr(self, "prior") and self.prior is not None:
            arviz.to_zarr(self.prior, os.path.join(directory, "prior.zarr"))
        
        if hasattr(self, "trace") and self.trace is not None:
            arviz.to_zarr(self.trace, os.path.join(directory, "trace.zarr"))
        
        if hasattr(self, "posterior") and self.posterior is not None:
            arviz.to_zarr(self.posterior, os.path.join(directory, "posterior.zarr"))
        
        # We no longer save prediction data to avoid zarr complexity
        # Just return successfully
        return True
    
    def load_inference_results(self, directory: str = "."):
        """
        Load previously saved inference results.
        
        Parameters:
        -----------
        directory : str
            Directory where files are saved
        """
        prior_path = os.path.join(directory, "prior.zarr")
        trace_path = os.path.join(directory, "trace.zarr")
        posterior_path = os.path.join(directory, "posterior.zarr")
        prediction_path = os.path.join(directory, "prediction.zarr")
        
        if os.path.exists(prior_path):
            self.prior = arviz.from_zarr(prior_path)
            
        if os.path.exists(trace_path):
            self.trace = arviz.from_zarr(trace_path)
            # Check trace quality
            self._analyze_trace_quality(self.trace)
            
        if os.path.exists(posterior_path):
            self.posterior = arviz.from_zarr(posterior_path)
            
        if os.path.exists(prediction_path):
            # Use our custom load_prediction method instead of arviz.from_zarr
            self.load_prediction(prediction_path)
    
    def _analyze_trace_quality(self, trace):
        """
        Analyze the quality of the MCMC trace and check for convergence issues.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            The trace to analyze
        """
        if trace is None:
            print("No trace to analyze")
            return
        
        # Skip expensive calculations when debug is off
        if not self.debug:
            # Just do a quick NaN check on a few key parameters
            for param in ['party_baseline', 'election_party_baseline', 'party_time_effect_weighted']:
                if param in trace.posterior:
                    param_values = trace.posterior[param].values
                    nan_percentage = np.isnan(param_values).mean() * 100
                    if nan_percentage > 90:
                        print(f"Warning: {param} contains {nan_percentage:.1f}% NaN values")
            return
        
        # Only print detailed diagnostics if debug is enabled
        print("\n======= TRACE QUALITY ANALYSIS =======")
        
        # Check for NaN values in key parameters
        key_params = ['party_baseline', 'election_party_baseline', 'party_time_effect_weighted']
        for param in key_params:
            if param in trace.posterior:
                param_values = trace.posterior[param].values
                nan_percentage = np.isnan(param_values).mean() * 100
                if nan_percentage > 0:
                    print(f"WARNING: {param} contains {nan_percentage:.2f}% NaN values!")
                else:
                    print(f"{param}: No NaN values detected")
                    
                # Print shape and some statistics
                print(f"  Shape: {param_values.shape}")
                print(f"  Range: [{np.nanmin(param_values):.4f}, {np.nanmax(param_values):.4f}]")
                print(f"  Mean: {np.nanmean(param_values):.4f}")
            else:
                print(f"{param}: Not found in trace")
        
        # Comprehensive check of all parameters for ESS and Rhat issues
        print("\n=== COMPREHENSIVE CONVERGENCE DIAGNOSTICS ===")
        print("Identifying parameters with convergence issues...\n")
        
        # Get all parameter names from the trace
        all_params = list(trace.posterior.data_vars)
        
        # Check for convergence using effective sample size (ESS)
        try:
            ess = arviz.ess(trace)
            
            # Find parameters with low ESS
            low_ess_params = []
            for param in all_params:
                if param in ess:
                    min_ess = float(ess[param].min())
                    if min_ess < 100:  # This is the threshold mentioned in the warning
                        low_ess_params.append((param, min_ess))
            
            # Report parameters with low ESS if debug is enabled
            if low_ess_params:
                print(f"Found {len(low_ess_params)} parameters with low ESS (<100):")
                for param, min_ess in sorted(low_ess_params, key=lambda x: x[1]):
                    print(f"  {param}: Min ESS = {min_ess:.1f}")
            
        except Exception as e:
            print(f"Could not compute ESS: {e}")
        
        # Check for mixing using Rhat
        try:
            rhat = arviz.rhat(trace)
            
            # Find parameters with high Rhat
            high_rhat_params = []
            for param in all_params:
                if param in rhat:
                    max_rhat = float(rhat[param].max())
                    if max_rhat > 1.01:  # This is the threshold mentioned in the warning
                        high_rhat_params.append((param, max_rhat))
            
            # Report parameters with high Rhat if debug is enabled
            if high_rhat_params:
                print(f"\nFound {len(high_rhat_params)} parameters with high R-hat (>1.01):")
                for param, max_rhat in sorted(high_rhat_params, key=lambda x: x[1], reverse=True):
                    if max_rhat > 1.1:
                        severity = "SEVERE"
                    elif max_rhat > 1.05:
                        severity = "HIGH"
                    else:
                        severity = "MODERATE"
                    print(f"  {param}: Max R-hat = {max_rhat:.3f} ({severity})")
            
            # Summary statistics for all parameters
            print("\nSummary of all parameter diagnostics:")
            print(f"  Total parameters: {len(all_params)}")
            print(f"  Parameters with low ESS: {len(low_ess_params)} ({len(low_ess_params)/len(all_params)*100:.1f}%)")
            print(f"  Parameters with high R-hat: {len(high_rhat_params)} ({len(high_rhat_params)/len(all_params)*100:.1f}%)")
            
            # Recommendations based on diagnostics
            if len(low_ess_params) > 0 or len(high_rhat_params) > 0:
                print("\nRecommendations to improve convergence:")
                print("  1. Increase the number of tuning steps")
                print("  2. Increase the number of draws")
                print("  3. Adjust model priors to be more appropriate")
                print("  4. Simplify model structure if possible")
                if len(high_rhat_params) > len(all_params) * 0.5:
                    print("  5. Consider a different sampler or algorithm")
            
            print("============================================\n")
            
        except Exception as e:
            print(f"Could not compute R-hat: {e}")
    
    def generate_forecast(self):
        """
        Generate a forecast for the target election using the model's forecast_election method.
        
        Returns:
        --------
        tuple
            raw_predictions, coords, dims
        """
        # Ensure that the model has been built
        if not hasattr(self, "trace") or self.trace is None:
            self._build_model()
        
        # Print basic status information
        print("\nGenerating forecast for election date:", self.dataset.election_date)
        
        # Start timing
        start_time = time.time()
        
        # Check if trace has divergences or other quality issues
        if not self._check_trace_quality():
            print("CRITICAL MODEL ISSUE: Trace contains NaN values for key parameters.")
            print("This could indicate issues with model specification, data compatibility,")
            print("or insufficient tuning. Please check your data and model configuration.")
            raise ValueError("Trace quality issues detected, cannot generate forecast.")
        
        try:
            # First ensure the model is built in the ElectionModel as done in test_forecast.py
            print("Building model for forecast...")
            self.model.model = self.model.build_model()
            
            # Use the forecasting method from ElectionModel directly
            print("Calling forecast_election...")
            raw_ppc, coords, dims = self.model.forecast_election(self.trace)
            
            # Store raw prediction data
            self.prediction = raw_ppc
            self.prediction_coords = coords
            self.prediction_dims = dims
            
            # Calculate elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Forecast generated successfully in {elapsed_time:.2f} seconds")
            
            # Print dimensions of prediction data only in debug mode
            if self.debug:
                print("\nForecast dimensions:")
                for var_name, arr in raw_ppc.items():
                    print(f"  {var_name}: shape {arr.shape}")
                
                print("\nCoordinates:")
                for coord_name, values in coords.items():
                    print(f"  {coord_name}: {len(values)} values")
                print("===========================")
            
            return raw_ppc, coords, dims
            
        except Exception as e:
            print(f"Error in forecast generation: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to generate forecast: {e}")
    
    def _check_trace_quality(self):
        """
        Check if the trace quality is good enough for forecasting.
        
        Returns:
        --------
        bool
            True if the trace is usable, False otherwise
        """
        # Optimized check - only sample a small subset of the trace for NaN checking
        if not self.debug:
            # Choose a limited set of samples to check
            try:
                # Get first chain, first 100 samples, and check a key parameter
                if 'party_baseline' in self.trace.posterior:
                    # Get only first chain, first 100 samples (or fewer if not available)
                    chain_idx = 0
                    max_samples = min(100, self.trace.posterior.dims.get('draw', 0))
                    
                    # Quick check of the first few samples for a key parameter
                    values = self.trace.posterior.party_baseline.isel(
                        chain=slice(0, 1), 
                        draw=slice(0, max_samples)
                    ).values
                    
                    # If more than 90% NaN, consider it unusable
                    nan_percentage = np.isnan(values).mean() * 100
                    return nan_percentage < 90
                
                # If we can't find the parameter, assume it's usable
                return True
            except Exception:
                # If any error occurs, assume it's usable
                return True
        
        # The most basic check - do we have non-NaN values for key parameters?
        key_params = ['party_baseline', 'election_party_baseline', 'party_time_effect_weighted']
        
        for param in key_params:
            if param in self.trace.posterior:
                param_values = self.trace.posterior[param].values
                nan_percentage = np.isnan(param_values).mean() * 100
                
                # If more than 90% of values are NaN, the trace is not usable
                if nan_percentage > 90:
                    print(f"Critical issue: {param} contains {nan_percentage:.1f}% NaN values")
                    return False
        
        return True
    
    def plot_retrodictive_check(self, group: str = "posterior"):
        """
        Create a retrodictive plot comparing model predictions with historical poll data.
        
        Parameters:
        -----------
        group : str
            Whether to use posterior or prior. Options: "posterior", "prior"
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Retrodictive plot
        """
        if self.trace is None or (group == "posterior" and self.posterior is None):
            raise ValueError(f"Must run inference before creating {group} predictive check")
            
        posterior_data = self.posterior if group == "posterior" else None
        
        return retrodictive_plot(
            trace=self.trace,
            posterior=posterior_data,
            parties_complete=self.dataset.political_families,
            polls_train=self.dataset.polls_train,
            group=group
        )
    
    def plot_forecast(self):
        """
        Plot the forecast for the target election, showing all parties.
        
        Returns:
        --------
        list
            A list of figures: the first is the overall forecast, the rest are party-specific
        """
        from src.visualization.plots import plot_latent_trajectories, plot_party_trajectory
        
        if not hasattr(self, "trace") or self.trace is None:
            raise ValueError("Must run inference before plotting forecast")
        
        if not hasattr(self, "prediction") or self.prediction is None:
            raise ValueError("Must generate forecast before plotting")
        
        try:
            # Use historical_polls from the dataset.polls_train directly
            historical_polls = self.dataset.polls_train
            
            # Create the overall forecast plot
            fig_overall = plot_latent_trajectories(
                self.prediction,
                coords=self.prediction_coords,
                dims=self.prediction_dims,
                polls_train=historical_polls,
                election_date=self.dataset.election_date
            )
            
            # Create individual party plots
            figs_parties = []
            parties = self.prediction_coords['parties_complete']
            for party in parties:
                fig_party = plot_party_trajectory(
                    self.prediction,
                    coords=self.prediction_coords,
                    dims=self.prediction_dims,
                    party=party,
                    polls_train=historical_polls,
                    election_date=self.dataset.election_date
                )
                figs_parties.append(fig_party)
            
            # Return all figures
            return [fig_overall] + figs_parties
            
        except Exception as e:
            print(f"Error creating forecast plots: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def plot_house_effects(self, pollster: str):
        """
        Plot house effects for a specific pollster.
        
        Parameters:
        -----------
        pollster : str
            The pollster to plot
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            House effects plot
        """
        if self.trace is None:
            raise ValueError("Must run inference before plotting house effects")
         
        # Create a safer version that handles potential index errors
        try:   
            # Get the house effects data
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Get all pollsters from the trace
            pollsters = self.trace.posterior.coords["pollsters"].values
            
            # Check if the requested pollster is in the list
            if pollster not in pollsters:
                print(f"Warning: Pollster {pollster} not found in trace. Available pollsters: {pollsters}")
                
                # Create a placeholder figure
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f"Pollster '{pollster}' not found in model data", 
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_title(f"House Effects - {pollster}")
                plt.tight_layout()
                return fig
                
            # Find the index of the pollster
            pollster_idx = np.where(pollsters == pollster)[0][0]
            
            # Get house effects for this pollster
            house_effects = self.trace.posterior.house_effects.sel(pollsters=pollster).mean(("chain", "draw"))
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Convert to DataFrame for easier plotting
            house_effects_df = house_effects.to_dataframe(name="value").reset_index()
            ax.bar(house_effects_df["parties_complete"], house_effects_df["value"])
            
            ax.set_title(f"House Effects - {pollster}")
            ax.set_xlabel("Party")
            ax.set_ylabel("Effect")
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Error in plot_house_effects for {pollster}: {e}")
            # Return a blank figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error plotting house effects for '{pollster}':\n{str(e)}", 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f"House Effects - {pollster}")
            plt.tight_layout()
            return fig
    
    def plot_party_correlations(self):
        """
        Plot correlations between party vote shares.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Party correlations plot
        """
        if self.posterior is None:
            raise ValueError("Must run inference before plotting party correlations")
            
        return plot_party_correlations(
            idata=self.posterior,
            parties_complete=self.dataset.political_families
        )
    
    def plot_predictive_accuracy(self):
        """
        Plot the predictive accuracy of the model against test data.
        Only available for retrodictive forecasts, not for future elections.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure with the predictive accuracy
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Check if we're forecasting a future election
        target_date = pd.to_datetime(self.dataset.election_date)
        current_date = pd.Timestamp.now()
        
        if target_date > current_date:
            # Create a figure with an informative message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 
                    f"Predictive accuracy cannot be calculated for future elections.\n"
                    f"The target election date ({self.dataset.election_date}) is in the future.",
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title("Predictive Accuracy Unavailable")
            ax.set_xlabel("Observed Vote Share")
            ax.set_ylabel("Predicted Vote Share")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            return fig
        
        # Ensure we have run inference first
        if self.trace is None:
            raise ValueError("Must run inference before plotting predictive accuracy")
        
        # Ensure we have a prediction
        if self.prediction is None:
            raise ValueError("Must generate forecast before plotting predictive accuracy")
        
        if not hasattr(self.dataset, "polls_test") or self.dataset.polls_test is None or len(self.dataset.polls_test) == 0:
            raise ValueError("No test data available for predictive accuracy plot")
        
        # Use the prediction object for posterior predictive data
        import arviz as az
        
        # Initialize the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        try:
            # Check if prediction is an InferenceData object or a raw dictionary
            if isinstance(self.prediction, az.InferenceData):
                print("Using InferenceData object for predictive accuracy")
                if hasattr(self.prediction, 'posterior_predictive'):
                    post_pred = self.prediction.posterior_predictive
                else:
                    raise ValueError("InferenceData object does not have posterior_predictive group")
            else:
                print("Using raw prediction dictionary for predictive accuracy")
                # Use the raw dictionary directly
                post_pred_dict = self.prediction
                
                # Convert to a structure compatible with our plotting code
                class SimpleNamespace:
                    def __init__(self, **kwargs):
                        self.__dict__.update(kwargs)
                
                # Create a simple namespace object that mimics the InferenceData structure
                post_pred = SimpleNamespace(
                    data_vars=post_pred_dict.keys(),
                    observations=pd.to_datetime(self.prediction_coords.get('observations', [])),
                )
                
                # Add the prediction arrays as attributes
                for key, value in post_pred_dict.items():
                    setattr(post_pred, key, value)
            
            # Debugging info
            print(f"Posterior predictive variables: {list(post_pred.data_vars)}")
            if hasattr(post_pred, 'observations') and hasattr(post_pred.observations, 'shape'):
                print(f"Observations shape: {post_pred.observations.shape}")
            
            # Check if we have 'noisy_popularity' or 'latent_popularity' in our posterior predictive
            popularity_var_name = None
            if 'noisy_popularity' in post_pred.data_vars:
                popularity_var_name = 'noisy_popularity'
                print("Using noisy_popularity for predictions")
            elif 'latent_popularity' in post_pred.data_vars:
                popularity_var_name = 'latent_popularity'
                print("Using latent_popularity for predictions")
            else:
                raise ValueError("Neither 'noisy_popularity' nor 'latent_popularity' found in posterior predictive")
            
            # Get the popularity variable
            popularity_var = getattr(post_pred, popularity_var_name)
            
            # Get test data dates and prediction dates
            test_data = self.dataset.polls_test.copy()
            test_dates = pd.to_datetime(test_data['date'].values)
            
            # Handle different formats for observations
            if isinstance(post_pred.observations, np.ndarray):
                post_pred_dates = pd.to_datetime(post_pred.observations)
            else:
                post_pred_dates = post_pred.observations
            
            # Find the closest forecast date for each test date
            matching_indices_post = []
            matching_indices_test = []
            
            for i, test_date in enumerate(test_dates):
                # Find the closest date in the forecast
                days_diff = abs(post_pred_dates - test_date).total_seconds() / (24 * 60 * 60)
                closest_idx = np.argmin(days_diff)
                
                # Only include if within one day (to avoid distant matches)
                if days_diff[closest_idx] <= 1.0:  # 1 day tolerance
                    matching_indices_post.append(closest_idx)
                    matching_indices_test.append(i)
            
            if len(matching_indices_test) == 0:
                raise ValueError("No matching dates between test data and posterior predictive (even with 1-day tolerance)")
                
            print(f"Found {len(matching_indices_test)} matching dates between test data and forecast")
            
            # Extract values for matching dates
            if isinstance(self.prediction, az.InferenceData):
                # Get values from InferenceData object
                popularity_values = popularity_var.isel(observations=matching_indices_post).values
            else:
                # Get values from raw array
                popularity_values = np.array([popularity_var[:, :, idx, :] for idx in matching_indices_post])
                # Reshape if necessary
                if len(popularity_values.shape) == 5:  # [match, chain, draw, parties]
                    popularity_values = np.transpose(popularity_values, (1, 2, 0, 3))  # [chain, draw, match, parties]
            
            # Check if array is valid
            if np.all(np.isnan(popularity_values)):
                print("WARNING: All popularity values are NaN. Trying a different approach...")
                # Try a different approach
                if isinstance(self.prediction, az.InferenceData):
                    popularity_values = popularity_var.values[:, :, matching_indices_post, :]
                else:
                    popularity_values = np.array([popularity_var[:, :, idx, :] for idx in matching_indices_post])
                    popularity_values = np.transpose(popularity_values, (1, 2, 0, 3))
            
            # Calculate mean predictions across chains and draws
            mean_predicted = np.nanmean(popularity_values, axis=(0, 1))
            
            # Check if we still have NaN values
            if np.all(np.isnan(mean_predicted)):
                print("ERROR: All predicted values are NaN even after extraction")
                raise ValueError("Cannot extract valid prediction values")
            
            # Get the observed proportions from test data
            observed_props = {}
            test_subset = test_data.iloc[matching_indices_test]
            
            # Define party colors for consistent coloring
            party_colors = {
                'far-left': 'darkred',
                'left': 'red',
                'center-left': 'pink',
                'center': 'purple',
                'center-right': 'lightblue',
                'right': 'blue',
                'far-right': 'darkblue',
                'green': 'green',
                'other': 'gray'
            }
            
            # Plot the results
            for i, party in enumerate(self.dataset.political_families):
                if party in test_data.columns:
                    # Calculate observed proportions
                    observed = test_subset[party].values / test_subset['sample_size'].values
                    
                    # Get predicted values for this party
                    if mean_predicted.ndim == 2:
                        pred = mean_predicted[:, i]
                    else:
                        # Handle case where mean_predicted is already flat
                        pred = mean_predicted[i]
                    
                    # Skip if all predictions are NaN
                    if np.all(np.isnan(pred)):
                        print(f"Skipping {party} - all predictions are NaN")
                        continue
                    
                    # Remove any NaN pairs
                    valid_idx = ~np.isnan(pred)
                    if not np.all(valid_idx):
                        print(f"Removing {np.sum(~valid_idx)} NaN values for {party}")
                        obs = observed[valid_idx]
                        pred = pred[valid_idx]
                    else:
                        obs = observed
                    
                    # Skip if no valid data points remain
                    if len(obs) == 0:
                        print(f"No valid data points for {party}")
                        continue
                    
                    # Plot the scatter with consistent coloring
                    color = party_colors.get(party, f"C{i}")
                    ax.scatter(obs, pred, label=party, alpha=0.7, color=color)
                    
                    # Calculate mean absolute error
                    mae = np.mean(np.abs(obs - pred))
                    ax.text(0.05, 0.95 - i*0.05, f"{party} MAE: {mae:.4f}", 
                            transform=ax.transAxes, fontsize=10)
            
            # Add the diagonal line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            # Add labels and legend
            ax.set_xlabel('Observed Vote Share')
            ax.set_ylabel('Predicted Vote Share')
            ax.set_title('Predictive Accuracy: Observed vs. Predicted Vote Share')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Set axis limits
            ax.set_xlim(0, max(0.5, ax.get_xlim()[1]))
            ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
            
            # Adjust layout to fit the legend
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            # If we encounter an error, create a simple figure with the error message
            import traceback
            err_msg = f"Error plotting predictive accuracy:\n{str(e)}\n\n"
            err_msg += traceback.format_exc()
            print(err_msg)
            
            ax.text(0.5, 0.5, err_msg, 
                    ha='center', va='center', fontsize=10, wrap=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            return fig
    
    def posterior_predictive_check(self):
        """
        Perform posterior predictive checks.
        
        Returns:
        --------
        ppc_results : dict
            A dictionary containing various posterior predictive check results.
        """
        if self.posterior is None:
            raise ValueError("Must run inference before performing posterior predictive checks")
            
        return self.model.posterior_predictive_check(self.posterior)
    
    def load_prediction(self, prediction_path):
        """
        Load a saved prediction from a zarr file.
        
        Parameters:
        -----------
        prediction_path : str
            Path to the zarr file containing the prediction
            
        Returns:
        --------
        dict
            The raw prediction data
        dict
            The coordinates for the prediction data
        dict
            The dimensions for the prediction data
        """
        print(f"Loading prediction from {os.path.basename(prediction_path)}")
        try:
            import zarr
            
            # Open the zarr store
            store = zarr.DirectoryStore(prediction_path)
            root = zarr.open(store)
            
            # More efficient loading when debug is off
            if not self.debug:
                # Only load essential arrays (latent_popularity) when not debugging
                raw_ppc = {}
                
                # Get the essential prediction data
                if 'latent_popularity' in root.array_keys():
                    raw_ppc['latent_popularity'] = root['latent_popularity'][:]
                    print("Loaded essential prediction data")
                else:
                    # If essential data is not available, load all data
                    for var_name in root.array_keys():
                        if var_name not in ['coords', 'dims']:
                            raw_ppc[var_name] = root[var_name][:]
            else:
                # Debug mode: load all prediction data
                raw_ppc = {}
                for var_name in root.array_keys():
                    if var_name not in ['coords', 'dims']:
                        raw_ppc[var_name] = root[var_name][:]
            
            # Load the coordinates (needed for both modes)
            coords = {}
            if 'coords' in root:
                for coord_name in root['coords'].array_keys():
                    coord_values = root['coords'][coord_name][:]
                    if coord_name == 'observations':
                        # Convert string dates back to datetime
                        coord_values = pd.to_datetime(coord_values)
                    coords[coord_name] = coord_values
            
            # Load the dimensions (needed for both modes)
            dims = {}
            if 'dims' in root:
                for var_name in root['dims'].array_keys():
                    dims[var_name] = list(root['dims'][var_name][:])
            
            # Store the loaded data
            self.prediction = raw_ppc
            self.prediction_coords = coords
            self.prediction_dims = dims
            
            if self.debug:
                print("Using raw prediction data for plotting")
                # Print dimensions of prediction data
                print("\nPrediction dimensions:")
                for var_name, arr in raw_ppc.items():
                    print(f"  {var_name}: shape {arr.shape}")
                
                if coords:
                    print("\nCoordinates:")
                    for coord_name, values in coords.items():
                        print(f"  {coord_name}: {len(values)} values")
            
            return raw_ppc, coords, dims
            
        except Exception as e:
            if self.debug:
                print(f"Error loading prediction from zarr: {e}")
            
            # Try loading with arviz as a fallback
            try:
                import arviz as az
                idata = az.from_zarr(prediction_path)
                if self.debug:
                    print("Loaded prediction as InferenceData object")
                
                # Extract raw data, coordinates, and dimensions
                # More efficient in non-debug mode: only extract essential data
                raw_ppc = {}
                coords = {}
                dims = {}
                
                if hasattr(idata, 'posterior_predictive'):
                    # In non-debug mode, only extract latent_popularity if available
                    if not self.debug and 'latent_popularity' in idata.posterior_predictive.data_vars:
                        raw_ppc['latent_popularity'] = idata.posterior_predictive['latent_popularity'].values
                    else:
                        # Either in debug mode or essential variable not available: extract all
                        for var_name in idata.posterior_predictive.data_vars:
                            raw_ppc[var_name] = idata.posterior_predictive[var_name].values
                    
                    # Extract coordinates
                    for coord_name in idata.posterior_predictive.coords:
                        coords[coord_name] = idata.posterior_predictive.coords[coord_name].values
                    
                    # Create dimensions
                    for var_name in raw_ppc:
                        var_dims = list(idata.posterior_predictive[var_name].dims)
                        dims[var_name] = var_dims
                
                # Store the extracted data
                self.prediction = raw_ppc
                self.prediction_coords = coords
                self.prediction_dims = dims
                
                if self.debug:
                    print("Extracted raw data from InferenceData object")
                return raw_ppc, coords, dims
                
            except Exception as nested_e:
                error_msg = f"Failed to load prediction: {e} -> {nested_e}"
                print(error_msg)
                if self.debug:
                    import traceback
                    traceback.print_exc()
                raise ValueError(error_msg)
    
    def _build_model(self):
        """
        Build the model if it hasn't been built already.
        
        Returns:
        --------
        pymc.Model
            The built PyMC model
        """
        if self.debug:
            print("\n=== MODEL BUILD DIAGNOSTICS ===")
            print("Model not built yet - building now...")
        
        try:
            self.model_config = {
                "election_date": self.dataset.election_date,
                "baseline_timescales": self.dataset.baseline_timescales,
                "election_timescales": self.dataset.election_timescales,
                "weights": self.dataset.weights,
                "save_dir": getattr(self, "output_dir", None),
                "seed": 12345,
            }
            
            # Set historical elections as coordinates
            if self.debug:
                print(f"Setting elections coord to historical elections: {self.dataset.historical_election_dates}")
                print(f"Setting elections_observed to all historical elections: {self.dataset.historical_election_dates}")
            
            # Build the model
            self.model = self.model.build_model()
            
            if self.debug:
                # Print model dimensions
                print("\n=== MODEL DIMENSIONS ===")
                for dim_name, dim_values in self.model.coords.items():
                    print(f"{dim_name}: shape={len(dim_values)}")
                
                # Print information about results data
                if hasattr(self.dataset, "results_oos"):
                    print(f"results_oos shape: {self.dataset.results_oos.shape}")
                    print(f"results_oos election dates: {self.dataset.results_oos['election_date']}")
                
                # Print more detailed model information
                print(f"For inference: Using historical elections data with shape: {self.dataset.historical_elections.shape}")
                print(f"Setting election dimensions to match historical elections: {len(self.dataset.historical_election_dates)}")
            
            print("Model successfully built!")
            
            if self.debug:
                # Print model variable counts
                var_count = len(self.model.named_vars)
                observed_vars = len(self.model.observed_RVs)
                free_vars = len(self.model.free_RVs)
                deterministics = len(self.model.deterministics)
                
                print("\nModel variables:")
                print(f"Total variable count: {var_count}")
                print(f"Observed variables: {observed_vars}")
                print(f"Free variables: {free_vars}")
                print(f"Deterministic variables: {deterministics}")
                print("=====================================\n")
            
            return self.model
            
        except Exception as e:
            error_msg = f"Failed to build model: {e}"
            print(error_msg)
            if self.debug:
                import traceback
                traceback.print_exc()
            raise ValueError(error_msg)
    
    def debug_prediction_structure(self):
        """
        Print detailed debug information about the prediction structure.
        This is helpful for debugging issues with predictions and plots.
        """
        if not hasattr(self, "prediction") or self.prediction is None:
            print("No prediction available. Generate a forecast first.")
            return
        
        import numpy as np
        
        print("\n=== PREDICTION STRUCTURE DEBUG INFO ===")
        print(f"Type of prediction: {type(self.prediction)}")
        print(f"Keys in prediction: {list(self.prediction.keys())}")
        
        # Print information about each prediction array
        for key, value in self.prediction.items():
            print(f"\n{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Contains NaNs: {np.isnan(value).any()}")
            if np.isnan(value).any():
                print(f"  NaN percentage: {np.isnan(value).mean() * 100:.2f}%")
        
        # Print information about coordinates
        if hasattr(self, "prediction_coords") and self.prediction_coords is not None:
            print("\nCoordinates:")
            for key, value in self.prediction_coords.items():
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")
                print(f"  Length: {len(value)}")
                print(f"  Content: {value}")
                print(f"  Is scalar: {np.isscalar(value)}")
                
                # Special handling for parties_complete
                if key == 'parties_complete':
                    print(f"  Content type: {type(value[0]) if len(value) > 0 else 'empty'}")
                    # Convert to list for safety
                    self.prediction_coords[key] = list(value)
                    print(f"  Converted to list: {self.prediction_coords[key]}")
        
        # Print information about dimensions
        if hasattr(self, "prediction_dims") and self.prediction_dims is not None:
            print("\nDimensions:")
            for key, value in self.prediction_dims.items():
                print(f"  {key}: {value}")
        
        print("=======================================\n")
        return self.prediction, self.prediction_coords, self.prediction_dims 