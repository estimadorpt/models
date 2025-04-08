import os
from typing import Dict, List, Tuple, Optional, Union, Type

import arviz
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import pymc as pm
import time
import datetime
import zarr

from src.data.dataset import ElectionDataset
from src.models.election_model import ElectionModel
from src.models.base_model import BaseElectionModel


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
        model_class: Type[BaseElectionModel] = ElectionModel,
        baseline_timescales: List[int] = [365],  # Annual cycle
        election_timescales: List[int] = [30, 15],  # Pre-campaign and official campaign
        test_cutoff: pd.Timedelta = None,
        debug: bool = False,
        **model_kwargs
    ):
        """
        Initialize the elections facade.
        
        Parameters:
        -----------
        election_date : str
            The target election date in 'YYYY-MM-DD' format
        model_class : Type[BaseElectionModel]
            The model class to use for the facade
        baseline_timescales : List[int]
            Timescales for baseline GP components in days
        election_timescales : List[int]
            Timescales for election-specific GP components in days
        test_cutoff : pd.Timedelta
            How much data to hold out for testing
        debug : bool
            Whether to print detailed diagnostic information
        model_kwargs :
            Additional keyword arguments passed to the model class constructor.
        """
        self.debug = debug
        self.election_date = election_date
        self.model_instance: Optional[BaseElectionModel] = None
        self.trace: Optional[az.InferenceData] = None
        self.test_cutoff = test_cutoff

        # Ensure timescales are lists
        self.baseline_timescales = baseline_timescales if isinstance(baseline_timescales, list) else [baseline_timescales]
        self.election_timescales = election_timescales if isinstance(election_timescales, list) else [election_timescales]
        self.config = {
            "election_date": self.election_date,
            "baseline_timescales": self.baseline_timescales,
            "election_timescales": self.election_timescales,
            "cutoff_date": self.test_cutoff
        }
        # Combine config with model-specific kwargs
        self.model_config_full = {**self.config, **model_kwargs}

        if self.debug:
            print(f"Initializing Facade with -> election_date: {self.election_date}")
            print(f"Initializing Facade with -> baseline_timescales: {self.baseline_timescales}")
            print(f"Initializing Facade with -> election_timescales: {self.election_timescales}")
            print(f"Initializing Facade with -> cutoff_date: {self.test_cutoff}")

        self.dataset = ElectionDataset(
            election_date=self.election_date,
            baseline_timescales=self.baseline_timescales,
            election_timescales=self.election_timescales,
            test_cutoff=self.test_cutoff,
        )
        
        # Create the specific model instance using the provided class
        print(f"Initializing model of type: {model_class.__name__}")
        self.model_instance = model_class(
            dataset=self.dataset,
            **self.model_config_full
        )
        
        # Initialize trace containers
        self.prior = None
        self.posterior = None
        self.prediction = None
        self.prediction_coords = None
        self.prediction_dims = None
        
        # Set debug flag
        # self.debug = debug # Already set
        
        # Initialize output_dir and model_config
        self.output_dir = None
        
        # Flag to track if inference results have been saved
        self._inference_results_saved = False
        
    def build_model(self):
        """Build the PyMC model using the specific model instance"""
        if self.model_instance is None:
            raise RuntimeError("Model instance has not been initialized.")
        # Delegate building to the specific model instance
        self.model_instance.model = self.model_instance.build_model()
        return self.model_instance.model
    
    def run_inference(
        self, 
        draws: int = 200, 
        tune: int = 100, 
        target_accept: float = 0.9, 
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
            "party_time_effect",
            "latent_popularity_trajectory",
            "noisy_popularity",
            "N_approve",
            "R",
            "house_effects"
        ]
        
        # Include draws and tune in sampler_kwargs instead of passing them directly
        sampler_kwargs['draws'] = draws
        sampler_kwargs['tune'] = tune
        sampler_kwargs['target_accept'] = target_accept
        sampler_kwargs['max_treedepth'] = 15 # Increase max tree depth
        
        self.prior, self.trace, self.posterior = self.model_instance.sample_all(
            var_names=var_names,
            **sampler_kwargs
        )
        
        return self.prior, self.trace, self.posterior
    
    def save_inference_results(self, directory: str = ".", force: bool = False):
        """
        Save inference results to disk using Zarr format.

        Removes the directory if no actual data (prior, trace, posterior) is saved.

        Parameters:
        -----------
        directory : str
            Directory where to save files
        force : bool
            If True, save even if results were already saved

        Returns:
        --------
        bool
            True if at least one inference result file was successfully saved, False otherwise.
        """
        # print("\nDEBUG: ENTERING save_inference_results", flush=True) # Removed debug print
        
        # Skip if already saved and not forcing
        if hasattr(self, "_inference_results_saved") and self._inference_results_saved and not force:
            print(f"Inference results previously saved to {directory}, skipping re-save.") # Removed flush
            # print("DEBUG: EXITING save_inference_results (already saved)", flush=True) # Removed debug print
            return True

        # Create output directory; exist_ok=True prevents error if it exists
        # We will remove it later if nothing is saved.
        os.makedirs(directory, exist_ok=True)
        print(f"Attempting to save inference results to: {directory}") # Removed flush

        results_saved = False # Flag to track if anything was actually saved

        # Define paths using the corrected naming scheme
        prior_path = os.path.join(directory, "prior_checks.zarr")
        trace_path = os.path.join(directory, "trace.zarr")
        posterior_path = os.path.join(directory, "posterior_checks.zarr")

        try:
            if hasattr(self, "prior") and self.prior is not None:
                print(f"Saving prior checks to: {prior_path}") # Removed flush
                arviz.to_zarr(self.prior, prior_path, mode='w') # Use mode='w' to overwrite if exists
                results_saved = True
            else:
                print("Prior object not found or is None, skipping save.") # Removed flush

            if hasattr(self, "trace") and self.trace is not None:
                print(f"Saving trace (posterior) to: {trace_path}") # Removed flush
                arviz.to_zarr(self.trace, trace_path, mode='w')
                results_saved = True
            else:
                print("Trace object not found or is None, skipping save.") # Removed flush

            if hasattr(self, "posterior") and self.posterior is not None:
                print(f"Saving posterior checks to: {posterior_path}") # Removed flush
                arviz.to_zarr(self.posterior, posterior_path, mode='w')
                results_saved = True
            else:
                print("Posterior object not found or is None, skipping save.") # Removed flush

            if results_saved:
                print(f"Successfully saved inference results to {directory}") # Removed flush
                self._inference_results_saved = True
                # print("DEBUG: EXITING save_inference_results (success)", flush=True) # Removed debug print
                return True
            else:
                print(f"No inference results (prior, trace, posterior) were available to save.") # Removed flush
                # Attempt to remove the directory as nothing was saved into it
                try:
                    # Check if directory is empty before removing (safer)
                    # Only check/remove if directory actually exists
                    if os.path.exists(directory) and not os.listdir(directory):
                        print(f"Removing empty directory: {directory}") # Removed flush
                        os.rmdir(directory)
                    elif os.path.exists(directory):
                         print(f"Warning: Directory {directory} was not empty despite no results being saved. Not removing.") # Removed flush
                    # else: directory doesn't exist, nothing to remove
                except OSError as e:
                    print(f"Warning: Could not remove directory {directory}: {e}") # Removed flush
                self._inference_results_saved = False # Ensure flag reflects reality
                # print("DEBUG: EXITING save_inference_results (nothing saved)", flush=True) # Removed debug print
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to save one or more inference results to {directory}: {e}") # Removed flush
            # Should we still attempt removal if an error occurred mid-save?
            # For now, let's leave the directory if an error happened during writing.
            self._inference_results_saved = False # Mark as not successfully saved
            # print("DEBUG: EXITING save_inference_results (exception)", flush=True) # Removed debug print
            return False
    
    def load_inference_results(self, directory: str = "."):
        """
        Load previously saved inference results.
        
        Parameters:
        -----------
        directory : str
            Directory where files are saved
        """
        # Use new filenames first, fall back to old names
        prior_path = os.path.join(directory, "prior_checks.zarr")
        trace_path = os.path.join(directory, "trace.zarr")
        posterior_path = os.path.join(directory, "posterior_checks.zarr")
        
        # Old filenames for backward compatibility
        old_prior_path = os.path.join(directory, "prior_check.zarr")
        old_posterior_path = os.path.join(directory, "posterior_check.zarr")
        # Old names used in a previous refactor
        legacy_prior_path = os.path.join(directory, "prior.zarr")
        legacy_posterior_path = os.path.join(directory, "posterior.zarr")
        
        # Load prior checks
        loaded_prior = False
        if os.path.exists(prior_path):
            print(f"Loading prior checks from {prior_path}")
            self.prior = arviz.from_zarr(prior_path)
            loaded_prior = True
        elif os.path.exists(old_prior_path):
            print(f"Loading prior checks from old path {old_prior_path}")
            self.prior = arviz.from_zarr(old_prior_path)
            loaded_prior = True
        elif os.path.exists(legacy_prior_path):
            print(f"Loading prior checks from legacy path {legacy_prior_path}")
            self.prior = arviz.from_zarr(legacy_prior_path)
            loaded_prior = True
            
        if not loaded_prior:
             print(f"Warning: Prior predictive checks file not found at expected paths in {directory}")

        # Load trace (main posterior)
        loaded_trace = False
        if os.path.exists(trace_path):
            print(f"Loading trace (posterior) from {trace_path}")
            self.trace = arviz.from_zarr(trace_path)
            self._analyze_trace_quality(self.trace) # Analyze after loading
            loaded_trace = True
            
        if not loaded_trace:
             print(f"ERROR: Main trace file (trace.zarr) not found in {directory}. Cannot proceed without posterior samples.")
             # Decide on behavior: raise error or allow continuation? Raising is safer.
             raise FileNotFoundError(f"Trace file not found: {trace_path}")
             
        # Load posterior checks
        loaded_posterior = False
        if os.path.exists(posterior_path):
            print(f"Loading posterior checks from {posterior_path}")
            self.posterior = arviz.from_zarr(posterior_path)
            loaded_posterior = True
        elif os.path.exists(old_posterior_path):
            print(f"Loading posterior checks from old path {old_posterior_path}")
            self.posterior = arviz.from_zarr(old_posterior_path)
            loaded_posterior = True
        elif os.path.exists(legacy_posterior_path):
            print(f"Loading posterior checks from legacy path {legacy_posterior_path}")
            self.posterior = arviz.from_zarr(legacy_posterior_path)
            loaded_posterior = True
            
        if not loaded_posterior:
             print(f"Warning: Posterior predictive checks file not found at expected paths in {directory}")

        print("Finished loading inference results.")
        # print("DEBUG: Attempting to return True from load_inference_results") # DEBUG
        result_to_return = True
        # print(f"DEBUG: Value to return: {result_to_return}, Type: {type(result_to_return)}") # DEBUG
        return result_to_return
    
    def _analyze_trace_quality(self, trace):
        """Analyze trace quality and print diagnostics."""
        try:  
            # Handle division by zero warnings in arviz functions
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
                
                if 'diverging' in trace.sample_stats:
                    n_divergent = trace.sample_stats.diverging.sum().item()
                    if n_divergent > 0:
                        pct_divergent = n_divergent / (trace.posterior.dims['chain'] * trace.posterior.dims['draw'])
                        print(f"WARNING: {n_divergent} divergent transitions detected ({pct_divergent:.1%} of samples)")
                        
                        # If there are divergences, try to identify which parameters might be problematic
                        # Look at parameters with high R-hat that might be related to divergences
                        try:
                            summary = arviz.summary(trace, var_names=['~house_effects', '~house_election_effects'])
                            high_rhat = summary[summary.r_hat > 1.05]
                            if not high_rhat.empty:
                                print("Parameters with high R-hat that might be related to divergences:")
                                print(high_rhat[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']])
                        except Exception as e:
                            print(f"Warning: Failed to compute summary statistics: {e}")
                
                # Check for tree depth issues
                if 'tree_depth' in trace.sample_stats:
                    max_treedepth = trace.sample_stats.attrs.get('max_treedepth', 10)
                    max_depths = (trace.sample_stats.tree_depth >= max_treedepth).sum().item()
                    if max_depths > 0:
                        pct_max_depth = max_depths / (trace.posterior.dims['chain'] * trace.posterior.dims['draw'])
                        print(f"WARNING: {max_depths} samples ({pct_max_depth:.1%}) reached maximum tree depth")
                        if pct_max_depth > 0.1:
                            print("Consider increasing max_treedepth in sampler parameters")
                
                # Compute ESS and R-hat for key parameters
                key_params = ['party_baseline', 'election_party_baseline', 'party_time_effect',
                            'concentration_polls', 'concentration_results', 'lsd_baseline',
                            'party_time_weight', 'house_effects_sd', 'log_concentration_polls',
                            'log_concentration_results']
                
                available_params = [p for p in key_params if p in trace.posterior]
                if available_params:
                    print("\n=== KEY PARAMETER DIAGNOSTICS ===")
                    summary = arviz.summary(trace, var_names=available_params)
                    
                    # Print parameters with low ESS or high R-hat
                    low_ess = summary[summary.ess_bulk < 400]
                    if not low_ess.empty:
                        print("\nParameters with low effective sample size (ESS < 400):")
                        print(low_ess[['mean', 'sd', 'ess_bulk', 'ess_tail', 'r_hat']])
                    
                    high_rhat = summary[summary.r_hat > 1.01]
                    if not high_rhat.empty:
                        print("\nParameters with high R-hat (> 1.01):")
                        print(high_rhat[['mean', 'sd', 'ess_bulk', 'ess_tail', 'r_hat']])
                    
                    # If no issues found, print a positive message
                    if low_ess.empty and high_rhat.empty:
                        print("All key parameters show good convergence (ESS > 400, R-hat < 1.01)")
        except Exception as e:
            print(f"Error in trace quality analysis: {str(e)}")
            
        # Analyze concentration parameters specifically - outside the warning suppression
        for param in ['concentration_polls', 'concentration_results', 
                      'log_concentration_polls', 'log_concentration_results']:
            if param in trace.posterior:
                param_values = trace.posterior[param].values
                param_mean = np.mean(param_values)
                param_std = np.std(param_values)
                print(f"\n{param}: mean={param_mean:.2f}, std={param_std:.2f}")
                
                # For log-parameterized versions, also show the exponentiated statistics
                if param.startswith('log_'):
                    exp_values = np.exp(param_values)
                    exp_mean = np.mean(exp_values)
                    exp_std = np.std(exp_values)
                    print(f"exp({param}): mean={exp_mean:.2f}, std={exp_std:.2f}")
    
    def _check_trace_quality(self):
        """
        Check if the trace quality is good enough for analysis.
        
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
        key_params = ['party_baseline', 'election_party_baseline', 'party_time_effect']
        
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
    
    def plot_predictive_accuracy(self):
        """
        Plot the predictive accuracy of the model against test data.
        Only available for retrodictive analysis of past elections.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure with the predictive accuracy
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Check if the target date is valid
        target_date = pd.to_datetime(self.dataset.election_date)
        current_date = pd.Timestamp.now()
        
        if target_date > current_date:
            # Create a figure with an informative message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 
                    f"Predictive accuracy cannot be calculated for future dates.\n"
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
            raise ValueError("Must generate prediction before plotting predictive accuracy")
        
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
                print("Neither 'noisy_popularity' nor 'latent_popularity' found in posterior predictive")
                # Create a figure with an informative message instead of raising an error
                ax.text(0.5, 0.5, 
                        "Neither 'noisy_popularity' nor 'latent_popularity' found in posterior predictive.\n"
                        "These variables are required for calculating predictive accuracy.",
                        ha='center', va='center', fontsize=12, transform=ax.transAxes,
                        wrap=True)
                ax.set_title("Predictive Accuracy Unavailable")
                ax.set_xlabel("Observed Vote Share")
                ax.set_ylabel("Predicted Vote Share")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                plt.tight_layout()
                return fig
            
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
            
            # Find the closest prediction date for each test date
            matching_indices_post = []
            matching_indices_test = []
            
            for i, test_date in enumerate(test_dates):
                # Find the closest date in the prediction
                days_diff = abs(post_pred_dates - test_date).total_seconds() / (24 * 60 * 60)
                closest_idx = np.argmin(days_diff)
                
                # Only include if within one day (to avoid distant matches)
                if days_diff[closest_idx] <= 1.0:  # 1 day tolerance
                    matching_indices_post.append(closest_idx)
                    matching_indices_test.append(i)
            
            if len(matching_indices_test) == 0:
                print("No matching dates between test data and posterior predictive (even with 1-day tolerance)")
                # Create a figure with an informative message instead of raising an error
                ax.text(0.5, 0.5, 
                        "No matching dates found between test data and posterior predictive.\n"
                        "This may occur during cross-validation when test poll dates don't align with prediction dates.",
                        ha='center', va='center', fontsize=12, transform=ax.transAxes,
                        wrap=True)
                ax.set_title("Predictive Accuracy Unavailable")
                ax.set_xlabel("Observed Vote Share")
                ax.set_ylabel("Predicted Vote Share")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                plt.tight_layout()
                return fig
                
            print(f"Found {len(matching_indices_test)} matching dates between test data and prediction")
            
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
                try:
                    if isinstance(self.prediction, az.InferenceData):
                        popularity_values = popularity_var.values[:, :, matching_indices_post, :]
                    else:
                        popularity_values = np.array([popularity_var[:, :, idx, :] for idx in matching_indices_post])
                        popularity_values = np.transpose(popularity_values, (1, 2, 0, 3))
                except Exception as e:
                    print(f"Error with alternative extraction approach: {e}")
            
            # Calculate mean predictions across chains and draws
            mean_predicted = np.nanmean(popularity_values, axis=(0, 1))
            
            # Check if we still have NaN values
            if np.all(np.isnan(mean_predicted)):
                print("ERROR: All predicted values are NaN even after extraction")
                # Create a figure with an informative message instead of raising an error
                ax.text(0.5, 0.5, 
                        "Cannot extract valid prediction values.\n"
                        "All predicted values are NaN even after extraction.\n"
                        "This may occur when the posterior predictive values are not properly aligned.",
                        ha='center', va='center', fontsize=12, transform=ax.transAxes,
                        wrap=True)
                ax.set_title("Predictive Accuracy Unavailable")
                ax.set_xlabel("Observed Vote Share")
                ax.set_ylabel("Predicted Vote Share")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                plt.tight_layout()
                return fig
            
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
            
        if self.model_instance is None or not hasattr(self.model_instance, 'posterior_predictive_check'):
            raise NotImplementedError("The current model does not implement posterior_predictive_check.")

        # Delegate to the specific model instance
        return self.model_instance.posterior_predictive_check(self.posterior)

    def _build_model(self):
        """
        Build the model if it hasn't been built already.
        
        Returns:
        --------
        pymc.Model
            The built PyMC model
        """
        if self.model_instance is None:
            raise RuntimeError("Model instance has not been initialized.")
        
        if self.debug:
            print("\n=== MODEL BUILD DIAGNOSTICS ===", flush=True)
            print("Model not built yet - building now...")
        
        try:
            # model_config_full is already set in __init__
            
            # Set historical elections as coordinates
            if self.debug:
                print(f"Setting elections coord to historical elections: {self.dataset.historical_election_dates}", flush=True)
                print(f"Setting elections_observed to all historical elections: {self.dataset.historical_election_dates}", flush=True)
            
            # Build the model using the instance method
            built_model = self.model_instance.build_model() # This assigns to self.model_instance.model
            print("Model successfully built!") 
            
            if built_model is None:
                raise RuntimeError("Model building returned None.")

            if self.debug:
                 # Print model variable counts
                 var_count = len(built_model.named_vars)
                 observed_vars = len(built_model.observed_RVs)
                 free_vars = len(built_model.free_RVs)
                 deterministics = len(built_model.deterministics)
                  
                 print("\nModel variables:", flush=True)
                 print(f"Total variable count: {var_count}", flush=True)
                 print(f"Observed variables: {observed_vars}", flush=True)
                 print(f"Free variables: {free_vars}", flush=True)
                 print(f"Deterministic variables: {deterministics}", flush=True)
                 print("=====================================", flush=True)
            
            return built_model
            
        except Exception as e:
            error_msg = f"Failed to build model: {e}"
            print(error_msg, flush=True)
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
            print("No prediction available. Generate a prediction first.")
            return
        
        import numpy as np
        
        print("\n=== PREDICTION STRUCTURE DEBUG INFO ===", flush=True)
        print(f"Type of prediction: {type(self.prediction)}", flush=True)
        print(f"Keys in prediction: {list(self.prediction.keys())}", flush=True)
        
        # Print information about each prediction array
        for key, value in self.prediction.items():
            print(f"\n{key}:", flush=True)
            print(f"  Shape: {value.shape}", flush=True)
            print(f"  Dtype: {value.dtype}", flush=True)
            print(f"  Contains NaNs: {np.isnan(value).any()}", flush=True)
            if np.isnan(value).any():
                print(f"  NaN percentage: {np.isnan(value).mean() * 100:.2f}%", flush=True)
        
        # Print information about coordinates
        if hasattr(self, "prediction_coords") and self.prediction_coords is not None:
            print("\nCoordinates:", flush=True)
            for key, value in self.prediction_coords.items():
                print(f"\n{key}:", flush=True)
                print(f"  Type: {type(value)}", flush=True)
                print(f"  Length: {len(value)}", flush=True)
                print(f"  Content: {value}", flush=True)
                print(f"  Is scalar: {np.isscalar(value)}", flush=True)
                
                # Special handling for parties_complete
                if key == 'parties_complete':
                    print(f"  Content type: {type(value[0]) if len(value) > 0 else 'empty'}", flush=True)
                    # Convert to list for safety
                    self.prediction_coords[key] = list(value)
                    print(f"  Converted to list: {self.prediction_coords[key]}", flush=True)
        
        # Print information about dimensions
        if hasattr(self, "prediction_dims") and self.prediction_dims is not None:
            print("\nDimensions:", flush=True)
            for key, value in self.prediction_dims.items():
                print(f"  {key}: {value}", flush=True)
        
        print("=======================================", flush=True)
        return self.prediction, self.prediction_coords, self.prediction_dims 

    def nowcast_party_support(self, *args, **kwargs):
        """Calls the nowcast method of the specific model instance, if available."""
        if self.model_instance is None or not hasattr(self.model_instance, 'nowcast_party_support'):
            raise NotImplementedError("The current model does not implement nowcast_party_support.")
        if self.trace is None:
             raise ValueError("Model must be fit with run_inference() before nowcasting")
        # Pass necessary arguments, including trace
        return self.model_instance.nowcast_party_support(idata=self.trace, *args, **kwargs)

    def predict(self, *args, **kwargs):
        """Calls the predict method of the specific model instance, if available."""
        if self.model_instance is None or not hasattr(self.model_instance, 'predict'):
            raise NotImplementedError("The current model does not implement predict.")
        if self.trace is None:
             raise ValueError("Model must be fit with run_inference() before predicting")
        return self.model_instance.predict(*args, **kwargs)

    def predict_history(self, *args, **kwargs):
        """Calls the predict_history method of the specific model instance, if available."""
        if self.model_instance is None or not hasattr(self.model_instance, 'predict_history'):
            raise NotImplementedError("The current model does not implement predict_history.")
        if self.trace is None:
             raise ValueError("Model must be fit with run_inference() before predicting history")
        return self.model_instance.predict_history(*args, **kwargs)

    def predict_latent_trajectory(self, *args, **kwargs):
        """Calls the predict_latent_trajectory method of the specific model instance, if available."""
        if self.model_instance is None or not hasattr(self.model_instance, 'predict_latent_trajectory'):
            raise NotImplementedError("The current model does not implement predict_latent_trajectory.")
        if self.trace is None:
             raise ValueError("Model must be fit with run_inference() before predicting trajectory")
        return self.model_instance.predict_latent_trajectory(idata=self.trace, *args, **kwargs)

    def generate_diagnostic_plots(self, directory: str = "."):
        """
        Generate and save arviz diagnostic plots for the trace.
        
        Parameters:
        -----------
        directory : str
            Directory where to save the plots.
        """
        if self.trace is None:
            print("Trace object not found. Cannot generate diagnostic plots.")
            return
            
        os.makedirs(directory, exist_ok=True)
        print(f"Generating diagnostic plots in: {directory}")
        
        # Key parameters to focus on (avoid plotting too many)
        key_params = [
            'party_baseline', 
            'election_party_baseline_sd',
            'election_party_baseline',
            'gp_coef_baseline', 
            'concentration_polls', 
            'concentration_results',
            'house_effects'
        ]
        plot_vars = [p for p in key_params if p in self.trace.posterior]

        # 1. Trace Plot (for key parameters)
        try:
            trace_plot_path = os.path.join(directory, "diagnostic_trace_plot.png")
            print(f"- Generating trace plot for key parameters...")
            az.plot_trace(self.trace, var_names=plot_vars)
            plt.savefig(trace_plot_path)
            plt.close()
            print(f"  Saved trace plot to {trace_plot_path}")
        except Exception as e:
            print(f"  Error generating trace plot: {e}")

        # 2. Energy Plot
        try:
            energy_plot_path = os.path.join(directory, "diagnostic_energy_plot.png")
            print(f"- Generating energy plot...")
            az.plot_energy(self.trace)
            plt.savefig(energy_plot_path)
            plt.close()
            print(f"  Saved energy plot to {energy_plot_path}")
        except Exception as e:
            print(f"  Error generating energy plot: {e}")

        # 3. Pair Plot (for a smaller subset, e.g., baseline parameters)
        try:
            pair_plot_vars = [p for p in ['party_baseline', 'election_party_baseline_sd', 'concentration_polls', 'concentration_results'] if p in self.trace.posterior]
            if pair_plot_vars:
                pair_plot_path = os.path.join(directory, "diagnostic_pair_plot.png")
                print(f"- Generating pair plot for {pair_plot_vars}...")
                az.plot_pair(self.trace, var_names=pair_plot_vars, divergences=True)
                plt.savefig(pair_plot_path)
                plt.close()
                print(f"  Saved pair plot to {pair_plot_path}")
        except Exception as e:
            print(f"  Error generating pair plot: {e}")
            
        # 4. ESS Plot
        try:
            ess_plot_path = os.path.join(directory, "diagnostic_ess_plot.png")
            print(f"- Generating ESS plot...")
            az.plot_ess(self.trace, var_names=plot_vars, kind="evolution")
            plt.savefig(ess_plot_path)
            plt.close()
            az.plot_ess(self.trace, var_names=plot_vars, kind="local")
            plt.savefig(ess_plot_path.replace(".png", "_local.png"))
            plt.close()
            print(f"  Saved ESS plots to {directory}")
        except Exception as e:
            print(f"  Error generating ESS plot: {e}")
            
        # 5. R-hat Plot
        try:
            rhat_plot_path = os.path.join(directory, "diagnostic_rhat_plot.png")
            print(f"- Generating R-hat plot...")
            az.plot_rank(self.trace, var_names=plot_vars)
            plt.savefig(rhat_plot_path)
            plt.close()
            print(f"  Saved R-hat plot to {rhat_plot_path}")
        except Exception as e:
            print(f"  Error generating R-hat plot: {e}")

        # 6. Generate Summary Text File
        summary_path = os.path.join(directory, "diagnostic_summary.txt")
        print(f"- Generating diagnostic summary text file...")
        try:
            summary = az.summary(self.trace)
            rhat_threshold = 1.05
            ess_threshold = 400 # Assuming 4 chains, 100 per chain

            bad_rhat = summary[summary['r_hat'] > rhat_threshold]
            bad_ess_bulk = summary[summary['ess_bulk'] < ess_threshold]
            bad_ess_tail = summary[summary['ess_tail'] < ess_threshold]

            with open(summary_path, 'w') as f:
                f.write("Convergence Diagnostics Summary\n")
                f.write("=============================\n")
                f.write(f"R-hat Threshold: > {rhat_threshold}\n")
                f.write(f"ESS Threshold (Bulk & Tail): < {ess_threshold}\n\n")

                if bad_rhat.empty and bad_ess_bulk.empty and bad_ess_tail.empty:
                    f.write("No major convergence issues detected based on R-hat and ESS thresholds.\n")
                else:
                    f.write("Potential convergence issues detected:\n")
                    if not bad_rhat.empty:
                        f.write("\n--- Parameters with R-hat > {}: ---\n".format(rhat_threshold))
                        f.write(bad_rhat[['r_hat']].to_string())
                        f.write("\n")
                    if not bad_ess_bulk.empty:
                        f.write("\n--- Parameters with Bulk ESS < {}: ---\n".format(ess_threshold))
                        f.write(bad_ess_bulk[['ess_bulk']].to_string())
                        f.write("\n")
                    if not bad_ess_tail.empty:
                        f.write("\n--- Parameters with Tail ESS < {}: ---\n".format(ess_threshold))
                        f.write(bad_ess_tail[['ess_tail']].to_string())
                        f.write("\n")
            print(f"  Saved diagnostic summary to {summary_path}")
        except Exception as e:
            print(f"  Error generating diagnostic summary: {e}")

        print("Diagnostic plot and summary generation complete.")

    def get_election_day_latent_popularity(self):
        """
        Extracts the posterior distribution of the latent popularity 
        specifically at the target election day (countdown=0).

        Returns:
        --------
        xr.DataArray or None:
            An xarray DataArray containing the posterior samples of latent popularity
            at countdown=0 for the target election. 
            Dimensions: (chain, draw, parties_complete).
            Returns None if the necessary data is not available in the trace.
        """
        if self.trace is None:
            print("Error: Trace object not found. Cannot extract latent popularity.")
            return None
        
        # Delegate to the model instance if the method exists
        return self.model_instance.get_election_day_latent_popularity()

    # TODO: Implement post-hoc adjustment for Blank/Null votes 
    #       when generating final forecasts (e.g., for seat projections). 
    #       The current model output represents shares among modeled parties.
    #       This adjustment step would estimate the Blank/Null share and 
    #       renormalize the modeled party shares to represent shares of valid votes. 