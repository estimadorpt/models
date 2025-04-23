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
from src.models.static_baseline_election_model import StaticBaselineElectionModel
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
        model_class: Type[BaseElectionModel] = StaticBaselineElectionModel,
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
        # <<< Init Start Debug >>>
        print("DEBUG FACADE INIT: Starting __init__...") 
        # <<< End Init Start Debug >>>
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
        # <<< Before Model Instance Debug >>>
        print(f"DEBUG FACADE INIT: About to instantiate model: {model_class.__name__}") 
        # <<< End Before Model Instance Debug >>>
        self.model_instance = model_class(
            dataset=self.dataset,
            **self.model_config_full
        )
        # Pass the election date to the model instance
        self.model_instance.election_date = self.election_date
        # <<< After Model Instance Debug >>>
        print(f"DEBUG FACADE INIT: Finished instantiating model instance. Type: {type(self.model_instance)}") 
        # <<< End After Model Instance Debug >>>
        
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
        # <<< Init End Debug >>>
        print("DEBUG FACADE INIT: Finished __init__.") 
        # <<< End Init End Debug >>>
        
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
        Runs the MCMC sampling process for the compiled model.

        Args:
            draws: Number of posterior samples per chain.
            tune: Number of tuning (burn-in) samples per chain.
            target_accept: Target acceptance probability for NUTS sampler.
            **sampler_kwargs: Additional keyword arguments passed to pm.sample().
        
        Returns:
            Tuple[az.InferenceData, az.InferenceData, az.InferenceData]: 
            A tuple containing prior checks, the main trace (posterior), and posterior predictive checks.
        """
        if self.model_instance.model is None:
            print("Building model...")
            self.build_model()

        with self.model_instance.model:
            # --- Prior Predictive Checks ---
            print("Sampling prior predictive...")
            self.prior = pm.sample_prior_predictive()
            # --- End Prior Predictive Checks ---

            # --- Main Sampling (Posterior) ---
            print("Sampling posterior...")
            self.trace = pm.sample(
                draws=draws, 
                tune=tune, 
                target_accept=target_accept,
            **sampler_kwargs
        )
            # --- End Main Sampling ---
            
            # --- Posterior Predictive Checks --- 
            # This is now handled explicitly in fit_metrics or if called separately.
            # --- End Posterior Predictive Checks ---
            
            # --- Construct final InferenceData --- 
            # OBSERVED DATA IS ADDED AUTOMATICALLY BY pm.sample based on likelihood's 'observed' arg
            # We will verify/rely on that in calculate_fit_metrics
            # REMOVED manual population of observed_data_dict and add_groups call
            # observed_data_dict = {}
            # if hasattr(self.model_instance, 'data_containers'):
            # ... (removed code) ...
            # else:
            #      print("Warning: Model instance has no data_containers attribute. Observed data might be incomplete.")

            print("Constructing final InferenceData...") # Adjusted print
            try:
                 # Combine prior and posterior (trace)
                 # Observed data should already be in self.trace from pm.sample
                 self.trace.extend(self.prior)
                 
                 # REMOVED: self.trace.add_groups(observed_data=observed_data_dict)

                 # Add constant data if available (check model instance)
                 # CONSTANT DATA IS ALSO LIKELY ADDED AUTOMATICALLY BY PYMC
                 # REMOVING manual addition to prevent errors
                 # if hasattr(self.model_instance, 'data_containers'):
                 #      constant_data_to_add = {}
                 #      potential_constants = [
                 #           "calendar_time_poll_idx", "pollster_idx", "poll_cycle_idx",
                 #           "calendar_time_result_district_idx", "result_cycle_district_idx", "result_district_idx",
                 #           "calendar_time_cycle_idx", "poll_days_numeric", "calendar_time_days_numeric"
                 #      ]
                 #      for const_key in potential_constants:
                 #           if const_key in self.model_instance.data_containers:
                 #                constant_data_to_add[const_key] = self.model_instance.data_containers[const_key].eval()
                 #      
                 #      if constant_data_to_add:
                 #           self.trace.add_groups(constant_data=constant_data_to_add)
                 #           print(f"DEBUG FACADE: Added constant_data with keys: {list(constant_data_to_add.keys())}")
                 
                 print("Final InferenceData constructed successfully.")
                 
            except Exception as idata_err:
                 print(f"Error constructing final InferenceData: {idata_err}")
                 # Fallback: trace might still be usable but potentially missing groups

        # Assign posterior separately if needed (currently not sampled here)
        self.posterior = None # Explicitly set to None as it's not sampled here

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
                
                # --- Debug coordinates before saving (Checking specific groups) --- 
                print("DEBUG SAVE: Inspecting trace coords before saving...")
                coord_found_save = False
                if hasattr(self.trace, 'posterior') and hasattr(self.trace.posterior, 'coords') and 'calendar_time' in self.trace.posterior.coords:
                     print("  DEBUG SAVE: calendar_time FOUND in posterior.coords")
                     coord_found_save = True
                     # Optionally print type/value again if needed
                     # pre_save_coords = self.trace.posterior.coords['calendar_time'].values
                     # print(f"  DEBUG SAVE: Dtype: {pre_save_coords.dtype}")
                else:
                     print("  DEBUG SAVE: calendar_time NOT found in posterior.coords")
                     
                if hasattr(self.trace, 'constant_data') and hasattr(self.trace.constant_data, 'coords') and 'calendar_time' in self.trace.constant_data.coords:
                     print("  DEBUG SAVE: calendar_time FOUND in constant_data.coords")
                     coord_found_save = True
                     # Optionally print type/value again if needed
                else:
                     print("  DEBUG SAVE: calendar_time NOT found in constant_data.coords")
                     
                if not coord_found_save:
                     print("  DEBUG SAVE: calendar_time NOT found in posterior or constant_data coords before saving.")
                # --- End debug --- 
                
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
    
    def get_latent_popularity(self, date_mode: str = 'election_day', district: Optional[str] = None) -> Optional[xr.DataArray]:
        """
        Extracts the posterior distribution of latent popularity at a specific time point,
        potentially adjusted for a specific district.

        Args:
            date_mode (str): Defines the time point: 'election_day', 'last_poll', 'today', or 'specific_date'.
                             If 'specific_date', `self.election_date` is used implicitly.
            district (str, optional): Name of the district to get adjusted popularity for.
                                      If None, returns the national latent popularity.

        Returns:
            xr.DataArray or None: Posterior samples of latent popularity for the specified date.
                                  Dimensions: (chain, draw, parties_complete).
                                  Returns None on error or if data/coords are missing.
        """
        if self.trace is None or 'posterior' not in self.trace:
            print("Error: Valid InferenceData with posterior samples required."); return None

        # --- Determine Target Date based on mode --- 
        actual_target_date = None
        try:
            if date_mode == 'election_day':
                if hasattr(self, 'election_date') and self.election_date:
                    actual_target_date = pd.Timestamp(self.election_date).normalize()
                    print(f"DEBUG get_latent: Using election_day: {actual_target_date.date()}")
                else:
                    print("Error: election_date not available for 'election_day' mode."); return None
            elif date_mode == 'last_poll':
                if hasattr(self.dataset, 'polls_train') and not self.dataset.polls_train.empty:
                    last_poll_date = pd.to_datetime(self.dataset.polls_train['date']).max()
                    actual_target_date = pd.Timestamp(last_poll_date).normalize()
                    print(f"DEBUG get_latent: Using last_poll date: {actual_target_date.date()}")
                else:
                    print("Error: Cannot determine last poll date from dataset."); return None
            elif date_mode == 'today':
                actual_target_date = pd.Timestamp.now().normalize()
                print(f"DEBUG get_latent: Using today's date: {actual_target_date.date()}")
            elif date_mode == 'specific_date': # Use election_date if mode is specific
                if hasattr(self, 'election_date') and self.election_date:
                    actual_target_date = pd.Timestamp(self.election_date).normalize()
                    print(f"DEBUG get_latent: Using specific_date (from election_date): {actual_target_date.date()}")
                else:
                    print("Error: election_date not available for 'specific_date' mode."); return None
            else:
                print(f"Error: Invalid date_mode '{date_mode}'."); return None
        except Exception as date_err:
            print(f"Error determining target date: {date_err}"); return None
        # --- End Determine Target Date ---

        # --- Get National Latent Popularity (already softmaxed) at Target Date --- 
        national_pop_var = "latent_popularity_national" # Use the new deterministic variable
        if national_pop_var not in self.trace.posterior:
            print(f"Error: National popularity variable '{national_pop_var}' not found in trace posterior."); return None
        if 'calendar_time' not in self.trace.posterior.coords:
             print("Error: Coordinate 'calendar_time' not found."); return None

        try:
            national_pop_da = self.trace.posterior[national_pop_var].copy()
            # Ensure coordinate is datetime and normalized
            calendar_coords = pd.to_datetime(national_pop_da['calendar_time'].values).normalize()
            national_pop_da['calendar_time'] = calendar_coords
            min_cal_date = calendar_coords.min()
            max_cal_date = calendar_coords.max()

            # Select the nearest date
            national_pop_at_date = national_pop_da.sel(
                calendar_time=actual_target_date,
                method="nearest",
                tolerance=pd.Timedelta(days=1) # Allow slight tolerance
            )
            selected_date = pd.Timestamp(national_pop_at_date.calendar_time.item()).normalize()
            if selected_date != actual_target_date:
                 print(f"Warning: Using nearest date {selected_date.date()} for target {actual_target_date.date()}")
            print(f"Selected national popularity for date: {selected_date.date()}")
        except Exception as e:
            print(f"Error selecting national popularity at {actual_target_date.date()}: {e}"); return None
        # --- End Get National Latent Popularity ---

        # --- District Adjustment (Placeholder/Future Enhancement) ---
        if district is not None:
            # IMPORTANT: Adjusting post-softmax probabilities is complex. 
            # Correct approach requires adding district effects *before* softmax in the model.
            # Current model structure adds district effects *before* softmax for the likelihood,
            # but we don't have a separate deterministic variable for district-level softmaxed popularity.
            print(f"Warning: District adjustment requested for '{district}', but getting district-specific popularity" 
                  " from the current trace requires recalculation or a dedicated model variable." 
                  " Returning NATIONAL popularity for now.")
            # For now, we just return the national popularity calculated above.
            # Future: Implement logic to either recalculate using latent means + district effects + softmax,
            # or add a pm.Deterministic("latent_popularity_district", ...) to the model.
            pass # Fall through to return national_pop_at_date
        # --- End District Adjustment --- 

        return national_pop_at_date
    
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
    
    def generate_diagnostic_plots(self, directory: str = "."):
        """
        Generates various diagnostic plots and a summary text file for the model trace.
        Focuses on trace plots per variable category for better readability.
        """
        if self.trace is None:
            print("Error: Trace object not found. Cannot generate diagnostic plots.")
            return
        if not hasattr(self.trace, 'posterior') or self.trace.posterior.sizes['draw'] == 0:
             print("Error: Posterior samples not found in trace. Cannot generate diagnostic plots.")
             return
        
        os.makedirs(directory, exist_ok=True)
        print(f"Generating diagnostic plots in: {directory}")

        # --- Temporarily increase Matplotlib warning limits ---
        original_max_warning = plt.rcParams.get('figure.max_open_warning', 20) # Get current or default
        # original_max_subplots = plt.rcParams.get('plot.max_subplots', 40) # Invalid key, removed
        new_max_warning = 150 # Set a higher limit for open figures warning
        try:
            plt.rcParams['figure.max_open_warning'] = new_max_warning
            # plt.rcParams['plot.max_subplots'] = new_max_subplots # Invalid key, removed
            print(f"Temporarily increased figure.max_open_warning to {new_max_warning}")
        except KeyError as e:
             print(f"Warning: Could not set rcParams: {e}. Using defaults.")


        # --- Categorize Variables ---
        # Identify variables present in the posterior trace
        present_vars = list(self.trace.posterior.data_vars)

        # Define categories (can be customized)
        var_categories = {
            'baseline_gp': [v for v in present_vars if 'baseline_gp' in v and '_raw' not in v],
            'short_term_gp': [v for v in present_vars if 'short_term_gp' in v and '_raw' not in v],
            'house_effects': [v for v in present_vars if 'house_effect' in v and '_raw' not in v], # Catch house_effects_sd too
            'district_effects': [v for v in present_vars if 'district_effect' in v and '_raw' not in v],
            'concentration': [v for v in present_vars if 'concentration_' in v],
            'raw_gp_coefs': [v for v in present_vars if '_gp_coef_raw' in v],
            'other': [] # Catch-all for remaining variables
        }

        # Assign remaining variables to 'other' category
        categorized_vars = set(v for cat_list in var_categories.values() for v in cat_list)
        var_categories['other'] = [v for v in present_vars if v not in categorized_vars and 'noisy_mu' not in v] # Exclude noisy_mu for now
        # Special handling for potentially huge variables like noisy_mu
        if 'noisy_mu_polls' in present_vars:
            var_categories.setdefault('large_likelihood_params', []).append('noisy_mu_polls')


        # Filter categories to only include those with present variables
        present_vars_by_category = {cat: var_list for cat, var_list in var_categories.items() if var_list}
        print("\nVariable Categories for Plotting:")
        for cat, var_list in present_vars_by_category.items():
            print(f"  {cat}: {var_list}")


        # --- Generate Plots ---

        # 1. Basic Trace Plot (Key Parameters)
        # Select a smaller subset of key parameters for the main trace plot
        key_params = []
        key_params.extend(present_vars_by_category.get('baseline_gp', []))
        key_params.extend(present_vars_by_category.get('short_term_gp', []))
        key_params.extend([v for v in present_vars_by_category.get('house_effects', []) if 'sd' in v]) # Only SDs
        key_params.extend([v for v in present_vars_by_category.get('district_effects', []) if 'sd' in v]) # Only SDs
        key_params.extend(present_vars_by_category.get('concentration', []))
        # Ensure we only include variables actually present
        key_params = [p for p in key_params if p in present_vars]

        if key_params:
            print(f"\n- Generating main trace plot for {len(key_params)} key parameters...")
            trace_plot_path = os.path.join(directory, "diagnostic_trace_plot_main.png")
            try:
                az.plot_trace(self.trace, var_names=key_params, compact=True, figsize=(12, 2 * len(key_params)))
                plt.suptitle("Trace Plot: Key Parameters", y=1.02)
            except Exception as e:
                print(f"  Error generating main trace plot: {e}")
        else:
             print("\n- Skipping main trace plot (no key parameters identified).")

        # 2. Energy Plot
        print("- Generating energy plot...")
        energy_plot_path = os.path.join(directory, "diagnostic_energy_plot.png")
        try:
            az.plot_energy(self.trace)
            plt.suptitle("Energy Plot")
            plt.savefig(energy_plot_path)
            plt.close()
            print(f"  Saved energy plot to {energy_plot_path}")
        except Exception as e:
            print(f"  Error generating energy plot: {e}")

        # 3. Pair Plot (Small subset of scalar parameters)
        # Select only scalar parameters (no dimensions other than chain/draw)
        scalar_params = []
        for cat, var_list in present_vars_by_category.items():
            if cat == 'raw_gp_coefs' or cat == 'large_likelihood_params': continue # Skip large/raw coefs
            for var_name in var_list:
                 try:
                     dims = self.trace.posterior[var_name].dims
                     if len(dims) == 2 and 'chain' in dims and 'draw' in dims:
                           scalar_params.append(var_name)
                 except Exception:
                     continue # Ignore if error accessing dims

        # Limit number of parameters for pair plot
        max_pair_plot_vars = 6
        if len(scalar_params) > max_pair_plot_vars:
             print(f"  (Pair plot limited to {max_pair_plot_vars} scalar params: {scalar_params[:max_pair_plot_vars]})")
             scalar_params = scalar_params[:max_pair_plot_vars]

        if scalar_params:
            print(f"- Generating pair plot for {len(scalar_params)} scalar parameters...")
            pair_plot_path = os.path.join(directory, "diagnostic_pair_plot.png")
            try:
                az.plot_pair(self.trace, var_names=scalar_params, kind='kde', divergences=True)
                plt.suptitle("Pair Plot (Scalar Parameters)")
                plt.savefig(pair_plot_path)
                plt.close()
                print(f"  Saved pair plot to {pair_plot_path}")
            except Exception as e:
                print(f"  Error generating pair plot: {e}")
        else:
            print("- Skipping pair plot (no suitable scalar parameters found).")

        # 4. NEW: Generate Trace Plots per Category
        print("- Generating trace plots per variable category...")
        for category, var_list in present_vars_by_category.items():
            if not var_list or category == 'large_likelihood_params': # Skip huge params category
                 print(f"  - Skipping category '{category}' (no variables or large params).")
                 continue

            # Filter out variables with excessive dimensions for standard trace plots
            # Limit based on the number of individual traces to plot (e.g., 8 parties * 20 districts = 160)
            MAX_ELEMENTS_FOR_TRACE = 50 # Adjust this limit as needed
            vars_to_plot = []
            skipped_vars = []
            for v in var_list:
                try:
                    # Calculate number of non-chain/draw elements
                    var_dims = self.trace.posterior[v].dims
                    num_elements = np.prod([self.trace.posterior[v].sizes[d] for d in var_dims if d not in ['chain', 'draw']])
                    if num_elements <= MAX_ELEMENTS_FOR_TRACE:
                        vars_to_plot.append(v)
                    else:
                        skipped_vars.append(f"{v} ({num_elements} elements)")
                except Exception as e:
                     print(f"    Warning: Could not check size for variable {v}: {e}")
                     skipped_vars.append(f"{v} (size check error)")


            if skipped_vars:
                 print(f"  - Skipping trace plot for some vars in category '{category}' due to large size: {skipped_vars}")

            if not vars_to_plot:
                 print(f"  - No suitable variables to plot trace for in category '{category}'.")
                 continue

            print(f"  - Generating trace plot for category: {category} (vars: {vars_to_plot})")
            trace_plot_path = os.path.join(directory, f"diagnostic_trace_plot_{category}.png")
            try:
                # Calculate a reasonable figsize. Base height + height per variable.
                num_vars_in_plot = len(vars_to_plot)
                fig_height = 4 + num_vars_in_plot * 1.5 # Adjust multiplier as needed
                az.plot_trace(self.trace, var_names=vars_to_plot, compact=True, figsize=(15, fig_height))
                plt.suptitle(f"Trace Plot: Category '{category}'", y=1.01, fontsize=14) # Adjust y slightly
                plt.tight_layout(rect=[0, 0.03, 1, 0.99]) # Adjust layout to prevent title overlap
                plt.savefig(trace_plot_path)
                plt.close()
                print(f"    Saved category trace plot to {trace_plot_path}")
            except Exception as e:
                print(f"    Error generating trace plot for category {category}: {e}")
        print("  Finished generating category trace plots.")

        # 5. Forest Plot for House Effects (Faceted by Party)
        if 'house_effects' in present_vars:
             print("- Generating forest plot for house_effects...")
             house_effects_plot_path = os.path.join(directory, "diagnostic_forest_plot_house_effects.png")
             try:
                 # Determine figsize based on number of pollsters AND parties
                 n_pollsters = len(self.trace.posterior.coords.get("pollsters", []))
                 n_parties = len(self.trace.posterior.coords.get("parties_complete", []))
                 fig_height_house = max(8, n_pollsters * n_parties * 0.4) # Increased base height and multiplied by n_parties
                 az.plot_forest(self.trace, var_names=['house_effects'], combined=True, figsize=(12, fig_height_house))
                 plt.suptitle("Forest Plot: House Effects (Faceted by Party)", y=1.02)
                 plt.tight_layout()
                 plt.savefig(house_effects_plot_path)
                 plt.close()
                 print(f"  Saved house effects forest plot to {house_effects_plot_path}")
             except Exception as e:
                 print(f"  Error generating house_effects forest plot: {e}")
        else:
            print("- Skipping house_effects forest plot (variable not found).")
            
        # 6. Forest Plot for District Effects (Faceted by Party)
        if 'district_effects' in present_vars:
             print("- Generating forest plot for district_effects...")
             district_effects_plot_path = os.path.join(directory, "diagnostic_forest_plot_district_effects.png")
             try:
                 # Determine figsize based on number of districts AND parties
                 n_districts = len(self.trace.posterior.coords.get("districts", []))
                 n_parties = len(self.trace.posterior.coords.get("parties_complete", [])) # Get number of parties
                 fig_height_district = max(10, n_districts * n_parties * 0.4) # Increased base height and multiplied by n_parties
                 az.plot_forest(self.trace, var_names=['district_effects'], combined=True, figsize=(12, fig_height_district))
                 plt.suptitle("Forest Plot: District Effects (Faceted by Party)", y=1.02)
                 plt.tight_layout()
                 plt.savefig(district_effects_plot_path)
                 plt.close()
                 print(f"  Saved district effects forest plot to {district_effects_plot_path}")
             except Exception as e:
                 print(f"  Error generating district_effects forest plot: {e}")
        else:
            print("- Skipping district_effects forest plot (variable not found).")

        # <<< Add Forest Plots for Base Offset and Beta >>>
        # 7. Forest Plot for Base Offset (Faceted by Party)
        base_offset_var = "base_offset_p"
        if base_offset_var in present_vars:
            print(f"- Generating forest plot for {base_offset_var}...")
            base_offset_plot_path = os.path.join(directory, f"diagnostic_forest_plot_{base_offset_var}.png")
            try:
                n_districts = len(self.trace.posterior.coords.get("districts", []))
                n_parties = len(self.trace.posterior.coords.get("parties_complete", []))
                fig_height = max(10, n_districts * n_parties * 0.4)
                az.plot_forest(self.trace, var_names=[base_offset_var], combined=True, figsize=(12, fig_height))
                plt.suptitle(f"Forest Plot: {base_offset_var} (Faceted by Party)", y=1.02)
                plt.tight_layout()
                plt.savefig(base_offset_plot_path)
                plt.close()
                print(f"  Saved {base_offset_var} forest plot to {base_offset_plot_path}")
            except Exception as e:
                print(f"  Error generating {base_offset_var} forest plot: {e}")
        else:
            print(f"- Skipping {base_offset_var} forest plot (variable not found).")

        # 8. Forest Plot for Beta Sensitivity (Faceted by Party)
        beta_var = "beta_p"
        if beta_var in present_vars:
            print(f"- Generating forest plot for {beta_var}...")
            beta_plot_path = os.path.join(directory, f"diagnostic_forest_plot_{beta_var}.png")
            try:
                n_districts = len(self.trace.posterior.coords.get("districts", []))
                n_parties = len(self.trace.posterior.coords.get("parties_complete", []))
                fig_height = max(10, n_districts * n_parties * 0.4)
                az.plot_forest(self.trace, var_names=[beta_var], combined=True, figsize=(12, fig_height))
                plt.suptitle(f"Forest Plot: {beta_var} (Faceted by Party)", y=1.02)
                plt.tight_layout()
                plt.savefig(beta_plot_path)
                plt.close()
                print(f"  Saved {beta_var} forest plot to {beta_plot_path}")
            except Exception as e:
                print(f"  Error generating {beta_var} forest plot: {e}")
        else:
            print(f"- Skipping {beta_var} forest plot (variable not found).")
        # <<< End Added Forest Plots >>>
        
        # 7. Generate Summary Text File (renumbered)
        summary_path = os.path.join(directory, "diagnostic_summary.txt")
        print(f"- Generating diagnostic summary text file...")
        try:
            # Calculate required ESS based on 10% of total samples
            # Default to 4000 if dimensions aren't available
            total_samples = 4000 # Fallback
            # Use .sizes instead of .dims to access dimension lengths
            if hasattr(self.trace, 'posterior') and 'chain' in self.trace.posterior.sizes and 'draw' in self.trace.posterior.sizes:
                 # Access lengths from the .sizes mapping
                 chain_dim = self.trace.posterior.sizes['chain']
                 draw_dim = self.trace.posterior.sizes['draw']
                 total_samples = int(chain_dim) * int(draw_dim)
            ess_threshold_pct = 0.10 # 10%
            ess_threshold = int(total_samples * ess_threshold_pct)
            if ess_threshold == 0: ess_threshold = 1 # Ensure threshold is at least 1

            summary = az.summary(self.trace)
            rhat_threshold = 1.01 # Set R-hat threshold to 1.01

            # Identify potentially problematic parameters based ONLY on R-hat
            bad_rhat = summary[summary['r_hat'] > rhat_threshold]
            # bad_ess_bulk = summary[summary['ess_bulk'] < ess_threshold] # Removed ESS check
            # bad_ess_tail = summary[summary['ess_tail'] < ess_threshold] # Removed ESS check

            # Combine indices of problematic parameters based ONLY on R-hat
            # bad_indices = set(bad_rhat.index) | set(bad_ess_bulk.index) | set(bad_ess_tail.index) # Original logic
            bad_indices = set(bad_rhat.index) # Only consider R-hat issues
            bad_summary = summary.loc[list(bad_indices)]

            with open(summary_path, 'w') as f:
                f.write("Convergence Diagnostics Summary (R-hat > 1.01)\\n")
                f.write("=============================================\\n")
                f.write(f"R-hat Threshold Used: > {rhat_threshold}\\n")
                # f.write(f"ESS Threshold Used for Summary (Bulk & Tail): < {ess_threshold} ({ess_threshold_pct:.0%} of {total_samples} total samples)\\n\\n") # Removed ESS info

                if bad_indices:
                    f.write("\\nParameters with Potential Convergence Issues (R-hat > 1.01):\\n")
                    f.write("------------------------------------------------------------------\\n")
                    # Select relevant columns, focusing on R-hat
                    problem_cols = ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat'] # Removed ESS cols
                    # Ensure columns exist before selecting
                    problem_cols = [col for col in problem_cols if col in bad_summary.columns]
                    f.write(bad_summary[problem_cols].to_string())
                    f.write("\\n\\n")
                else:
                    f.write("\\nNo parameters found with R-hat > 1.01.\\n\\n")

                # Append the full summary for reference -- REMOVED
                # f.write("Full ArviZ Summary:\\n")
                # f.write("==================\\n")
                # f.write(summary.to_string())

            print(f"  Saved diagnostic summary (R-hat > {rhat_threshold}) to {summary_path}")
        except Exception as e:
            print(f"  Error generating diagnostic summary: {e}")

        # --- Restore original Matplotlib settings ---
        try:
            plt.rcParams['figure.max_open_warning'] = original_max_warning
            # plt.rcParams['plot.max_subplots'] = original_max_subplots # Invalid key, removed
            print(f"Restored figure.max_open_warning to {plt.rcParams.get('figure.max_open_warning', 'default')}")
            # print(f"Restored plot.max_subplots to {plt.rcParams.get('plot.max_subplots', 'default')}\") # Invalid key, removed
        except KeyError as e:
             print(f"Warning: Could not restore rcParams: {e}.")
        finally:
             pass # Add finally clause to satisfy linter


        print("Diagnostic plot and summary generation complete.")