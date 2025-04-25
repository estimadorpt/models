import abc
from abc import abstractmethod
from typing import Dict, List, Tuple, Optional

import arviz
import arviz as az
import pandas as pd
import pymc as pm
import numpy as np
import xarray as xr

from src.data.dataset import ElectionDataset


class BaseElectionModel(abc.ABC):
    """Abstract base class for Bayesian election models."""

    def __init__(self, dataset: ElectionDataset, **kwargs):
        """
        Initialize the base model and prepare common data structures.

        Parameters
        ----------
        dataset : ElectionDataset
            The dataset object containing polls, results, etc.
        **kwargs : dict
            Additional model-specific configuration parameters.
        """
        self.dataset = dataset
        self.model_config = kwargs # Store kwargs for potential use by subclasses

        # --- Store common dataset attributes ---
        self.political_families = dataset.political_families
        self.all_election_dates = dataset.all_election_dates
        self.historical_election_dates = dataset.historical_election_dates
        self.polls_train = dataset.polls_train
        self.results_national = dataset.results_national # Usually historical results + target election placeholder
        self.results_oos = dataset.results_oos # Strictly historical results for likelihood evaluation
        self.government_status = dataset.government_status
        self.campaign_preds = dataset.campaign_preds # Predictors aligned with polls
        self.results_preds = dataset.results_preds # Predictors aligned with results

        # --- Initialize attributes for model components ---
        self.coords: Dict = {}
        self.data_containers: Dict = {}
        self.model: Optional[pm.Model] = None # To store the built PyMC model
        self.trace: Optional[az.InferenceData] = None # To store the posterior trace

        # --- Prepare common data structures used by most models ---
        self._prepare_common_data()

    def _prepare_common_data(self):
        """
        Prepare common data structures like non-competing masks.
        These are calculated once and stored as attributes.
        """
        # --- Non-competing masks for polls (based on polls_train) ---
        polls = self.polls_train # Use the training polls for the base masks
        is_here_polls = polls[self.political_families] > 0
        self.non_competing_polls_additive_base = np.where(is_here_polls, 0, -100).astype(np.int32)
        self.is_here_polls_base = is_here_polls.astype(int).to_numpy()

        # --- Non-competing masks for results (aligned with ALL election dates) ---
        # Reindex results_national (which includes historical+target) to all election dates
        reindexed_results = (
            self.results_national[self.political_families]
            .reindex(pd.to_datetime(self.all_election_dates))
        )
        # is_competing_mask is True only for parties with > 0 votes in historical results
        # NaN > 0 is False, so future date row will correctly be False here
        is_competing_mask = reindexed_results > 0
        self.non_competing_parties_results_base = np.where(is_competing_mask, 0, -100).astype(np.int32)

        print(f"Shape of non_competing_parties_results_base (aligned with all elections): {self.non_competing_parties_results_base.shape}")

        # --- Non-competing masks for calendar time (aligned with ALL calendar dates) ---
        # Calculate unique calendar dates needed for this mask
        poll_dates_dt = pd.to_datetime(self.polls_train['date']).unique()
        all_election_dates_dt = pd.to_datetime(self.all_election_dates).unique()
        calendar_dates_dt = pd.to_datetime(np.union1d(poll_dates_dt, all_election_dates_dt)).unique()
        calendar_dates_dt = calendar_dates_dt.sort_values()
        self.all_calendar_dates_dt = calendar_dates_dt # Store if needed elsewhere

        # Determine first appearance date for each party based on results_national > 0
        first_appearance = self.results_national[self.political_families].gt(0).idxmax()
        # Create a boolean mask: True if calendar date >= first appearance date
        is_present_calendar = pd.DataFrame(
            {party: calendar_dates_dt >= first_appearance[party] for party in self.political_families},
            index=calendar_dates_dt
        )
        self.non_competing_calendar_additive_base = np.where(is_present_calendar, 0, -100).astype(np.int32)
        self.is_non_competing_calendar_mask_base = ~is_present_calendar.to_numpy() # Boolean: True if NOT competing
        print(f"Shape of non_competing_calendar_additive_base (aligned with calendar time): {self.non_competing_calendar_additive_base.shape}")

    @abc.abstractmethod
    def _build_coords(self, polls: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, List[int]]:
        """
        Build the coordinates required for the PyMC model.

        Must be implemented by subclasses.

        Parameters
        ----------
        polls : pd.DataFrame, optional
            Poll data to use for building coordinates. If None, uses training data.

        Returns
        -------
        tuple
            Contains pollster_id, countdown_id, election_id, COORDS dictionary, 
            and observed_election_indices.
        """
        pass

    @abc.abstractmethod
    def _build_data_containers(self, polls: pd.DataFrame = None) -> Dict:
        """
        Build the data containers (pm.Data) for the PyMC model.

        Must be implemented by subclasses.

        Parameters
        ----------
        polls : pd.DataFrame, optional
            Poll data to use. If None, use the training data.

        Returns
        -------
        dict
            Dictionary mapping container names to pm.Data objects.
        """
        pass

    @abc.abstractmethod
    def build_model(self, polls: pd.DataFrame = None) -> pm.Model:
        """
        Build the specific PyMC model structure.

        Must be implemented by subclasses.

        Parameters
        ----------
        polls : pd.DataFrame, optional
            Poll data to incorporate into the model build. If None, use training data.

        Returns
        -------
        pm.Model
            The constructed PyMC model object.
        """
        pass

    def sample_all(
        self, *, model: pm.Model = None, var_names: List[str], **sampler_kwargs
    ):
        """
        Sample the model (prior, posterior, posterior predictive) and return the traces.

        Parameters
        ----------
        model : pm.Model, optional
            A model previously created using `self.build_model()`. 
            Builds a new model if None (default).
        var_names: List[str]
            Variables names passed to `pm.sample_posterior_predictive`.
        **sampler_kwargs : dict
            Additional arguments to `pm.sample`.

        Returns
        -------
        tuple
            (prior_checks, trace, post_checks) containing arviz.InferenceData objects.
        """
        if model is None:
            if self.model is None: # Build model if not already built
                print("Building model within sample_all...")
                self.model = self.build_model() 
            model = self.model # Use the instance's model

        # Set defaults for common parameters if not already specified
        sampler_kwargs.setdefault('nuts_sampler', 'numpyro')
        sampler_kwargs.setdefault('return_inferencedata', True)
        
        # Increase defaults for draws and tune if not specified
        sampler_kwargs.setdefault('draws', 3000)
        sampler_kwargs.setdefault('tune', 3000)
        
        # Add chain initialization strategy if not specified
        if 'init' not in sampler_kwargs:
            sampler_kwargs['init'] = 'jitter+adapt_diag'
        
        # Set number of chains if not specified
        sampler_kwargs.setdefault('chains', 4)
        
        # Set max tree depth
        sampler_kwargs.setdefault('max_treedepth', 15)
        
        # Recommend sampling parameters for better performance
        sampler_kwargs.setdefault('target_accept', 0.95)
        
        prior_checks = None
        trace = None
        post_checks = None

        with model:
            print("Sampling prior predictive...")
            try:
                prior_checks = pm.sample_prior_predictive()
            except Exception as e:
                print(f"Warning: Failed to sample prior predictive: {e}")

            print("Sampling posterior...")
            try:
                # Define essential deterministic variables to track
                vars_to_track = [
                    "latent_popularity_calendar_trajectory", # Already plotted
                    "latent_mu_calendar", # Raw latent score sum
                    "baseline_effect_calendar", # Baseline GP contribution
                    "short_term_effect_calendar", # Short-term GP contribution
                    "latent_mu_polls", # Penalized latent score for polls
                    "noisy_mu_polls", # Penalized + house effect for polls
                    "noisy_popularity_polls", # Final poll probability
                    "latent_mu_results", # Penalized latent score for results
                    "latent_pop_results" # Final result probability
                    # Add any other deterministics you might want to inspect
                ]

                # Filter to only variables present in the model
                model_det_names = [v.name for v in model.deterministics]
                vars_actually_in_model = [v for v in vars_to_track if v in model_det_names]

                if not vars_actually_in_model:
                    print("Warning: None of the specified vars_to_track are in the model's deterministics.")

                # Sample using InferenceData=True, explicitly requesting deterministics
                # Note: track_deterministics is relatively new, might need PyMC v5+
                # If older PyMC, this might not work, and we might need return_inferencedata=False + manual conversion
                try:
                    trace = pm.sample(
                        **sampler_kwargs
                    )
                    # print(f"DEBUG (BaseModel): Sampling attempted.")
                except Exception as e:
                    # Keep general exception handling
                    print(f"ERROR: Exception during posterior sampling: {e}")
                    self.trace = None
                    trace = None
                    # print(f"DEBUG (BaseModel): Returning None trace due to exception.")
                    return prior_checks, None, None

                # print(f"DEBUG (BaseModel): Type of trace returned by pm.sample: {type(trace)}")
                if not isinstance(trace, az.InferenceData):
                    print("Error: pm.sample did not return InferenceData as expected.")
                    return prior_checks, None, None # Return None trace

                # Check if variables were actually saved
                if "posterior" in trace:
                    saved_vars = list(trace.posterior.data_vars)
                    # print(f"DEBUG (BaseModel): Variables actually saved in posterior: {sorted(saved_vars)}")
                    missing_vars = set(vars_actually_in_model) - set(saved_vars)
                    if missing_vars:
                         print(f"Warning: The following requested deterministics were NOT saved in posterior: {missing_vars}")
                         # Potentially try adding them from posterior_predictive if sampled separately later?
                else:
                     print("Warning: No posterior group found in returned InferenceData.")


                # Store additional convergence diagnostics
                if trace is not None and isinstance(trace, arviz.InferenceData):
                    # Check for divergences
                    n_divergent = trace.sample_stats.diverging.sum().item()
                    if n_divergent > 0:
                        print(f"WARNING: {n_divergent} divergent transitions detected")

                    # Check if any parameters hit max tree depth frequently
                    if 'tree_depth' in trace.sample_stats:
                        max_depths = (trace.sample_stats.tree_depth >= sampler_kwargs.get('max_treedepth', 10)).sum().item()
                        if max_depths > 0:
                            pct_max_depth = max_depths / (trace.posterior.dims['chain'] * trace.posterior.dims['draw'])
                            print(f"WARNING: {max_depths} samples ({pct_max_depth:.1%}) reached maximum tree depth")

                # Debug print before returning
                # print(f"DEBUG (BaseModel): Returning trace of type: {type(trace)}")
                return prior_checks, trace, None # Post checks TBD / maybe sampled later

            except Exception as e:
                # Keep general exception handling
                print(f"ERROR: Exception during posterior sampling: {e}")
                self.trace = None
                trace = None
                # print(f"DEBUG (BaseModel): Returning None trace due to exception.")
                return prior_checks, None, None

    # --- Prediction/Nowcasting methods might be specific to models ---
    # Subclasses should implement their own versions if needed, 
    # tailored to their specific model structure and variables.
    # Example signatures (could be made abstract if a common interface is desired later):
    
    # def nowcast_party_support(self, idata, current_polls, latest_election_date, **kwargs):
    #     raise NotImplementedError("Nowcasting needs to be implemented by the specific model subclass.")

    # def predict(self, oos_data):
    #     raise NotImplementedError("Prediction needs to be implemented by the specific model subclass.")
        
    # def predict_history(self, elections_to_predict):
    #      raise NotImplementedError("Historical prediction needs to be implemented by the specific model subclass.")

    # def predict_latent_trajectory(self, idata, start_date, end_date):
    #     raise NotImplementedError("Latent trajectory prediction needs to be implemented by the specific model subclass.") 

    @abstractmethod
    def get_latent_popularity(self, idata: az.InferenceData, target_date: pd.Timestamp) -> xr.DataArray:
        """
        Abstract method to extract the posterior distribution of latent popularity
        at a specific target date.

        Args:
            idata (az.InferenceData): The InferenceData object containing the posterior samples.
            target_date (pd.Timestamp): The specific date for which to extract the popularity.

        Returns:
            xr.DataArray: Posterior samples of latent popularity at the target_date.
                          Dimensions should typically include (chain, draw, parties).
        """
        pass 