import abc
from typing import Dict, List, Tuple, Optional

import arviz
import arviz as az
import pandas as pd
import pymc as pm
import numpy as np

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
        self.results_mult = dataset.results_mult # Usually historical results + target election placeholder
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
        self.non_competing_polls_additive_base = np.where(is_here_polls, 0, -10).astype(np.int32)
        self.is_here_polls_base = is_here_polls.astype(int).to_numpy()

        # --- Non-competing masks for results (aligned with ALL election dates) ---
        # Reindex results_mult (which includes historical+target) to all election dates
        reindexed_results = (
            self.results_mult[self.political_families]
            .reindex(pd.to_datetime(self.all_election_dates))
        )
        # is_competing_mask is True only for parties with > 0 votes in historical results
        # NaN > 0 is False, so future date row will correctly be False here
        is_competing_mask = reindexed_results > 0
        self.non_competing_parties_results_base = np.where(is_competing_mask, 0, -10).astype(np.int32)

        print(f"Shape of non_competing_parties_results_base (aligned with all elections): {self.non_competing_parties_results_base.shape}")


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
                # Pass var_names to pm.sample to ensure deterministics are included in the main trace
                trace = pm.sample(var_names=var_names, **sampler_kwargs)
                # --- DEBUG PRINTS --- 
                print(f"DEBUG (BaseModel): Type of trace returned by pm.sample: {type(trace)}")
                if trace is None:
                     print("DEBUG (BaseModel): trace is None AFTER pm.sample call!")
                else:
                     print("DEBUG (BaseModel): trace seems to exist AFTER pm.sample call.")
                     if isinstance(trace, arviz.InferenceData):
                          print(f"DEBUG (BaseModel): Trace is InferenceData with groups: {list(trace.groups())}")
                     else:
                          print(f"DEBUG (BaseModel): Trace is NOT InferenceData.")
                # --- END DEBUG PRINTS --- 
                self.trace = trace # Store trace in the instance

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
                            
                # Include posterior predictive sampling within the try block as well
                print("Sampling posterior predictive...")
                if trace is not None: # Only sample post predictive if posterior sampling succeeded somewhat
                     try:
                         post_checks = pm.sample_posterior_predictive(
                             trace, var_names=var_names
                         )
                     except Exception as e_ppc: # Specific exception for PPC
                         print(f"Warning: Failed to sample posterior predictive: {e_ppc}")
                else:
                    print("Skipping posterior predictive sampling as posterior trace is missing.")

                # Debug print before returning
                print(f"DEBUG (BaseModel): Returning trace of type: {type(trace)}")
                return prior_checks, trace, post_checks
                
            # Catch exception for the whole sampling block
            except Exception as e: 
                print(f"ERROR: Exception during posterior sampling or post-processing: {e}")
                # Ensure trace is None if an error occurred after pm.sample potentially assigned it
                self.trace = None 
                trace = None # Ensure local trace is also None before returning
                # Still try to return something, even if trace failed
                print(f"DEBUG (BaseModel): Returning None trace due to exception.")
                return prior_checks, None, post_checks

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