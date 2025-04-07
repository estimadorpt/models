import abc
from typing import Dict, List, Tuple

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
        Initialize the base model.

        Parameters
        ----------
        dataset : ElectionDataset
            The dataset object containing polls, results, etc.
        **kwargs : dict
            Additional model-specific configuration parameters.
        """
        self.dataset = dataset
        self.coords: Dict = {}
        self.data_containers: Dict = {}
        # Store kwargs for potential use by subclasses
        self.model_config = kwargs
        
        # Initialize attributes expected by sample_all and potentially other methods
        self.model: Optional[pm.Model] = None # To store the built PyMC model
        self.trace: Optional[az.InferenceData] = None # To store the posterior trace


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
                trace = pm.sample(**sampler_kwargs)
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
            except Exception as e:
                print(f"ERROR: Failed during posterior sampling: {e}")
                # Decide whether to continue or raise - for now, let's try post predictive if trace exists
            
            print("Sampling posterior predictive...")
            if trace is not None: # Only sample post predictive if posterior sampling succeeded somewhat
                 try:
                     post_checks = pm.sample_posterior_predictive(
                         trace, var_names=var_names
                     )
                 except Exception as e:
                     print(f"Warning: Failed to sample posterior predictive: {e}")
            else:
                print("Skipping posterior predictive sampling as posterior trace is missing.")

        return prior_checks, trace, post_checks

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