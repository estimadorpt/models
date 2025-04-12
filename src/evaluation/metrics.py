# src/evaluation/metrics.py

import numpy as np
import xarray as xr
from scipy.stats import multinomial
from typing import Dict


def calculate_mae(predicted_probs: np.ndarray, observed_probs: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error between predicted and observed probabilities.

    Args:
        predicted_probs: Array of predicted probabilities (n_observations, n_categories).
                         Should be the mean or median of the posterior predictive distribution.
        observed_probs: Array of observed probabilities/proportions (n_observations, n_categories).

    Returns:
        The Mean Absolute Error.
    """
    if predicted_probs.shape != observed_probs.shape:
        raise ValueError("Predicted and observed probability shapes must match.")
    # Calculate absolute errors per observation and category
    absolute_errors = np.abs(predicted_probs - observed_probs)
    # Average over all observations and categories
    return np.mean(absolute_errors)

def calculate_rmse(predicted_probs: np.ndarray, observed_probs: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error between predicted and observed probabilities.

    Args:
        predicted_probs: Array of predicted probabilities (n_observations, n_categories).
                         Should be the mean or median of the posterior predictive distribution.
        observed_probs: Array of observed probabilities/proportions (n_observations, n_categories).

    Returns:
        The Root Mean Squared Error.
    """
    if predicted_probs.shape != observed_probs.shape:
        raise ValueError("Predicted and observed probability shapes must match.")
    # Calculate squared errors per observation and category
    squared_errors = (predicted_probs - observed_probs)**2
    # Average over all observations and categories, then take sqrt
    return np.sqrt(np.mean(squared_errors))

def calculate_log_score(predicted_probs: np.ndarray, observed_counts: np.ndarray) -> np.ndarray:
    """
    Calculates the Log Score (negative log-likelihood) for each multinomial outcome.

    Uses the mean predicted probabilities for the calculation.

    Args:
        predicted_probs: Array of predicted probabilities (n_observations, n_categories).
                         This should represent the mean probability vector for each observation.
        observed_counts: Array of observed counts (n_observations, n_categories).

    Returns:
        An array of Log Scores (negative log-likelihoods) for each observation (lower is better, but
        this function returns the negative, so higher is worse for direct comparison).
        Contains np.inf if any predicted probability is zero where observed count is non-zero.
    """
    if predicted_probs.shape != observed_counts.shape:
        raise ValueError("Predicted probability and observed count shapes must match.")
    if not np.allclose(np.sum(predicted_probs, axis=1), 1.0, atol=1e-6):
         print("Warning: Predicted probabilities do not sum to 1 for all observations.")
         # Normalize just in case, though this indicates an upstream issue
         predicted_probs = predicted_probs / predicted_probs.sum(axis=1, keepdims=True)

    total_log_likelihood = 0
    n_observations = observed_counts.shape[0]
    log_scores = np.zeros(n_observations)

    for i in range(n_observations):
        # Get total N for this observation
        n_total = np.sum(observed_counts[i])
        if n_total == 0:
            continue # Skip observations with zero total count

        # Get probabilities for this observation
        probs_i = predicted_probs[i]

        # Avoid log(0) issues - small probability assigned if needed
        # Check if any prob is zero where counts are non-zero
        if np.any((probs_i <= 0) & (observed_counts[i] > 0)):
            # Cannot calculate log-likelihood, return infinity or handle as error
            # For scoring, a very large penalty is appropriate.
             print(f"Warning: Zero probability predicted for non-zero observed count at index {i}.")
             return np.inf # Indicate impossible prediction

        # Use scipy's multinomial logpmf for numerical stability
        log_likelihood_i = multinomial.logpmf(x=observed_counts[i], n=n_total, p=probs_i)
        # Store the negative log-likelihood (log score)
        log_scores[i] = -log_likelihood_i

    # Return the array of individual log scores
    return log_scores

def calculate_rps(predicted_probs: np.ndarray, observed_probs: np.ndarray) -> float:
    """
    Calculates the average Rank Probability Score (RPS).

    Compares the cumulative probability distributions.

    Args:
        predicted_probs: Array of predicted probabilities (n_observations, n_categories).
        observed_probs: Array of observed probabilities/proportions (n_observations, n_categories).
                        (Can also be {0, 1} for the winning category, but proportions work too).

    Returns:
        The average Rank Probability Score (lower is better).
    """
    if predicted_probs.shape != observed_probs.shape:
        raise ValueError("Predicted and observed probability shapes must match.")

    n_observations, n_categories = predicted_probs.shape

    # Calculate cumulative probabilities
    predicted_cdfs = np.cumsum(predicted_probs, axis=1)
    observed_cdfs = np.cumsum(observed_probs, axis=1)

    # Ensure the last cumulative probability is 1 (handle potential floating point issues)
    predicted_cdfs[:, -1] = 1.0
    observed_cdfs[:, -1] = 1.0

    # Calculate squared differences between CDFs
    squared_diff_cdfs = (predicted_cdfs - observed_cdfs)**2

    # Sum squared differences for each observation
    rps_per_observation = np.sum(squared_diff_cdfs, axis=1)

    # Average RPS over all observations
    return np.mean(rps_per_observation)

def calculate_calibration_data(predicted_probs: np.ndarray, observed_probs: np.ndarray, n_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    Calculates data needed for a reliability diagram (marginal calibration).

    Bins all individual party probability predictions and compares the average
    predicted probability in each bin to the average observed proportion.

    Args:
        predicted_probs: Array of predicted probabilities (n_observations, n_categories).
                         Should be the mean or median of the posterior predictive distribution.
        observed_probs: Array of observed probabilities/proportions (n_observations, n_categories).
        n_bins: Number of bins to divide the [0, 1] probability range into.

    Returns:
        A dictionary containing:
        - 'mean_predicted_prob': Average predicted probability in each bin.
        - 'mean_observed_prob': Average observed proportion in each bin.
        - 'bin_counts': Number of predictions falling into each bin.
        - 'bin_edges': The edges of the bins used.
    """
    if predicted_probs.shape != observed_probs.shape:
        raise ValueError("Predicted and observed probability shapes must match.")

    # Flatten arrays to treat each party prediction independently
    flat_pred_probs = predicted_probs.flatten()
    flat_obs_probs = observed_probs.flatten()

    # Define bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Add small epsilon to the upper bound to include 1.0 in the last bin
    bin_edges[-1] += 1e-8 
    
    # Digitize predictions into bins
    # np.digitize returns indices starting from 1, so subtract 1 for 0-based indexing
    bin_indices = np.digitize(flat_pred_probs, bin_edges) - 1

    # Ensure indices are within bounds (especially for values exactly at the edges)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_predicted_prob = np.zeros(n_bins)
    mean_observed_prob = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        in_bin = (bin_indices == i)
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            mean_predicted_prob[i] = np.mean(flat_pred_probs[in_bin])
            mean_observed_prob[i] = np.mean(flat_obs_probs[in_bin])
        else:
            # Handle empty bins - assign NaN or keep as 0? Let's use NaN
            mean_predicted_prob[i] = np.nan
            mean_observed_prob[i] = np.nan
            
    # Return bin midpoints as well for plotting if needed?
    # bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2 

    return {
        "mean_predicted_prob": mean_predicted_prob,
        "mean_observed_prob": mean_observed_prob,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges[:-1] # Return lower edges
    }

# TODO: Add functions for calibration assessment (e.g., reliability diagrams) 