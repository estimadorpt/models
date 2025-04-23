# src/evaluation/metrics.py

import numpy as np
import xarray as xr
from scipy.stats import multinomial
from typing import Dict


def calculate_mae(predicted_probs: np.ndarray, observed_counts: np.ndarray, total_n: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error between predicted probabilities and observed proportions.

    Args:
        predicted_probs: Array of predicted probabilities (n_observations, n_categories).
                         Should be the mean or median of the posterior predictive distribution.
        observed_counts: Array of observed counts (n_observations, n_categories).
        total_n: Array of total counts for each observation (n_observations,).

    Returns:
        The Mean Absolute Error. Returns np.nan if shapes mismatch or calculation fails.
    """
    if predicted_probs.shape != observed_counts.shape or predicted_probs.shape[0] != total_n.shape[0]:
        print(f"Error: Shape mismatch in calculate_mae. pred: {predicted_probs.shape}, obs: {observed_counts.shape}, N: {total_n.shape}")
        return np.nan # Return NaN on shape mismatch

    # Calculate observed probabilities, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ensure total_n is float for division and add new axis for broadcasting
        observed_probs = observed_counts / total_n.astype(float)[:, np.newaxis]
    observed_probs = np.nan_to_num(observed_probs) # Replace NaNs (from 0/0) with 0

    # Calculate absolute errors per observation and category
    absolute_errors = np.abs(predicted_probs - observed_probs)
    # Average over all observations and categories
    return np.mean(absolute_errors)

def calculate_rmse(predicted_probs: np.ndarray, observed_counts: np.ndarray, total_n: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error between predicted probabilities and observed proportions.

    Args:
        predicted_probs: Array of predicted probabilities (n_observations, n_categories).
                         Should be the mean or median of the posterior predictive distribution.
        observed_counts: Array of observed counts (n_observations, n_categories).
        total_n: Array of total counts for each observation (n_observations,).

    Returns:
        The Root Mean Squared Error. Returns np.nan if shapes mismatch or calculation fails.
    """
    if predicted_probs.shape != observed_counts.shape or predicted_probs.shape[0] != total_n.shape[0]:
        print(f"Error: Shape mismatch in calculate_rmse. pred: {predicted_probs.shape}, obs: {observed_counts.shape}, N: {total_n.shape}")
        return np.nan # Return NaN on shape mismatch

    # Calculate observed probabilities, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ensure total_n is float for division and add new axis for broadcasting
        observed_probs = observed_counts / total_n.astype(float)[:, np.newaxis]
    observed_probs = np.nan_to_num(observed_probs) # Replace NaNs (from 0/0) with 0

    # Calculate squared errors per observation and category
    squared_errors = (predicted_probs - observed_probs)**2
    # Average over all observations and categories, then take sqrt
    return np.sqrt(np.mean(squared_errors))

def calculate_log_score(predicted_probs: np.ndarray, observed_counts: np.ndarray) -> np.ndarray:
    """
    Calculates the Log Score (negative log-likelihood) for each multinomial outcome.

    Uses the mean predicted probabilities for the calculation. Requires integer counts.

    Args:
        predicted_probs: Array of predicted probabilities (n_observations, n_categories).
                         This should represent the mean probability vector for each observation.
        observed_counts: Array of observed counts (n_observations, n_categories). MUST be integer type.

    Returns:
        An array of Log Scores (negative log-likelihoods) for each observation (lower is better, but
        this function returns the negative, so higher is worse for direct comparison).
        Contains np.inf if any predicted probability is zero where observed count is non-zero.
    """
    if predicted_probs.shape != observed_counts.shape:
        raise ValueError("Predicted probability and observed count shapes must match.")
    if not np.issubdtype(observed_counts.dtype, np.integer):
         print(f"Warning: observed_counts dtype is {observed_counts.dtype}, attempting cast to int for log_score.")
         observed_counts = observed_counts.astype(int) # Ensure integer type for multinomial
         
    if not np.allclose(np.sum(predicted_probs, axis=1), 1.0, atol=1e-6):
         print("Warning: Predicted probabilities do not sum to 1 for all observations.")
         # Normalize just in case, though this indicates an upstream issue
         predicted_probs = predicted_probs / predicted_probs.sum(axis=1, keepdims=True)

    total_log_likelihood = 0
    n_observations = observed_counts.shape[0]
    log_scores = np.zeros(n_observations)

    for i in range(n_observations):
        # Get total N for this observation directly from the counts passed
        n_total = np.sum(observed_counts[i]) # Must be int
        if n_total == 0:
             log_scores[i] = np.nan # Assign NaN for observations with zero total count
             continue

        # Get probabilities for this observation
        probs_i = predicted_probs[i]

        # Avoid log(0) issues - check if any prob is zero where counts are non-zero
        zero_prob_non_zero_count = np.any((probs_i <= 1e-9) & (observed_counts[i] > 0)) # Use small threshold
        
        if zero_prob_non_zero_count:
            # Cannot calculate log-likelihood, return infinity or handle as error
            # For scoring, a very large penalty is appropriate.
             print(f"Warning: Near-zero probability predicted for non-zero observed count at index {i}.")
             log_scores[i] = np.inf # Indicate impossible prediction
             continue # Move to next observation

        # Clip probabilities slightly away from 0 and 1 for stability if needed,
        # though normalization above helps. Using scipy's logpmf handles most cases.
        # probs_i = np.clip(probs_i, 1e-9, 1.0 - 1e-9)
        # probs_i /= probs_i.sum() # Re-normalize after clipping

        try:
            # Use scipy's multinomial logpmf for numerical stability
            # It expects integer counts and total N
            log_likelihood_i = multinomial.logpmf(x=observed_counts[i], n=n_total, p=probs_i)
            # Store the negative log-likelihood (log score)
            log_scores[i] = -log_likelihood_i
        except ValueError as e:
             print(f"Error in multinomial.logpmf at index {i}: {e}. Counts: {observed_counts[i]}, N: {n_total}, Probs: {probs_i}")
             log_scores[i] = np.inf # Assign infinity on error


    # Return the array of individual log scores
    return log_scores

def calculate_rps(predicted_probs: np.ndarray, observed_counts: np.ndarray, total_n: np.ndarray) -> float:
    """
    Calculates the average Rank Probability Score (RPS).

    Compares the cumulative probability distributions.

    Args:
        predicted_probs: Array of predicted probabilities (n_observations, n_categories).
        observed_counts: Array of observed counts (n_observations, n_categories).
        total_n: Array of total counts for each observation (n_observations,).

    Returns:
        The average Rank Probability Score (lower is better). Returns np.nan if shapes mismatch.
    """
    if predicted_probs.shape != observed_counts.shape or predicted_probs.shape[0] != total_n.shape[0]:
        print(f"Error: Shape mismatch in calculate_rps. pred: {predicted_probs.shape}, obs: {observed_counts.shape}, N: {total_n.shape}")
        return np.nan

    # Calculate observed probabilities, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ensure total_n is float for division and add new axis for broadcasting
        observed_probs = observed_counts / total_n.astype(float)[:, np.newaxis]
    observed_probs = np.nan_to_num(observed_probs) # Replace NaNs (from 0/0) with 0

    n_observations, n_categories = predicted_probs.shape

    # Calculate cumulative probabilities
    predicted_cdfs = np.cumsum(predicted_probs, axis=1)
    observed_cdfs = np.cumsum(observed_probs, axis=1)

    # Ensure the last cumulative probability is 1 (handle potential floating point issues)
    # This might not be strictly necessary after nan_to_num if sums were correct before division by zero
    predicted_cdfs[:, -1] = np.round(predicted_cdfs[:, -1], 6) # Round to avoid small fp errors
    observed_cdfs[:, -1] = np.round(observed_cdfs[:, -1], 6)

    # Calculate squared differences between CDFs
    squared_diff_cdfs = (predicted_cdfs - observed_cdfs)**2

    # Sum squared differences for each observation
    rps_per_observation = np.sum(squared_diff_cdfs, axis=1)

    # Average RPS over all observations
    return np.mean(rps_per_observation)

def calculate_calibration_data(
    posterior_probs: xr.DataArray, # Pass full posterior distribution (chain, draw, obs, party)
    observed_counts: np.ndarray,
    total_n: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Calculates data needed for a reliability diagram (marginal calibration) using
    the full posterior distribution of predicted probabilities.

    Bins all individual party probability posterior samples and compares the average
    predicted probability in each bin to the average observed proportion.

    Args:
        posterior_probs: xarray DataArray of posterior samples for probabilities
                         (chain, draw, n_observations, n_categories).
        observed_counts: Array of observed counts (n_observations, n_categories).
        total_n: Array of total counts for each observation (n_observations,).
        n_bins: Number of bins to divide the [0, 1] probability range into.

    Returns:
        A dictionary containing:
        - 'mean_predicted_prob': Average predicted probability from posterior samples in each bin.
        - 'mean_observed_prob': Average observed proportion of corresponding observations in each bin.
        - 'bin_counts': Number of posterior samples falling into each bin.
        - 'bin_edges': The edges of the bins used.
        Returns dict with NaNs if shapes mismatch or calculation fails.
    """
    # --- REMOVE DEBUG PRINT ---
    # print(f"DEBUG CALIBRATION: Received posterior_probs dimensions: {posterior_probs.dims}") 
    # --- END REMOVE DEBUG PRINT ---

    # 1. Input Validation
    if not isinstance(posterior_probs, xr.DataArray):
        print(f"Error: posterior_probs must be an xarray DataArray.")
        return {
            "mean_predicted_prob": np.full(n_bins, np.nan),
            "mean_observed_prob": np.full(n_bins, np.nan),
            "bin_counts": np.zeros(n_bins, dtype=int),
            "bin_edges": np.linspace(0, 1, n_bins + 1)[:-1]
        }

    if posterior_probs.dims != ('chain', 'draw', 'observations', 'parties_complete') or \
       posterior_probs.shape[2:] != observed_counts.shape or \
       posterior_probs.shape[2] != total_n.shape[0]:
        print(f"Error: Shape mismatch in calculate_calibration_data. posterior: {posterior_probs.shape}, obs_counts: {observed_counts.shape}, N: {total_n.shape}")
        return {
            "mean_predicted_prob": np.full(n_bins, np.nan),
            "mean_observed_prob": np.full(n_bins, np.nan),
            "bin_counts": np.zeros(n_bins, dtype=int),
            "bin_edges": np.linspace(0, 1, n_bins + 1)[:-1]
        }

    # Calculate observed probabilities, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ensure total_n is float for division and add new axis for broadcasting
        observed_probs = observed_counts / total_n.astype(float)[:, np.newaxis]
    observed_probs = np.nan_to_num(observed_probs) # Replace NaNs (from 0/0) with 0

    # Flatten the posterior samples and observed probs across observations and parties
    # Shape becomes (n_samples, n_obs * n_parties) -> transpose -> (n_obs * n_parties, n_samples)
    flat_posterior_probs = posterior_probs.stack(datapoint=['observations', 'parties_complete']).transpose('datapoint', 'chain', 'draw').values.reshape(-1, posterior_probs.shape[0]*posterior_probs.shape[1])
    # Shape becomes (n_obs * n_parties,)
    flat_obs_probs = observed_probs.flatten()

    # Define bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Add small epsilon to the upper bound to include 1.0 in the last bin
    bin_edges[-1] += 1e-8

    mean_predicted_prob = np.zeros(n_bins)
    mean_observed_prob = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int) # Counts the number of *posterior samples* in each bin

    # Iterate through each posterior sample probability prediction
    for i in range(flat_posterior_probs.shape[0]): # Iterate through data points (obs*party)
        # Digitize all posterior samples for this data point
        sample_probs = flat_posterior_probs[i, :] # Shape (n_samples,)
        bin_indices_for_samples = np.digitize(sample_probs, bin_edges) - 1
        bin_indices_for_samples = np.clip(bin_indices_for_samples, 0, n_bins - 1)

        # Get the single observed probability for this data point
        obs_prob_for_datapoint = flat_obs_probs[i]

        # Add contributions to bins based on where samples fall
        for bin_idx in range(n_bins):
             samples_in_bin_mask = (bin_indices_for_samples == bin_idx)
             n_samples_in_bin = samples_in_bin_mask.sum()

             if n_samples_in_bin > 0:
                 bin_counts[bin_idx] += n_samples_in_bin
                 # Add the mean of the *samples that fell into this bin* to the running sum
                 mean_predicted_prob[bin_idx] += np.sum(sample_probs[samples_in_bin_mask])
                 # Add the corresponding *observed probability* multiplied by the number of samples in the bin
                 mean_observed_prob[bin_idx] += obs_prob_for_datapoint * n_samples_in_bin


    # Calculate the final averages for each bin
    # Avoid division by zero for empty bins
    valid_bins = bin_counts > 0
    mean_predicted_prob[valid_bins] /= bin_counts[valid_bins]
    mean_observed_prob[valid_bins] /= bin_counts[valid_bins]

    # Handle empty bins - assign NaN
    mean_predicted_prob[~valid_bins] = np.nan
    mean_observed_prob[~valid_bins] = np.nan


    return {
        "mean_predicted_prob": mean_predicted_prob,
        "mean_observed_prob": mean_observed_prob,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges[:-1] # Return lower edges
    }

# TODO: Add functions for plotting calibration (e.g., reliability diagrams) 