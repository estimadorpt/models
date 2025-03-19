"""
This module contains functions for evaluating the retrodictive accuracy of election forecasts.
"""

import numpy as np
import pandas as pd


def evaluate_retrodictive_accuracy(elections_model, election_date):
    """
    Evaluate the accuracy of the model's predictions for a specific election date.
    
    Parameters:
    -----------
    elections_model : ElectionsFacade
        The elections model object containing the predictions
    election_date : str
        The election date to evaluate, in YYYY-MM-DD format
        
    Returns:
    --------
    dict
        A dictionary containing accuracy metrics
    """
    # Get the actual results for this election
    actual_results = get_actual_results(elections_model, election_date)
    if actual_results is None:
        print(f"Warning: No actual results found for election {election_date}")
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
    
    # Get the predictions for this election
    try:
        predictions = get_election_day_predictions(elections_model)
        if predictions is None:
            print(f"Warning: No predictions available for election {election_date}")
            return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
    
    # Calculate error metrics
    metrics = calculate_error_metrics(predictions, actual_results)
    
    # Print a comparison table
    print("\nRETRODICTIVE ACCURACY EVALUATION")
    print("="*50)
    print(f"Election date: {election_date}")
    print("="*50)
    print(f"{'Party':15s} {'Predicted':10s} {'Actual':10s} {'Error':10s}")
    print("-"*50)
    
    for party in actual_results.index:
        pred = predictions.get(party, 0) * 100
        actual = actual_results.loc[party] * 100
        error = pred - actual
        print(f"{party:15s} {pred:10.2f}% {actual:10.2f}% {error:+10.2f}%")
    
    print("-"*50)
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAPE: {metrics['mape']:.3f}%")
    print("="*50)
    
    return metrics


def get_actual_results(elections_model, election_date):
    """
    Get the actual election results for a specific election date.
    
    Parameters:
    -----------
    elections_model : ElectionsFacade
        The elections model object containing the results data
    election_date : str
        The election date to get results for, in YYYY-MM-DD format
        
    Returns:
    --------
    pd.Series
        A series containing the actual results for each party
    """
    # Check if results are available for this election
    try:
        # Get results from the dataset
        results = elections_model.dataset.results_mult
        
        # Filter to the specific election date
        election_results = results[results['election_date'] == pd.to_datetime(election_date)]
        
        if len(election_results) == 0:
            return None
        
        # Extract party vote shares
        party_columns = elections_model.dataset.political_families
        
        # Convert from counts to percentages
        result_row = election_results.iloc[0]
        total_votes = result_row[party_columns].sum()
        party_shares = result_row[party_columns] / total_votes
        
        return party_shares
    except Exception as e:
        print(f"Error getting actual results: {e}")
        return None


def get_election_day_predictions(elections_model):
    """
    Get the model's predictions for the election day.
    
    Parameters:
    -----------
    elections_model : ElectionsFacade
        The elections model object containing the predictions
        
    Returns:
    --------
    dict
        A dictionary mapping party names to predicted vote shares
    """
    if not hasattr(elections_model, 'prediction') or elections_model.prediction is None:
        return None
    
    # Extract the forecast data for the election date
    latent_pop = elections_model.prediction.get('latent_popularity')
    if latent_pop is None:
        return None
    
    # Get forecast for the last date (election day)
    election_day_forecast = latent_pop[:, :, -1, :]
    
    # Calculate mean forecast
    mean_forecast = np.mean(election_day_forecast, axis=(0, 1))
    
    # Get party names
    parties = elections_model.prediction_coords['parties_complete']
    
    # Create a dictionary mapping party names to predictions
    predictions = {}
    for i, party in enumerate(parties):
        predictions[party] = mean_forecast[i]
    
    return predictions


def calculate_error_metrics(predictions, actual_results):
    """
    Calculate error metrics between predictions and actual results.
    
    Parameters:
    -----------
    predictions : dict
        A dictionary mapping party names to predicted vote shares
    actual_results : pd.Series
        A series containing the actual results for each party
        
    Returns:
    --------
    dict
        A dictionary containing error metrics:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - mape: Mean Absolute Percentage Error
    """
    # Convert predictions dict to a series with the same index as actual_results
    pred_series = pd.Series(index=actual_results.index)
    for party in actual_results.index:
        pred_series[party] = predictions.get(party, 0)
    
    # Calculate errors
    absolute_errors = np.abs(pred_series - actual_results)
    squared_errors = (pred_series - actual_results) ** 2
    
    # Calculate percentage errors, avoiding division by zero
    percentage_errors = np.zeros_like(absolute_errors)
    for party in actual_results.index:
        if actual_results[party] > 0.001:  # threshold to avoid division by very small numbers
            percentage_errors[party] = absolute_errors[party] / actual_results[party] * 100
    
    # Calculate overall metrics
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    mape = np.mean(percentage_errors)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    } 