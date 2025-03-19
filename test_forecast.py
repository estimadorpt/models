import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project root to the path if needed
sys.path.append('.')

# Import your model and dataset classes
from src.models.election_model import ElectionModel
from src.data.dataset import ElectionDataset

def test_forecast():
    """
    Test the forecast_election method with a real dataset.
    """
    print("Initializing dataset...")
    # Use a historical election for testing
    test_election_date = "2024-03-10"
    dataset = ElectionDataset(
        election_date=test_election_date,
        baseline_timescales=[365],
        election_timescales=[60],
    )
    
    print("\nInitializing model...")
    # Initialize the model
    model = ElectionModel(dataset)
    
    # Define trace directory path
    trace_dir = "outputs/2025-03-18_174139/trace.zarr"
    
    # Load the trace using correct approach
    print(f"\nLoading trace from {trace_dir}")
    idata = az.from_zarr(trace_dir)
    print("Successfully loaded trace!")
    
    # Print trace summary
    print("\nTrace Summary:")
    print(f"Posterior variables: {list(idata.posterior.data_vars)}")
    print(f"Chains: {idata.posterior.dims['chain']}")
    print(f"Draws: {idata.posterior.dims['draw']}")
    
    print("\nBuilding the model...")
    # First build the model to ensure self.model exists
    model.model = model.build_model()
    
    print("\nTesting forecast_election method...")
    # Now run the forecast method with properly loaded trace
    raw_predictions, coords, dims = model.forecast_election(idata)
    
    # Verify the results
    print("\nVerifying forecast results:")
    print(f"- Number of forecast dates: {len(coords['observations'])}")
    print(f"- Parties in forecast: {coords['parties_complete']}")
    
    # Check the shapes of the predictions
    for var_name, var_data in raw_predictions.items():
        print(f"- {var_name} shape: {var_data.shape}")
    
    # Output detailed results to the terminal
    print("\n=== FORECAST RESULTS SUMMARY ===")
    
    # Calculate mean of latent popularity for each party across chains and draws
    popularity = raw_predictions['latent_popularity']
    mean_popularity = np.mean(popularity, axis=(0, 1))  # Average over chains and draws
    
    # Get 95% credible intervals
    lower_bound = np.percentile(popularity, 2.5, axis=(0, 1))
    upper_bound = np.percentile(popularity, 97.5, axis=(0, 1))
    
    # Output the final election forecast (last date)
    final_date_idx = -1  # Last date in the forecast
    election_date = coords['observations'][final_date_idx].strftime('%Y-%m-%d')
    
    print(f"\nFinal Election Forecast ({election_date}):")
    print("Party    | Mean    | 95% CI         | Rank")
    print("---------|---------|----------------|-----")
    
    # Sort parties by their mean prediction on election day
    final_preds = mean_popularity[final_date_idx]
    party_ranks = np.argsort(-final_preds)  # Descending order
    
    for rank, party_idx in enumerate(party_ranks):
        party = coords['parties_complete'][party_idx]
        mean_val = final_preds[party_idx]
        lb = lower_bound[final_date_idx, party_idx]
        ub = upper_bound[final_date_idx, party_idx]
        print(f"{party:<8} | {mean_val:.1%} | [{lb:.1%}, {ub:.1%}] | {rank+1}")
    
    # Show trend over time at key intervals
    print("\nTrend over forecast period:")
    
    # Select dates at approximately 1-month intervals
    total_days = len(coords['observations'])
    if total_days > 30:
        interval_days = 30
        selected_indices = range(0, total_days, interval_days)
        
        # Get the top 4 parties from the final prediction
        top_parties = [coords['parties_complete'][idx] for idx in party_ranks[:4]]
        
        # Print header with dates
        header = "Party    |"
        for idx in selected_indices:
            header += f" {coords['observations'][idx].strftime('%Y-%m-%d')} |"
        print(header)
        print("-" * len(header))
        
        # Print trends for top parties
        for party_name in top_parties:
            party_idx = coords['parties_complete'].index(party_name)
            row = f"{party_name:<8} |"
            
            for idx in selected_indices:
                value = mean_popularity[idx, party_idx]
                row += f" {value:.1%} |"
            
            print(row)
    
    # Plot some results for visual inspection
    try:
        print("\nPlotting forecast results...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot for each party
        dates = coords['observations']
        for i, party in enumerate(coords['parties_complete']):
            ax.plot(dates, mean_popularity[:, i], label=party)
        
        ax.set_title(f"Forecast for {test_election_date} election")
        ax.set_xlabel("Date")
        ax.set_ylabel("Latent popularity")
        ax.legend()
        
        # Save the plot
        plt.savefig(f"forecast_test_{test_election_date.replace('-', '')}.png")
        print(f"Plot saved as forecast_test_{test_election_date.replace('-', '')}.png")
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("Continuing without plot...")
    
    return raw_predictions, coords, dims

if __name__ == "__main__":
    test_forecast() 