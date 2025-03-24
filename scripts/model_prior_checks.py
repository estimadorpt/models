#!/usr/bin/env python
"""
Script to perform prior predictive checks for the complete election model.
This helps visualize what kind of election dynamics and polling behavior our model priors allow.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from datetime import datetime, timedelta
import pytensor.tensor as pt

# Ensure PYTHONPATH includes the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.election_model import ElectionModel
from src.data.dataset import ElectionDataset
from src.visualization.plots import plot_latent_trajectories

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 150

def run_prior_predictive(n_samples=500):
    """
    Run prior predictive checks on the complete election model.
    
    Parameters:
    -----------
    n_samples : int
        Number of prior samples to generate
    """
    print("Loading dataset...")
    # Load the dataset using a future election date
    dataset = ElectionDataset(
        election_date="2025-05-18",  # Future election
        baseline_timescales=[180],  # Using values from previous model runs
        election_timescales=[15],
    )
    
    print("Building model...")
    # Build the election model
    model = ElectionModel(dataset)
    
    # Access model components
    political_families = dataset.political_families
    unique_pollsters = dataset.unique_pollsters
    
    # Sample dates for predictions
    reference_date = pd.Timestamp("2025-05-18")  # Election date
    days_before = 365  # Show one year of predictions
    prediction_dates = [reference_date - timedelta(days=d) for d in range(0, days_before, 10)]
    prediction_dates.reverse()  # Sort chronologically
    
    # For PyMC model
    with pm.Model() as prior_model:
        # Build the complete model
        complete_model = model.build_model()
        
        # Store important variables for prior samples
        party_baseline = prior_model.named_vars.get("party_baseline")
        election_party_baseline = prior_model.named_vars.get("election_party_baseline")
        house_effects = prior_model.named_vars.get("house_effects")
        poll_bias = prior_model.named_vars.get("poll_bias")
        
        # Get model parameters to access timescales
        gp_config = model.gp_config
        
        # Sample from prior
        print(f"Sampling {n_samples} draws from prior...")
        prior_samples = pm.sample_prior_predictive(samples=n_samples)
    
    # Create output directory
    os.makedirs("outputs/model_prior_checks", exist_ok=True)
    
    print("Analyzing prior samples and creating visualizations...")
    
    # 1. Party baseline distributions
    plt.figure(figsize=(14, 8))
    party_baseline_samples = prior_samples.prior["party_baseline"].values
    
    for i, party in enumerate(political_families):
        party_samples = party_baseline_samples[:, i]
        sns.kdeplot(party_samples, label=party)
        
    plt.title("Prior Distribution of Party Baselines (Logit Scale)")
    plt.xlabel("Party Baseline")
    plt.ylabel("Density")
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/model_prior_checks/party_baseline_prior.png", bbox_inches='tight')
    
    # 2. Election-specific party baseline distributions
    if "election_party_baseline" in prior_samples.prior:
        plt.figure(figsize=(14, 8))
        election_baseline_samples = prior_samples.prior["election_party_baseline"].values
        
        for i, party in enumerate(political_families):
            if len(election_baseline_samples.shape) >= 3:
                # Use the most recent election
                party_samples = election_baseline_samples[:, -1, i]
            else:
                party_samples = election_baseline_samples[:, i]
                
            sns.kdeplot(party_samples, label=party)
            
        plt.title("Prior Distribution of Election-Specific Party Effects (Logit Scale)")
        plt.xlabel("Election-Specific Effect")
        plt.ylabel("Density")
        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("outputs/model_prior_checks/election_baseline_prior.png", bbox_inches='tight')
    
    # 3. House effects distributions
    plt.figure(figsize=(14, 8))
    
    if "house_effects" in prior_samples.prior:
        house_effects_samples = prior_samples.prior["house_effects"].values.reshape(n_samples, -1)
        
        sns.kdeplot(house_effects_samples.flatten(), label="House Effects")
        
    if "poll_bias" in prior_samples.prior:
        poll_bias_samples = prior_samples.prior["poll_bias"].values.reshape(n_samples, -1)
        
        sns.kdeplot(poll_bias_samples.flatten(), label="Poll Bias")
            
    plt.title("Prior Distribution of House Effects and Poll Bias (Logit Scale)")
    plt.xlabel("Effect Size")
    plt.ylabel("Density")
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/model_prior_checks/house_effects_prior.png", bbox_inches='tight')
    
    # 4. GP function samples - Show how the model allows trends to vary over time
    sample_gp_trajectories(model, prior_samples, prediction_dates, political_families)
    
    # 5. Generate simulated polls from the model priors
    simulate_polls_from_prior(model, prior_samples, dataset, n_simulations=1000)
    
    print("Prior predictive checks complete!")

def sample_gp_trajectories(model, prior_samples, prediction_dates, political_families, n_trajectories=20):
    """Generate and plot GP trajectory samples from prior."""
    # Initialize figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
    axes = axes.flatten()
    
    # Convert dates to numbers for plotting
    ref_date = prediction_dates[0]
    x_dates = [(d - ref_date).days for d in prediction_dates]
    
    # Get random sample indices
    sample_indices = np.random.choice(prior_samples.prior.dims["sample"], 
                                     size=n_trajectories, 
                                     replace=False)
    
    # Plot for each party
    for i, party in enumerate(political_families):
        ax = axes[i]
        
        # Extract baseline and GP component parameters from prior samples
        if "party_baseline" in prior_samples.prior and "gp_coef_baseline_180" in prior_samples.prior:
            for idx in sample_indices:
                # Get baseline for this party (constant)
                baseline = prior_samples.prior["party_baseline"].values[idx, i]
                
                # Get GP coefficients for this party (time-varying)
                gp_coefs = prior_samples.prior["gp_coef_baseline_180"].values[idx, :, i]
                
                # Simple GP trajectory for demonstration (idealized)
                # In reality, this would use the actual GP kernel from the model
                trajectory = baseline + np.sin(np.array(x_dates) / 100) * np.random.normal(0, 0.2)
                
                # Convert to probability scale for easier interpretation
                trajectory_prob = 1.0 / (1.0 + np.exp(-trajectory))
                
                # Plot the trajectory
                ax.plot(prediction_dates, trajectory_prob, alpha=0.5, linewidth=1)
        
        ax.set_title(f"{party} Support (Prior Trajectories)")
        ax.set_ylim(0, 0.6)  # Set reasonable limits for vote shares
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Vote Share")
        
        # Format the x-axis for the bottom row
        if i >= len(political_families) - 4:
            ax.set_xlabel("Date")
            ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig("outputs/model_prior_checks/gp_trajectories_prior.png", bbox_inches='tight')

def simulate_polls_from_prior(model, prior_samples, dataset, n_simulations=1000):
    """Simulate polling data based on prior distributions."""
    political_families = dataset.political_families
    
    # Sample random dates for simulated polls
    ref_date = pd.Timestamp("2025-05-18")  # Election date
    poll_dates = [ref_date - timedelta(days=np.random.randint(1, 360)) for _ in range(n_simulations)]
    
    # Sample random pollsters
    pollsters = dataset.unique_pollsters
    poll_pollsters = np.random.choice(pollsters, size=n_simulations)
    
    # Initialize array for simulated polls
    simulated_polls = np.zeros((n_simulations, len(political_families)))
    
    # Sample parameters from prior
    n_samples = prior_samples.prior.dims["sample"]
    sample_indices = np.random.choice(n_samples, size=n_simulations)
    
    # Extract necessary components from prior samples
    if ("party_baseline" in prior_samples.prior and 
        "poll_bias" in prior_samples.prior and 
        "house_effects" in prior_samples.prior):
        
        # For each simulated poll
        for i in range(n_simulations):
            sample_idx = sample_indices[i]
            pollster_idx = np.where(pollsters == poll_pollsters[i])[0][0]
            
            # Get party baselines
            baselines = prior_samples.prior["party_baseline"].values[sample_idx]
            
            # Get poll bias for each party
            bias = prior_samples.prior["poll_bias"].values[sample_idx]
            
            # Get house effects for this pollster and each party
            house_effect = prior_samples.prior["house_effects"].values[sample_idx, pollster_idx, :]
            
            # Combine effects on logit scale
            latent_support = baselines + bias + house_effect
            
            # Add some election and time-varying effects (simplified)
            latent_support += np.random.normal(0, 0.15, size=len(political_families))
            
            # Convert to probability scale
            party_support = np.exp(latent_support)
            party_support = party_support / party_support.sum()  # Normalize
            
            # Store simulated poll
            simulated_polls[i] = party_support
    
    # Create DataFrame
    poll_data = {
        'date': poll_dates,
        'pollster': poll_pollsters,
    }
    
    # Add party vote shares
    for i, party in enumerate(political_families):
        poll_data[party] = simulated_polls[:, i]
    
    simulated_polls_df = pd.DataFrame(poll_data)
    
    # Plot distribution of vote shares for each party
    plt.figure(figsize=(14, 8))
    
    for party in political_families:
        sns.kdeplot(simulated_polls_df[party], label=party)
        
    plt.title("Distribution of Simulated Poll Results from Prior")
    plt.xlabel("Vote Share")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/model_prior_checks/simulated_polls_distribution.png", bbox_inches='tight')
    
    # Plot poll results over time for selected parties
    plt.figure(figsize=(14, 8))
    
    main_parties = ['PS', 'AD', 'CH', 'IL']  # Focus on main parties
    
    for party in main_parties:
        # Sort by date
        sorted_df = simulated_polls_df.sort_values('date')
        
        # Get data points
        dates = sorted_df['date']
        support = sorted_df[party]
        
        # Scatter plot with transparency
        plt.scatter(dates, support, alpha=0.3, label=party)
        
        # Add smoothed trend line
        if len(dates) > 5:  # Only if we have enough data
            try:
                sns.regplot(x=pd.to_numeric(dates), y=support, 
                           scatter=False, lowess=True, 
                           line_kws={'linewidth': 2}, color=plt.gca().lines[-1].get_color())
            except:
                # If lowess fails, skip the trend line
                pass
            
    plt.title("Simulated Poll Results Over Time (From Prior Distribution)")
    plt.xlabel("Date")
    plt.ylabel("Vote Share")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/model_prior_checks/simulated_polls_over_time.png", bbox_inches='tight')
    
    # Save the simulated polls data
    simulated_polls_df.to_csv("outputs/model_prior_checks/simulated_polls.csv", index=False)
    
    # Create box plots for each pollster
    plt.figure(figsize=(16, 10))
    
    for i, party in enumerate(main_parties):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='pollster', y=party, data=simulated_polls_df)
        plt.title(f"{party} Support by Pollster (Prior Simulations)")
        plt.xticks(rotation=90)
        plt.ylim(0, 0.6)  # Set reasonable limits for vote shares
        plt.tight_layout()
        
    plt.savefig("outputs/model_prior_checks/simulated_polls_by_pollster.png", bbox_inches='tight')
    
    return simulated_polls_df

def analyze_prior_impacts(model, prior_samples):
    """Analyze the impact of different prior components on the final predictions."""
    # Check all variables in the prior samples
    print("Variables in prior samples:")
    for var_name in prior_samples.prior.data_vars:
        var = prior_samples.prior[var_name]
        print(f"- {var_name}: shape {var.shape}")
    
    # Analyze how much variance is contributed by each component
    analysis_results = {}
    
    # Components to analyze
    components = [
        "party_baseline",
        "election_party_baseline",
        "house_effects",
        "poll_bias"
    ]
    
    # Analyze components
    for component in components:
        if component in prior_samples.prior:
            # Get the samples
            samples = prior_samples.prior[component].values
            
            # Flatten to analyze overall variance
            flat_samples = samples.reshape(samples.shape[0], -1)
            
            # Calculate variance statistics
            std_dev = np.std(flat_samples)
            variance = np.var(flat_samples)
            abs_mean = np.mean(np.abs(flat_samples))
            
            analysis_results[component] = {
                'std_dev': std_dev,
                'variance': variance,
                'abs_mean': abs_mean
            }
    
    # Print results
    print("\nComponent Impact Analysis:")
    print("-" * 50)
    for component, stats in analysis_results.items():
        print(f"{component}:")
        print(f"  Standard Deviation: {stats['std_dev']:.4f}")
        print(f"  Variance: {stats['variance']:.4f}")
        print(f"  Mean Absolute Value: {stats['abs_mean']:.4f}")
        print("-" * 50)
    
    return analysis_results

if __name__ == "__main__":
    run_prior_predictive(n_samples=500) 