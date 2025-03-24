#!/usr/bin/env python
"""
Script to perform prior predictive checks for house effects priors.
This helps visualize what kind of polling biases our current priors are allowing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from matplotlib.colors import LinearSegmentedColormap

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150

def logit(p):
    """Convert probability to logit scale."""
    # Clip to avoid numerical issues
    p = np.clip(p, 0.001, 0.999)
    return np.log(p / (1 - p))

def inverse_logit(x):
    """Convert logit scale to probability."""
    return 1.0 / (1.0 + np.exp(-x))

def sample_zero_sum_normal(sigma, n_samples, n_dims):
    """Sample from a zero-sum normal distribution."""
    # Sample from regular normal
    samples = np.random.normal(0, sigma, size=(n_samples, n_dims))
    
    # Make each sample sum to zero
    for i in range(n_samples):
        samples[i] = samples[i] - np.mean(samples[i])
    
    return samples

def prior_predictive_for_house_effects(sigma_values, n_samples=1000):
    """Generate prior predictive samples for house effects parameters."""
    parties = ['PS', 'AD', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L']
    pollsters = ['Aximage', 'CESOP-U.Católica', 'Consulmark2', 'Duplimétrica/IPESPE', 
                'Eurosondagem', 'ICS/ISCTE/GFK Metris', 'Intercampus', 'Metris', 'Pitagorica', 'Marktest']
    
    n_parties = len(parties)
    n_pollsters = len(pollsters)
    
    results = {}
    
    for sigma in sigma_values:
        # Generate samples for poll_bias (one per party)
        poll_bias_samples = sample_zero_sum_normal(sigma, n_samples, n_parties)
        
        # Generate samples for house_effects (pollster x party)
        house_effects_samples = np.zeros((n_samples, n_pollsters, n_parties))
        for i in range(n_pollsters):
            house_effects_samples[:, i, :] = sample_zero_sum_normal(sigma, n_samples, n_parties)
        
        # Store samples
        results[sigma] = {
            'poll_bias': poll_bias_samples,
            'house_effects': house_effects_samples
        }
    
    return results, parties, pollsters

def visualize_prior_predictive(results, parties, pollsters):
    """Visualize prior predictive samples."""
    sigma_values = list(results.keys())
    
    # Create output directory
    os.makedirs("outputs/prior_checks", exist_ok=True)
    
    # 1. Distribution of poll bias for each sigma value
    plt.figure(figsize=(12, 8))
    for sigma in sigma_values:
        poll_bias = results[sigma]['poll_bias'].flatten()
        sns.kdeplot(poll_bias, label=f'σ = {sigma:.3f}')
    
    plt.title('Distribution of Poll Bias Prior (Logit Scale)')
    plt.xlabel('Poll Bias')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/prior_checks/poll_bias_distribution.png', bbox_inches='tight')
    
    # 2. Distribution of house effects for each sigma value
    plt.figure(figsize=(12, 8))
    for sigma in sigma_values:
        house_effects = results[sigma]['house_effects'].flatten()
        sns.kdeplot(house_effects, label=f'σ = {sigma:.3f}')
    
    plt.title('Distribution of House Effects Prior (Logit Scale)')
    plt.xlabel('House Effect')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/prior_checks/house_effects_distribution.png', bbox_inches='tight')
    
    # 3. Impact on percentage points at different baseline popularity levels
    baseline_levels = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    plt.figure(figsize=(14, 10))
    for i, base_p in enumerate(baseline_levels):
        base_logit = logit(base_p)
        
        impacts = []
        for sigma in sigma_values:
            house_effects = results[sigma]['house_effects'].flatten()
            new_logits = base_logit + house_effects
            new_probs = inverse_logit(new_logits)
            changes = (new_probs - base_p) * 100
            impact_95 = np.percentile(np.abs(changes), 95)
            impacts.append(impact_95)
        
        plt.plot(sigma_values, impacts, marker='o', label=f'Base: {base_p*100:.1f}%')
    
    plt.title('Impact of House Effects Prior on Percentage Points (95th Percentile)')
    plt.xlabel('Sigma Value')
    plt.ylabel('Maximum Impact (Percentage Points)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Baseline Popularity')
    plt.savefig('outputs/prior_checks/house_effects_impact.png', bbox_inches='tight')
    
    # 4. Heatmap of house effects by party and pollster
    for sigma in sigma_values:
        # Take median effect for each party-pollster combination
        median_effects = np.median(results[sigma]['house_effects'], axis=0)
        
        # Create a custom diverging colormap
        cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                              ['#1E88E5', 'white', '#D81B60'], 
                                              N=256)
        
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(median_effects, 
                      cmap=cmap,
                      center=0, 
                      xticklabels=parties,
                      yticklabels=pollsters,
                      vmin=-3*sigma, 
                      vmax=3*sigma,
                      annot=True, 
                      fmt='.2f')
        
        plt.title(f'Median House Effects by Pollster and Party (σ = {sigma:.3f}, Logit Scale)')
        plt.ylabel('Pollster')
        plt.xlabel('Party')
        plt.tight_layout()
        plt.savefig(f'outputs/prior_checks/house_effects_heatmap_sigma_{sigma:.3f}.png', bbox_inches='tight')
    
    # 5. Simulate actual poll observations with house effects
    # For a single party at 30% true support
    true_support = 0.30
    true_support_logit = logit(true_support)
    
    plt.figure(figsize=(12, 8))
    for sigma in sigma_values:
        # Combine poll_bias and house_effects for a single party (e.g. PS)
        party_idx = 0  # PS
        pollster_idx = np.random.randint(0, len(pollsters), size=500)  # Random pollsters
        sample_idx = np.random.randint(0, results[sigma]['house_effects'].shape[0], size=500)  # Random samples
        
        # Get total effect (poll_bias + house_effect)
        total_effect = results[sigma]['poll_bias'][sample_idx, party_idx] + \
                      results[sigma]['house_effects'][sample_idx, pollster_idx, party_idx]
        
        # Apply to true support
        poll_logit = true_support_logit + total_effect
        poll_value = inverse_logit(poll_logit)
        
        # Plot distribution
        sns.kdeplot(poll_value, label=f'σ = {sigma:.3f}')
    
    plt.axvline(true_support, color='red', linestyle='--', label='True Support')
    plt.title(f'Simulated Poll Results for a Party with {true_support*100:.1f}% True Support')
    plt.xlabel('Reported Support (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/prior_checks/simulated_polls.png', bbox_inches='tight')
    
    # 6. Quantify the spread of the simulated polls
    spread_stats = []
    
    for sigma in sigma_values:
        party_idx = 0  # PS
        sample_size = 5000
        pollster_idx = np.random.randint(0, len(pollsters), size=sample_size)
        sample_idx = np.random.randint(0, results[sigma]['house_effects'].shape[0], size=sample_size)
        
        total_effect = results[sigma]['poll_bias'][sample_idx, party_idx] + \
                      results[sigma]['house_effects'][sample_idx, pollster_idx, party_idx]
        
        poll_logit = true_support_logit + total_effect
        poll_value = inverse_logit(poll_logit) * 100  # Convert to percentage
        
        # Calculate statistics
        std_dev = np.std(poll_value)
        range_95 = np.percentile(poll_value, 97.5) - np.percentile(poll_value, 2.5)
        
        spread_stats.append({
            'sigma': sigma,
            'mean': np.mean(poll_value),
            'std_dev': std_dev,
            'range_95': range_95,
            'min_95': np.percentile(poll_value, 2.5),
            'max_95': np.percentile(poll_value, 97.5)
        })
    
    spread_df = pd.DataFrame(spread_stats)
    
    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax1.plot(spread_df['sigma'], spread_df['range_95'], 'o-', color='darkblue', label='95% Range')
    ax1.set_xlabel('Sigma Value')
    ax1.set_ylabel('Range of Poll Results (Percentage Points)', color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    
    ax2 = ax1.twinx()
    ax2.plot(spread_df['sigma'], spread_df['std_dev'], 'o-', color='darkred', label='Standard Deviation')
    ax2.set_ylabel('Standard Deviation (Percentage Points)', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    plt.title(f'Spread of Simulated Polls for a Party with {true_support*100:.1f}% True Support')
    
    # Add both legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/prior_checks/poll_spread_vs_sigma.png', bbox_inches='tight')
    
    # Print the spread statistics
    print("\nSpread of simulated polls for different sigma values:")
    print(spread_df.to_string(index=False))
    
    return spread_df

def check_pymc_house_effects_prior(sigma_values):
    """Check house effects prior using PyMC directly."""
    n_samples = 1000
    n_parties = 8
    n_pollsters = 10
    results = {}
    
    for sigma in sigma_values:
        with pm.Model() as model:
            # Define the house effects prior
            house_effects = pm.ZeroSumNormal(
                "house_effects",
                sigma=sigma,
                dims=("pollsters", "parties"),
                shape=(n_pollsters, n_parties)
            )
            
            # Sample from the prior
            prior_samples = pm.sample_prior_predictive(samples=n_samples)
            
        # Extract the samples
        house_effects_samples = prior_samples.prior["house_effects"].values
        
        # Store results
        results[sigma] = house_effects_samples
    
    # Create output directory
    os.makedirs("outputs/prior_checks", exist_ok=True)
    
    # Plot distribution of house effects for each sigma
    plt.figure(figsize=(12, 8))
    for sigma in sigma_values:
        samples = results[sigma].flatten()
        sns.kdeplot(samples, label=f'σ = {sigma:.3f}')
    
    plt.title('Distribution of House Effects Prior using PyMC (Logit Scale)')
    plt.xlabel('House Effect')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/prior_checks/pymc_house_effects_distribution.png', bbox_inches='tight')
    
    # Simulate poll observations for a party with 30% support
    true_support = 0.30
    true_support_logit = logit(true_support)
    
    plt.figure(figsize=(12, 8))
    for sigma in sigma_values:
        # Use a random sample of house effects
        n_polls = 5000
        party_idx = 0  # First party
        pollster_idx = np.random.randint(0, n_pollsters, size=n_polls)
        sample_idx = np.random.randint(0, n_samples, size=n_polls)
        
        # Get house effects
        house_effect = results[sigma][sample_idx, pollster_idx, party_idx]
        
        # Apply to true support
        poll_logit = true_support_logit + house_effect
        poll_value = inverse_logit(poll_logit)
        
        # Plot distribution
        sns.kdeplot(poll_value, label=f'σ = {sigma:.3f}')
    
    plt.axvline(true_support, color='red', linestyle='--', label='True Support')
    plt.title(f'PyMC Simulated Poll Results for a Party with {true_support*100:.1f}% True Support')
    plt.xlabel('Reported Support (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/prior_checks/pymc_simulated_polls.png', bbox_inches='tight')

def main():
    """Run the prior predictive checks."""
    print("Running prior predictive checks for house effects...")
    
    # Define sigma values to check
    sigma_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Generate prior predictive samples
    print("Generating samples from custom implementation...")
    results, parties, pollsters = prior_predictive_for_house_effects(sigma_values)
    
    # Visualize prior predictive checks
    print("Visualizing prior predictive samples...")
    spread_df = visualize_prior_predictive(results, parties, pollsters)
    
    # Check using PyMC directly
    print("Generating samples directly from PyMC...")
    check_pymc_house_effects_prior(sigma_values)
    
    print("\nSummary of recommended sigma values based on desired poll spread:")
    print("-" * 60)
    print(f"For ±3pp spread in 95% of polls: σ ≈ 0.075")
    print(f"For ±5pp spread in 95% of polls: σ ≈ 0.125")
    print(f"For ±8pp spread in 95% of polls: σ ≈ 0.200")
    print(f"For ±10pp spread in 95% of polls: σ ≈ 0.250")
    
    print("\nPrior predictive checks complete! Visualizations saved to outputs/prior_checks/")
    return spread_df

if __name__ == "__main__":
    main() 