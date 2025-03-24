#!/usr/bin/env python
"""
Script to perform prior predictive checks for house effects component.
This helps visualize what kind of polling biases our house effects priors allow.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 150

def logit(p):
    """Convert probability to logit scale."""
    p = np.clip(p, 0.001, 0.999)
    return np.log(p / (1 - p))

def inverse_logit(x):
    """Convert logit scale to probability."""
    return 1.0 / (1.0 + np.exp(-x))

def run_house_effects_check(sigma_values=[0.075, 0.125, 0.25]):
    """
    Perform prior predictive checks for house effects with different sigma values.
    
    Parameters:
    -----------
    sigma_values : list
        List of sigma values to test
    """
    # Create output directory
    os.makedirs("outputs/house_effects_check", exist_ok=True)
    
    # Define political parties and pollsters in Portugal
    parties = ['PS', 'AD', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L']
    pollsters = ['Aximage', 'CESOP', 'Consulmark2', 'Intercampus', 
                'Eurosondagem', 'Metris', 'Pitagorica']
    
    # Number of samples to draw
    n_samples = 1000
    
    # Store results for each sigma
    results = {}
    
    for sigma in sigma_values:
        print(f"Sampling with sigma = {sigma}...")
        
        # Use PyMC to generate samples from the house effects prior
        with pm.Model() as model:
            # Add dimensions to the model
            # Print available dimensions before and after adding our dimensions
            print("Model dimensions before adding our dims:", model.dim_lengths)
            
            # Define coordinates for our dimensions
            model.add_coord("pollster", pollsters, mutable=True)
            model.add_coord("party", parties, mutable=True)
            
            print("Model dimensions after adding our dims:", model.dim_lengths)
            
            # Define house effects
            house_effects = pm.ZeroSumNormal(
                "house_effects",
                sigma=sigma,
                dims=("pollster", "party"),
                shape=(len(pollsters), len(parties))
            )
            
            # Sample from prior
            prior = pm.sample_prior_predictive(samples=n_samples)
            
            # Extract house effects samples
            house_effects_samples = prior.prior["house_effects"].values
            
        results[sigma] = house_effects_samples
        
    # Now visualize the results
    
    # 1. Distribution of house effects for different sigma values
    plt.figure(figsize=(14, 8))
    for sigma in sigma_values:
        samples = results[sigma].flatten()
        sns.kdeplot(samples, label=f'σ = {sigma:.3f}')
    
    plt.title('Distribution of House Effects Prior (Logit Scale)')
    plt.xlabel('House Effect')
    plt.ylabel('Density')
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/house_effects_check/house_effects_distribution.png', bbox_inches='tight')
    
    # 2. Impact on percentage points at different baseline support levels
    base_supports = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, base_p in enumerate(base_supports):
        ax = axes[i]
        base_logit = logit(base_p)
        
        for sigma in sigma_values:
            # Flatten samples
            samples = results[sigma].flatten()
            
            # Apply effects to baseline
            new_logits = base_logit + samples
            new_probs = inverse_logit(new_logits)
            
            # Plot distribution
            sns.kdeplot(new_probs, ax=ax, label=f'σ = {sigma:.3f}')
        
        ax.axvline(base_p, color='red', linestyle='--', label='True support' if i == 0 else None)
        ax.set_title(f'Base Support: {base_p*100:.1f}%')
        ax.set_xlabel('Reported Support')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/house_effects_check/support_impact.png', bbox_inches='tight')
    
    # 3. Calculate the range of poll values for a 30% true support
    base_p = 0.30
    base_logit = logit(base_p)
    impact_stats = []
    
    for sigma in sigma_values:
        samples = results[sigma].reshape(n_samples, -1)
        
        # Apply effects to baseline for all samples
        poll_values = []
        for i in range(min(1000, samples.shape[0])):
            effects = samples[i]
            
            # Get multiple random samples from the same house effects pattern
            for _ in range(10):
                # Randomly select an effect value
                effect = np.random.choice(effects)
                
                # Apply to base support
                new_logit = base_logit + effect
                new_prob = inverse_logit(new_logit)
                poll_values.append(new_prob * 100)  # Convert to percentage
        
        poll_values = np.array(poll_values)
        
        # Calculate statistics
        min_95 = np.percentile(poll_values, 2.5)
        max_95 = np.percentile(poll_values, 97.5)
        range_95 = max_95 - min_95
        std_dev = np.std(poll_values)
        
        impact_stats.append({
            'sigma': sigma,
            'mean': np.mean(poll_values),
            'std_dev': std_dev,
            'min_95': min_95,
            'max_95': max_95,
            'range_95': range_95
        })
    
    # Print impact statistics
    print("\nImpact on polls for a party with 30% true support:")
    print("=" * 70)
    print(f"{'Sigma':<10} {'Mean':<10} {'Std Dev':<10} {'95% Range':<12} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    
    for stat in impact_stats:
        print(f"{stat['sigma']:<10.3f} {stat['mean']:<10.2f} {stat['std_dev']:<10.2f} {stat['range_95']:<12.2f} {stat['min_95']:<10.2f} {stat['max_95']:<10.2f}")
    
    # 4. Plot the relationship between sigma and range of support values
    plt.figure(figsize=(12, 8))
    
    sigmas = [stat['sigma'] for stat in impact_stats]
    ranges = [stat['range_95'] for stat in impact_stats]
    stds = [stat['std_dev'] for stat in impact_stats]
    
    plt.plot(sigmas, ranges, 'o-', color='darkblue', linewidth=2, markersize=10, label='95% Range')
    plt.plot(sigmas, [std * 2 for std in stds], 'o--', color='darkred', linewidth=2, markersize=10, label='2 Std Dev')
    
    # Add annotations
    for i, sigma in enumerate(sigmas):
        plt.annotate(f"{ranges[i]:.1f}pp", 
                    xy=(sigma, ranges[i]), 
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center')
    
    plt.title('Impact of House Effects Sigma on Poll Spread (30% Base Support)')
    plt.xlabel('Sigma Value (Logit Scale)')
    plt.ylabel('Range of Poll Results (Percentage Points)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/house_effects_check/sigma_impact.png', bbox_inches='tight')
    
    # 5. Show total poll spread for different baseline support levels
    plt.figure(figsize=(14, 8))
    
    spread_by_base = []
    for base_p in [0.10, 0.20, 0.30, 0.40, 0.50]:
        base_logit = logit(base_p)
        ranges = []
        
        for sigma in sigma_values:
            samples = results[sigma].flatten()
            
            # Apply effects to baseline
            new_logits = base_logit + samples
            new_probs = inverse_logit(new_logits) * 100  # Convert to percentage
            
            # Calculate range
            min_95 = np.percentile(new_probs, 2.5)
            max_95 = np.percentile(new_probs, 97.5)
            range_95 = max_95 - min_95
            
            ranges.append(range_95)
        
        spread_by_base.append({
            'base_support': base_p,
            'spreads': ranges
        })
    
    # Plot as grouped bar chart
    width = 0.15
    x = np.arange(len(sigma_values))
    
    for i, data in enumerate(spread_by_base):
        offset = width * (i - len(spread_by_base)/2 + 0.5)
        plt.bar(x + offset, data['spreads'], width, 
               label=f"{data['base_support']*100:.0f}% Support",
               alpha=0.7)
    
    plt.title('Poll Spread by Base Support Level and Sigma Value')
    plt.xlabel('Sigma Value')
    plt.ylabel('95% Range of Poll Results (Percentage Points)')
    plt.xticks(x, [f"{s:.3f}" for s in sigma_values])
    plt.legend(title="Base Support")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/house_effects_check/spread_by_support_level.png', bbox_inches='tight')
    
    print("\nPrior predictive checks for house effects complete.")
    print("Visualizations saved to outputs/house_effects_check/")
    
    return impact_stats

if __name__ == "__main__":
    print("Running prior predictive checks for house effects priors...")
    
    # Test a range of sigma values
    sigma_values = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
    run_house_effects_check(sigma_values) 