#!/usr/bin/env python
"""
Script to analyze historical polling errors and recommend appropriate house effect priors.
This will help us understand the scale of house effects and suggest reasonable sigma values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
import pymc as pm

# Ensure PYTHONPATH contains the root directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import ElectionDataset
from src.data.loaders import load_election_results

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

def calculate_polling_errors():
    """Calculate polling errors for each pollster and party."""
    # Load dataset for the most recent election
    print("Loading dataset...")
    dataset = ElectionDataset(
        election_date="2024-03-10",  # Most recent election
        baseline_timescales=[365],
        election_timescales=[60],
    )
    
    # Get all polls
    polls = dataset.polls.copy()
    
    # Get the actual election results
    results = dataset.results_mult.copy()
    
    # Analyze only the final polls (e.g., within 14 days of the election)
    final_window_days = 14
    
    # Analysis results
    all_errors = []
    pollster_errors = {}
    party_errors = {}
    
    # Process each election
    for election_date in dataset.historical_election_dates:
        print(f"\nAnalyzing election: {election_date}")
        
        # Get election results
        election_result = results[results['election_date'] == election_date].iloc[0]
        result_values = {}
        
        # Extract party vote shares from results
        political_families = dataset.political_families
        for party in political_families:
            if party in election_result:
                result_values[party] = election_result[party]
            else:
                result_values[party] = 0.0
                print(f"Warning: {party} not found in {election_date} results, assuming 0%")
        
        # Get final polls for this election
        election_polls = polls[polls['election_date'] == election_date].copy()
        election_date_dt = pd.to_datetime(election_date)
        final_polls = election_polls[
            pd.to_datetime(election_polls['date']) >= (election_date_dt - pd.Timedelta(days=final_window_days))
        ]
        
        print(f"Found {len(final_polls)} polls in final {final_window_days} days before election")
        
        # Calculate errors for each poll
        for _, poll in final_polls.iterrows():
            pollster = poll['pollster']
            poll_date = poll['date']
            
            # Initialize error structures if not already
            if pollster not in pollster_errors:
                pollster_errors[pollster] = {
                    'raw_errors': [], 
                    'abs_errors': [], 
                    'logit_errors': [],
                    'parties': {}
                }
            
            # Calculate errors for each party
            for party in political_families:
                if party not in poll:
                    continue
                
                # Extract poll value and actual result
                poll_value = poll[party]
                result_value = result_values.get(party, 0.0)
                
                # Calculate raw error (poll - actual)
                raw_error = poll_value - result_value
                
                # Calculate absolute error
                abs_error = abs(raw_error)
                
                # Calculate error on logit scale
                try:
                    poll_logit = logit(poll_value)
                    result_logit = logit(result_value)
                    logit_error = poll_logit - result_logit
                except:
                    # Skip calculations if logit transformation fails
                    logit_error = np.nan
                
                # Store error data
                error_data = {
                    'election_date': election_date,
                    'pollster': pollster,
                    'poll_date': poll_date,
                    'party': party,
                    'poll_value': poll_value,
                    'result_value': result_value,
                    'raw_error': raw_error,
                    'abs_error': abs_error,
                    'logit_error': logit_error
                }
                
                all_errors.append(error_data)
                
                # Add to pollster errors
                pollster_errors[pollster]['raw_errors'].append(raw_error)
                pollster_errors[pollster]['abs_errors'].append(abs_error)
                pollster_errors[pollster]['logit_errors'].append(logit_error)
                
                # Initialize party-specific errors for this pollster
                if party not in pollster_errors[pollster]['parties']:
                    pollster_errors[pollster]['parties'][party] = {
                        'raw_errors': [], 
                        'abs_errors': [], 
                        'logit_errors': []
                    }
                
                # Add to party-specific pollster errors
                pollster_errors[pollster]['parties'][party]['raw_errors'].append(raw_error)
                pollster_errors[pollster]['parties'][party]['abs_errors'].append(abs_error)
                pollster_errors[pollster]['parties'][party]['logit_errors'].append(logit_error)
                
                # Initialize party errors if not already
                if party not in party_errors:
                    party_errors[party] = {
                        'raw_errors': [], 
                        'abs_errors': [], 
                        'logit_errors': []
                    }
                
                # Add to party errors
                party_errors[party]['raw_errors'].append(raw_error)
                party_errors[party]['abs_errors'].append(abs_error)
                party_errors[party]['logit_errors'].append(logit_error)
    
    # Convert to DataFrame
    all_errors_df = pd.DataFrame(all_errors)
    
    return all_errors_df, pollster_errors, party_errors

def analyze_errors(all_errors_df, pollster_errors, party_errors):
    """Analyze polling errors and suggest prior values."""
    print("\n===== ANALYZING POLLING ERRORS =====")
    
    # Overall error statistics
    print("\nOverall Error Statistics:")
    print(f"Number of poll-party combinations analyzed: {len(all_errors_df)}")
    
    # Raw error (percentage points)
    raw_mean = all_errors_df['raw_error'].mean()
    raw_std = all_errors_df['raw_error'].std()
    raw_abs_mean = all_errors_df['abs_error'].mean()
    print(f"Mean error (percentage points): {raw_mean:.3f}")
    print(f"Standard deviation of errors (percentage points): {raw_std:.3f}")
    print(f"Mean absolute error (percentage points): {raw_abs_mean:.3f}")
    
    # Logit scale errors
    logit_df = all_errors_df.dropna(subset=['logit_error'])
    logit_mean = logit_df['logit_error'].mean()
    logit_std = logit_df['logit_error'].std() 
    print(f"Mean error (logit scale): {logit_mean:.3f}")
    print(f"Standard deviation of errors (logit scale): {logit_std:.3f}")
    
    # Analyze by pollster
    print("\nError by Pollster (Mean Absolute Error in percentage points):")
    pollster_summary = []
    for pollster, data in pollster_errors.items():
        mean_abs_error = np.mean(data['abs_errors'])
        std_abs_error = np.std(data['abs_errors'])
        logit_std = np.nanstd(data['logit_errors'])
        n_polls = len(data['abs_errors'])
        pollster_summary.append({
            'pollster': pollster,
            'mean_abs_error': mean_abs_error,
            'std_abs_error': std_abs_error,
            'logit_std': logit_std,
            'n_polls': n_polls
        })
    
    pollster_df = pd.DataFrame(pollster_summary)
    pollster_df = pollster_df.sort_values('mean_abs_error')
    print(pollster_df[['pollster', 'mean_abs_error', 'logit_std', 'n_polls']].to_string(index=False))
    
    # Analyze by party
    print("\nError by Party (Mean Absolute Error in percentage points):")
    party_summary = []
    for party, data in party_errors.items():
        mean_abs_error = np.mean(data['abs_errors'])
        std_abs_error = np.std(data['abs_errors'])
        logit_std = np.nanstd(data['logit_errors'])
        n_polls = len(data['abs_errors'])
        party_summary.append({
            'party': party,
            'mean_abs_error': mean_abs_error,
            'std_abs_error': std_abs_error,
            'logit_std': logit_std,
            'n_polls': n_polls
        })
    
    party_df = pd.DataFrame(party_summary)
    party_df = party_df.sort_values('mean_abs_error')
    print(party_df[['party', 'mean_abs_error', 'logit_std', 'n_polls']].to_string(index=False))
    
    # Generate recommended sigma values based on logit errors
    pollster_party_logit_stds = []
    for pollster, data in pollster_errors.items():
        for party, party_data in data['parties'].items():
            if len(party_data['logit_errors']) >= 3:  # Only consider if we have enough data
                logit_std = np.nanstd(party_data['logit_errors'])
                if not np.isnan(logit_std):
                    pollster_party_logit_stds.append(logit_std)
    
    # Filter out extreme values for more stable estimates
    filtered_logit_stds = [x for x in pollster_party_logit_stds if x < 2.0]  
    median_logit_std = np.median(filtered_logit_stds)
    
    print("\n===== RECOMMENDED PRIOR VALUES =====")
    print(f"Recommended sigma for house_effects: {median_logit_std:.3f}")
    print(f"Recommended sigma for house_election_effects_sd: {median_logit_std/2:.3f}")
    
    # Simulate impact of different sigma values
    simulate_sigma_impacts(median_logit_std)
    
    # Create visualizations
    create_visualizations(all_errors_df, pollster_df, party_df, pollster_party_logit_stds)
    
    return median_logit_std

def simulate_sigma_impacts(base_sigma):
    """Simulate the impact of different sigma values on percentage scale."""
    print("\n===== SIMULATING SIGMA IMPACTS =====")
    
    # Starting probabilities to transform (different baseline levels)
    base_probs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    # Range of sigma values to test
    sigma_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    print("\nImpact on percentage points by baseline level:")
    print("Sigma | Starting at:")
    header = "      |" + "".join(f" {p*100:4.1f}% |" for p in base_probs)
    print(header)
    print("-" * len(header))
    
    for sigma in sigma_values:
        # Generate 10,000 random samples for each sigma
        samples = np.random.normal(0, sigma, 10000)
        
        impacts = []
        for base_p in base_probs:
            # Convert base probability to logit
            base_logit = logit(base_p)
            
            # Add samples to logit
            new_logits = base_logit + samples
            
            # Convert back to probability
            new_probs = inverse_logit(new_logits)
            
            # Calculate change in percentage points
            changes = (new_probs - base_p) * 100
            
            # Calculate 95% range of impacts
            impact_95 = np.percentile(np.abs(changes), 95)
            impacts.append(impact_95)
        
        # Print results
        impacts_str = "".join(f" {i:4.1f}pp |" for i in impacts)
        print(f"{sigma:.2f} |{impacts_str}")
    
    print("\nInterpretation: For each sigma value, the table shows the typical (95th percentile)")
    print("maximum impact in percentage points at different baseline popularity levels.")

def create_visualizations(all_errors_df, pollster_df, party_df, pollster_party_logit_stds):
    """Create visualizations for the error analysis."""
    print("\nCreating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs/analysis", exist_ok=True)
    
    # 1. Distribution of raw errors
    plt.figure(figsize=(10, 6))
    sns.histplot(all_errors_df['raw_error'], bins=30, kde=True)
    plt.title('Distribution of Raw Polling Errors (Percentage Points)')
    plt.xlabel('Error (Poll - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.axvline(0, color='red', linestyle='-', alpha=0.7)
    plt.savefig('outputs/analysis/raw_error_distribution.png', bbox_inches='tight')
    
    # 2. Distribution of logit errors
    plt.figure(figsize=(10, 6))
    sns.histplot(all_errors_df['logit_error'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Polling Errors on Logit Scale')
    plt.xlabel('Error (Poll - Actual) on Logit Scale')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.axvline(0, color='red', linestyle='-', alpha=0.7)
    plt.savefig('outputs/analysis/logit_error_distribution.png', bbox_inches='tight')
    
    # 3. Error by pollster
    plt.figure(figsize=(12, 8))
    pollster_plot = pollster_df.sort_values('mean_abs_error', ascending=False).head(15)  # Top 15 pollsters
    ax = sns.barplot(x='pollster', y='mean_abs_error', data=pollster_plot)
    plt.title('Mean Absolute Error by Pollster (Percentage Points)')
    plt.xlabel('Pollster')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/analysis/error_by_pollster.png', bbox_inches='tight')
    
    # 4. Error by party
    plt.figure(figsize=(10, 6))
    party_plot = party_df.sort_values('mean_abs_error', ascending=False)
    ax = sns.barplot(x='party', y='mean_abs_error', data=party_plot)
    plt.title('Mean Absolute Error by Party (Percentage Points)')
    plt.xlabel('Party')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/analysis/error_by_party.png', bbox_inches='tight')
    
    # 5. Scatterplot of poll vs actual results
    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(
        x='result_value', 
        y='poll_value', 
        hue='party',
        alpha=0.6,
        data=all_errors_df
    )
    # Add perfect prediction line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.title('Poll Values vs. Actual Results')
    plt.xlabel('Actual Result')
    plt.ylabel('Poll Prediction')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/analysis/poll_vs_actual.png', bbox_inches='tight')
    
    # 6. Distribution of logit standard deviations
    plt.figure(figsize=(10, 6))
    sns.histplot(pollster_party_logit_stds, bins=30, kde=True)
    plt.title('Distribution of Logit Error Standard Deviations by Pollster-Party')
    plt.xlabel('Standard Deviation of Logit Errors')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.median(pollster_party_logit_stds), color='red', linestyle='-', alpha=0.7, 
                label=f'Median: {np.median(pollster_party_logit_stds):.3f}')
    plt.legend()
    plt.savefig('outputs/analysis/logit_std_distribution.png', bbox_inches='tight')
    
    # 7. Simulation of sigma impacts on different base probabilities
    base_probs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    sigma_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    plt.figure(figsize=(12, 8))
    for i, base_p in enumerate(base_probs):
        impacts = []
        for sigma in sigma_values:
            samples = np.random.normal(0, sigma, 10000)
            base_logit = logit(base_p)
            new_logits = base_logit + samples
            new_probs = inverse_logit(new_logits)
            changes = np.abs(new_probs - base_p) * 100
            impact_95 = np.percentile(changes, 95)
            impacts.append(impact_95)
        
        plt.plot(sigma_values, impacts, marker='o', label=f'Base: {base_p*100:.1f}%')
    
    plt.title('Impact of Sigma Values on Percentage Points (95th Percentile)')
    plt.xlabel('Sigma Value (Logit Scale)')
    plt.ylabel('Maximum Impact (Percentage Points)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Starting Popularity')
    plt.tight_layout()
    plt.savefig('outputs/analysis/sigma_impacts.png', bbox_inches='tight')
    
    print(f"Visualizations saved to outputs/analysis/")

def generate_recommended_priors(median_logit_std):
    """Generate recommended prior configurations for model."""
    print("\n===== RECOMMENDED MODEL CONFIGURATIONS =====")
    
    # Conservative configuration
    conservative_sigma = median_logit_std * 0.5
    # Moderate configuration (based on median)
    moderate_sigma = median_logit_std
    # Liberal configuration
    liberal_sigma = median_logit_std * 1.5
    
    configurations = {
        "Conservative": {
            "house_effects_sigma": conservative_sigma,
            "house_election_effects_sd_sigma": conservative_sigma * 0.5
        },
        "Data-driven (Recommended)": {
            "house_effects_sigma": moderate_sigma,
            "house_election_effects_sd_sigma": moderate_sigma * 0.5
        },
        "Liberal": {
            "house_effects_sigma": liberal_sigma,
            "house_election_effects_sd_sigma": liberal_sigma * 0.5
        }
    }
    
    print("\nRecommended Prior Configurations:")
    for name, config in configurations.items():
        print(f"\n{name} Configuration:")
        print(f"    house_effects = pm.ZeroSumNormal(")
        print(f"        \"house_effects\",")
        print(f"        sigma={config['house_effects_sigma']:.3f},")
        print(f"        dims=(\"pollsters\", \"parties_complete\"),")
        print(f"    )")
        print(f"")
        print(f"    house_election_effects_sd = pm.HalfNormal(")
        print(f"        \"house_election_effects_sd\",")
        print(f"        sigma={config['house_election_effects_sd_sigma']:.3f},")
        print(f"        dims=(\"pollsters\", \"parties_complete\"),")
        print(f"    )")
        
        # Calculate typical impacts
        for sigma_name, sigma_val in [
            ("house_effects_sigma", config['house_effects_sigma']),
            ("house_election_effects_sd_sigma", config['house_election_effects_sd_sigma'])
        ]:
            typical_impact_30pct = np.percentile(
                np.abs(inverse_logit(logit(0.3) + np.random.normal(0, sigma_val, 10000)) - 0.3) * 100, 
                95
            )
            typical_impact_50pct = np.percentile(
                np.abs(inverse_logit(logit(0.5) + np.random.normal(0, sigma_val, 10000)) - 0.5) * 100, 
                95
            )
            print(f"    # {sigma_name}: Typical impact at 30% popularity: ±{typical_impact_30pct:.1f}pp")
            print(f"    # {sigma_name}: Typical impact at 50% popularity: ±{typical_impact_50pct:.1f}pp")

if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs("outputs/analysis", exist_ok=True)
    
    # Calculate polling errors
    print("Analyzing polling errors...")
    all_errors_df, pollster_errors, party_errors = calculate_polling_errors()
    
    # Analyze errors and suggest prior values
    median_logit_std = analyze_errors(all_errors_df, pollster_errors, party_errors)
    
    # Generate recommended prior configurations
    generate_recommended_priors(median_logit_std) 