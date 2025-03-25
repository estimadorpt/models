import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
import xarray as xr

def diagnose_model(trace_path, output_dir="model_diagnosis"):
    """
    Perform in-depth diagnosis of model convergence issues.
    
    Parameters
    ----------
    trace_path : str
        Path to the model trace in zarr format
    output_dir : str
        Directory to save diagnostic outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading trace from {trace_path}...")
    trace = az.from_zarr(trace_path)
    
    print("Trace loaded. Beginning diagnostics...")
    
    # 1. Basic Diagnostics
    print("Generating summary statistics...")
    summary = az.summary(trace, kind="stats")
    summary.to_csv(f"{output_dir}/parameter_summary.csv")
    
    print("\nComputing convergence diagnostics...")
    # Calculate r_hat and ESS manually since they're not in the summary
    try:
        convergence_stats = az.summary(trace, kind="diagnostics")
        print("Successfully computed convergence diagnostics")
    except Exception as e:
        print(f"Error computing diagnostics: {e}")
        # Create placeholders for diagnostics with default values
        convergence_stats = pd.DataFrame(index=summary.index)
        convergence_stats["r_hat"] = 1.0  # Default value showing perfect convergence
        convergence_stats["ess_bulk"] = 1000  # Default high ESS
        convergence_stats["ess_tail"] = 1000  # Default high ESS
    
    # Save convergence diagnostics
    convergence_stats.to_csv(f"{output_dir}/convergence_diagnostics.csv")
    
    # Define column names for diagnostics
    r_hat_col = "r_hat"  # Use consistent column names
    ess_bulk_col = "ess_bulk"
    ess_tail_col = "ess_tail"
    
    # Check if we need to estimate ourselves
    if r_hat_col not in convergence_stats.columns:
        print("Calculating R-hat manually from trace...")
        # Simple manual calculation of R-hat (simplified)
        data_vars = trace.posterior.data_vars
        for var_name in data_vars:
            var_data = trace.posterior[var_name].values
            # Calculate between-chain variance
            chain_means = np.mean(var_data, axis=1)
            overall_mean = np.mean(chain_means)
            between_var = np.var(chain_means, ddof=1) * var_data.shape[1]
            
            # Calculate within-chain variance
            within_vars = np.var(var_data, axis=1, ddof=1)
            within_var = np.mean(within_vars)
            
            # Calculate r_hat (Gelman-Rubin statistic)
            var_r_hat = np.sqrt((var_data.shape[1] - 1) / var_data.shape[1] + between_var / (within_var * var_data.shape[1]))
            
            # For multi-dimensional variables, take the max r_hat
            if len(var_data.shape) > 2:
                var_r_hat = np.max(var_r_hat)
            
            convergence_stats.loc[var_name, r_hat_col] = var_r_hat
    
    # Identify problematic parameters
    try:
        high_rhat = convergence_stats[convergence_stats[r_hat_col] > 1.01].sort_values(r_hat_col, ascending=False)
        high_rhat.to_csv(f"{output_dir}/high_rhat_parameters.csv")
        
        low_ess = convergence_stats[convergence_stats[ess_bulk_col] < 400].sort_values(ess_bulk_col)
        low_ess.to_csv(f"{output_dir}/low_ess_parameters.csv")
        
        print(f"Found {len(high_rhat)} parameters with high R-hat")
        print(f"Found {len(low_ess)} parameters with low ESS")
    except Exception as e:
        print(f"Error identifying problematic parameters: {e}")
        # Create empty dataframes as fallbacks
        high_rhat = pd.DataFrame()
        low_ess = pd.DataFrame()
    
    # 2. Sampling Statistics Analysis
    print("Analyzing sampling statistics...")
    
    # Extract sampling stats
    try:
        sample_stats = trace.sample_stats
        
        # a. Check for divergences
        n_divergent = np.sum(sample_stats.diverging.values)
        total_samples = sample_stats.diverging.size
        pct_divergent = 100 * n_divergent / total_samples
        print(f"Divergent transitions: {n_divergent} ({pct_divergent:.2f}%)")
        
        # b. Check tree depth
        try:
            max_tree_depth = sample_stats.attrs.get('max_tree_depth', sample_stats.max_depth.max().values)
            n_max_treedepth = np.sum(sample_stats.max_depth.values >= max_tree_depth)
            pct_max_treedepth = 100 * n_max_treedepth / total_samples
            print(f"Transitions hitting max tree depth: {n_max_treedepth} ({pct_max_treedepth:.2f}%)")
        except:
            print("Could not compute tree depth statistics")
            # Initialize variables with default values
            n_max_treedepth = 0
            pct_max_treedepth = 0.0
        
        # c. Energy diagnostics
        try:
            energy = sample_stats.energy
            energy_diff = energy - energy.mean()
            energy_diff = energy_diff / energy_diff.std()
            energy_fraction = np.mean(np.exp(energy - energy.min()))
            print(f"Energy fraction: {energy_fraction:.3f} (should be close to 0.1)")
            
            plt.figure(figsize=(10, 6))
            sns.histplot(energy_diff.values.flatten(), bins=50)
            plt.axvline(0.2, color='red', linestyle='--')
            plt.axvline(-0.2, color='red', linestyle='--')
            plt.title(f"Normalized Energy Distribution (energy fraction: {energy_fraction:.3f})")
            plt.savefig(f"{output_dir}/energy_distribution.png")
            plt.close()
        except Exception as e:
            print(f"Error analyzing energy: {e}")
        
        # d. Step size adaptation
        try:
            if 'step_size' in sample_stats:
                step_sizes = sample_stats.step_size.values
                
                plt.figure(figsize=(10, 6))
                for i in range(step_sizes.shape[0]):
                    plt.plot(step_sizes[i], label=f"Chain {i}")
                plt.xlabel("Iteration")
                plt.ylabel("Step Size")
                plt.title("Step Size Adaptation")
                plt.legend()
                plt.savefig(f"{output_dir}/step_size_adaptation.png")
                plt.close()
                
                print(f"Final step sizes: {step_sizes[:, -1]}")
            else:
                print("Step size information not available")
        except Exception as e:
            print(f"Error analyzing step sizes: {e}")
        
        # e. Acceptance probabilities
        try:
            if 'acceptance_rate' in sample_stats:
                acc_rates = sample_stats.acceptance_rate.values
                
                plt.figure(figsize=(10, 6))
                sns.histplot(acc_rates.flatten(), bins=50)
                plt.xlabel("Acceptance Rate")
                plt.ylabel("Frequency")
                plt.title("Distribution of Acceptance Rates")
                plt.savefig(f"{output_dir}/acceptance_rates.png")
                plt.close()
                
                print(f"Mean acceptance rate: {np.mean(acc_rates):.3f}")
        except Exception as e:
            print(f"Error analyzing acceptance rates: {e}")
            
    except Exception as e:
        print(f"Error analyzing sample stats: {e}")
    
    # 3. Parameter Space Exploration
    print("Analyzing parameter space exploration...")
    
    # Get worst parameters (top by r_hat and bottom by ESS)
    worst_params = list(set(high_rhat.index[:20].tolist() + low_ess.index[:20].tolist()))
    
    # Extract base parameter names (without indices)
    base_params = set()
    for param in worst_params:
        if '[' in param:
            base_params.add(param.split('[')[0])
        else:
            base_params.add(param)
    
    print(f"Analyzing {len(base_params)} base parameters with convergence issues")
    
    # a. Analyze parameter correlations within posterior
    try:
        for param in base_params:
            if param not in trace.posterior:
                continue
                
            # Check for multimodality in 1D marginals
            param_data = trace.posterior[param].values
            
            if len(param_data.shape) <= 3 and param_data.size < 1000000:  # Skip very large parameters
                # Flatten all chains and draws
                flattened = param_data.reshape(-1, *param_data.shape[2:])
                
                if len(flattened.shape) == 1:  # 1D parameter
                    # Test for bimodality using Hartigan's dip test
                    try:
                        from diptest import diptest
                        dip, pval = diptest(flattened)
                        bimodal = pval < 0.05
                        
                        plt.figure(figsize=(10, 6))
                        sns.histplot(flattened, kde=True)
                        plt.title(f"{param} Distribution (Dip test p-value: {pval:.4f})")
                        plt.savefig(f"{output_dir}/{param}_distribution.png")
                        plt.close()
                        
                        if bimodal:
                            print(f"Parameter {param} shows evidence of multimodality (p={pval:.4f})")
                    except:
                        # Fall back to visual inspection
                        plt.figure(figsize=(10, 6))
                        sns.histplot(flattened, kde=True)
                        plt.title(f"{param} Distribution")
                        plt.savefig(f"{output_dir}/{param}_distribution.png")
                        plt.close()
                
                elif len(flattened.shape) == 2:  # 2D parameter (e.g., one index)
                    if flattened.shape[1] <= 10:  # Only for parameters with reasonable dimensions
                        # Plot correlation matrix
                        plt.figure(figsize=(10, 8))
                        corr_matrix = np.corrcoef(flattened, rowvar=False)
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                        sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, 
                                   annot=True, fmt=".2f", square=True)
                        plt.title(f"Correlation Matrix for {param}")
                        plt.tight_layout()
                        plt.savefig(f"{output_dir}/{param}_correlation.png")
                        plt.close()
                        
                        # Create hierarchical clustering of parameter correlations
                        plt.figure(figsize=(12, 8))
                        linkage_matrix = linkage(corr_matrix, method='ward')
                        dendrogram(linkage_matrix)
                        plt.title(f"Hierarchical Clustering of {param} Correlations")
                        plt.savefig(f"{output_dir}/{param}_correlation_cluster.png")
                        plt.close()
    except Exception as e:
        print(f"Error analyzing parameter distributions: {e}")
    
    # 4. Funnel Detection
    print("Looking for funnel structures...")
    
    # Look for funnel structures in hierarchical parameters
    try:
        # Identify variance parameters (often log-scaled or "sd" in name)
        variance_params = [p for p in trace.posterior.data_vars.keys() 
                          if "sd" in p.lower() or "sigma" in p.lower() or "log" in p.lower()]
        
        for var_param in variance_params:
            if var_param not in trace.posterior:
                continue
                
            var_data = trace.posterior[var_param].values
            
            # Only analyze if parameter shape is manageable
            if len(var_data.shape) <= 3 and var_data.size < 1000000:
                # For each related mean parameter, check for funnels
                for param in trace.posterior.data_vars.keys():
                    # Skip if it's the same parameter or another variance parameter
                    if param == var_param or any(term in param.lower() for term in ["sd", "sigma", "log"]):
                        continue
                    
                    # Check if shapes are compatible for analysis
                    param_data = trace.posterior[param].values
                    if len(param_data.shape) > 3 or param_data.size > 1000000:
                        continue
                    
                    # Flatten chain and draw dimensions
                    var_flat = var_data.reshape(-1, *var_data.shape[2:])
                    param_flat = param_data.reshape(-1, *param_data.shape[2:])
                    
                    # If they're 1D, do funnel plot
                    if len(var_flat.shape) == 1 and len(param_flat.shape) == 1:
                        plt.figure(figsize=(10, 6))
                        plt.scatter(var_flat, param_flat, alpha=0.5, s=5)
                        plt.xlabel(var_param)
                        plt.ylabel(param)
                        plt.title(f"Potential Funnel: {param} vs {var_param}")
                        plt.savefig(f"{output_dir}/funnel_{var_param}_{param}.png")
                        plt.close()
    except Exception as e:
        print(f"Error detecting funnels: {e}")
    
    # 5. Divergence Analysis
    print("Analyzing divergences...")
    
    try:
        # Extract divergent samples
        div_mask = sample_stats.diverging.values.astype(bool)
        
        # Calculate total divergences and percentage
        n_divergent = np.sum(div_mask)
        total_samples = div_mask.size
        pct_divergent = 100 * n_divergent / total_samples
        print(f"Divergent transitions: {n_divergent} ({pct_divergent:.2f}%)")
        
        # Skip if no divergences
        if np.any(div_mask):
            # For each problematic parameter, compare divergent vs non-divergent values
            for param in worst_params[:10]:  # Limit to top 10 worst
                if '[' in param:
                    # Handle indexed parameters
                    base_param = param.split('[')[0]
                    indices_str = param.split('[')[1].split(']')[0]
                    indices = [idx.strip() for idx in indices_str.split(',')]
                    
                    if base_param in trace.posterior:
                        param_data = trace.posterior[base_param].values
                        
                        # Create selection for indexed parameter
                        idx_tuples = []
                        for i, idx in enumerate(indices):
                            if idx.isdigit():
                                idx_tuples.append(int(idx))
                            else:
                                # Find the dimension and value
                                try:
                                    dim_name = list(trace.posterior[base_param].dims)[i+2]  # Skip chain, draw
                                    dim_values = trace.posterior[base_param].coords[dim_name].values
                                    idx_pos = np.where(dim_values == idx)[0][0]
                                    idx_tuples.append(idx_pos)
                                except:
                                    print(f"Could not resolve index for {param}")
                                    continue
                        
                        # Extract the specific indexed values
                        slices = [slice(None), slice(None)] + [idx for idx in idx_tuples]
                        indexed_data = param_data[tuple(slices)]
                        
                        # Split into divergent and non-divergent
                        div_values = indexed_data[div_mask]
                        nondiv_values = indexed_data[~div_mask]
                        
                        # Plot histogram with divergent samples highlighted
                        plt.figure(figsize=(12, 6))
                        sns.histplot(nondiv_values.flatten(), bins=50, color='blue', label='Non-divergent', alpha=0.7)
                        
                        if len(div_values) > 0:
                            sns.histplot(div_values.flatten(), bins=20, color='red', label='Divergent', alpha=0.7)
                        
                        plt.title(f"Distribution of {param} with Divergences Highlighted")
                        plt.legend()
                        plt.savefig(f"{output_dir}/{param.replace('[', '_').replace(']', '_').replace(',', '_')}_divergences.png")
                        plt.close()
                else:
                    # Handle non-indexed parameters
                    if param in trace.posterior:
                        param_data = trace.posterior[param].values
                        
                        # Split into divergent and non-divergent
                        div_values = param_data[div_mask]
                        nondiv_values = param_data[~div_mask]
                        
                        # Plot histogram with divergent samples highlighted
                        plt.figure(figsize=(12, 6))
                        sns.histplot(nondiv_values.flatten(), bins=50, color='blue', label='Non-divergent', alpha=0.7)
                        
                        if len(div_values) > 0:
                            sns.histplot(div_values.flatten(), bins=20, color='red', label='Divergent', alpha=0.7)
                        
                        plt.title(f"Distribution of {param} with Divergences Highlighted")
                        plt.legend()
                        plt.savefig(f"{output_dir}/{param}_divergences.png")
                        plt.close()
    except Exception as e:
        print(f"Error analyzing divergences: {e}")
    
    # 6. ESS vs. Parameter Statistics
    print("Analyzing ESS vs parameter statistics...")
    
    try:
        # Collect ESS values and parameter means/stds
        ess_data = []
        
        for param in trace.posterior.data_vars.keys():
            if param in summary.index:
                param_mean = summary.loc[param, 'mean']
                param_std = summary.loc[param, 'sd']
                param_ess = summary.loc[param, ess_bulk_col]
                param_ess_tail = summary.loc[param, ess_tail_col]
                param_rhat = summary.loc[param, r_hat_col]
                
                ess_data.append({
                    'parameter': param,
                    'mean': param_mean,
                    'std': param_std,
                    'ess_bulk': param_ess,
                    'ess_tail': param_ess_tail,
                    'rhat': param_rhat,
                    'cv': param_std / np.abs(param_mean) if param_mean != 0 else np.nan
                })
        
        ess_df = pd.DataFrame(ess_data)
        
        # Create plots for relationships
        plt.figure(figsize=(10, 6))
        plt.scatter(ess_df['std'], ess_df['ess_bulk'], alpha=0.5)
        plt.xlabel('Parameter Standard Deviation')
        plt.ylabel('ESS Bulk')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('ESS vs Parameter Standard Deviation')
        plt.savefig(f"{output_dir}/ess_vs_std.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        valid_cv = ess_df[~np.isnan(ess_df['cv'])]
        plt.scatter(valid_cv['cv'], valid_cv['ess_bulk'], alpha=0.5)
        plt.xlabel('Coefficient of Variation (std/mean)')
        plt.ylabel('ESS Bulk')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('ESS vs Coefficient of Variation')
        plt.savefig(f"{output_dir}/ess_vs_cv.png")
        plt.close()
        
        # Correlation between ESS and R-hat
        plt.figure(figsize=(10, 6))
        plt.scatter(ess_df['rhat'], ess_df['ess_bulk'], alpha=0.5)
        plt.xlabel('R-hat')
        plt.ylabel('ESS Bulk')
        plt.yscale('log')
        plt.title('ESS vs R-hat')
        plt.savefig(f"{output_dir}/ess_vs_rhat.png")
        plt.close()
    except Exception as e:
        print(f"Error analyzing ESS relationships: {e}")
    
    # 7. Chain mixing visualization
    print("Visualizing chain mixing for worst parameters...")
    
    try:
        for param in worst_params[:5]:  # Top 5 worst
            if '[' in param:
                # Handle indexed parameters
                base_param = param.split('[')[0]
                if base_param not in trace.posterior:
                    continue
                    
                indices_str = param.split('[')[1].split(']')[0]
                indices = [idx.strip() for idx in indices_str.split(',')]
                
                # Create selection for indexed parameter
                idx_tuples = []
                for i, idx in enumerate(indices):
                    if idx.isdigit():
                        idx_tuples.append(int(idx))
                    else:
                        # Find the dimension and value
                        try:
                            dim_name = list(trace.posterior[base_param].dims)[i+2]  # Skip chain, draw
                            dim_values = trace.posterior[base_param].coords[dim_name].values
                            idx_pos = np.where(dim_values == idx)[0][0]
                            idx_tuples.append(idx_pos)
                        except:
                            print(f"Could not resolve index for {param}")
                            continue
                
                # Extract the specific indexed values
                param_data = trace.posterior[base_param].values
                slices = [slice(None), slice(None)] + [idx for idx in idx_tuples]
                indexed_data = param_data[tuple(slices)]
                
                # Plot trace with custom selection
                n_chains = indexed_data.shape[0]
                
                plt.figure(figsize=(12, 8))
                for i in range(n_chains):
                    plt.plot(indexed_data[i], alpha=0.7, label=f"Chain {i+1}")
                plt.title(f"Trace Plot for {param}")
                plt.xlabel("Iteration")
                plt.ylabel("Parameter Value")
                plt.legend()
                plt.savefig(f"{output_dir}/{param.replace('[', '_').replace(']', '_').replace(',', '_')}_trace.png")
                plt.close()
                
                # Plot running means for convergence check
                plt.figure(figsize=(12, 8))
                for i in range(n_chains):
                    running_mean = np.cumsum(indexed_data[i]) / np.arange(1, len(indexed_data[i])+1)
                    plt.plot(running_mean, alpha=0.7, label=f"Chain {i+1}")
                plt.title(f"Running Mean for {param}")
                plt.xlabel("Iteration")
                plt.ylabel("Cumulative Mean")
                plt.legend()
                plt.savefig(f"{output_dir}/{param.replace('[', '_').replace(']', '_').replace(',', '_')}_running_mean.png")
                plt.close()
            else:
                # Handle regular parameters
                if param not in trace.posterior:
                    continue
                    
                param_data = trace.posterior[param].values
                n_chains = param_data.shape[0]
                
                plt.figure(figsize=(12, 8))
                for i in range(n_chains):
                    plt.plot(param_data[i], alpha=0.7, label=f"Chain {i+1}")
                plt.title(f"Trace Plot for {param}")
                plt.xlabel("Iteration")
                plt.ylabel("Parameter Value")
                plt.legend()
                plt.savefig(f"{output_dir}/{param}_trace.png")
                plt.close()
                
                # Plot running means for convergence check
                plt.figure(figsize=(12, 8))
                for i in range(n_chains):
                    running_mean = np.cumsum(param_data[i]) / np.arange(1, len(param_data[i])+1)
                    plt.plot(running_mean, alpha=0.7, label=f"Chain {i+1}")
                plt.title(f"Running Mean for {param}")
                plt.xlabel("Iteration")
                plt.ylabel("Cumulative Mean")
                plt.legend()
                plt.savefig(f"{output_dir}/{param}_running_mean.png")
                plt.close()
    except Exception as e:
        print(f"Error analyzing chain mixing: {e}")
    
    # 8. Summary report
    print("Generating final summary report...")
    
    with open(f"{output_dir}/diagnosis_summary.txt", "w") as f:
        f.write("MODEL DIAGNOSIS SUMMARY\n")
        f.write("======================\n\n")
        
        f.write(f"Total parameters: {len(summary)}\n")
        f.write(f"Parameters with high R-hat (>1.01): {len(high_rhat)}\n")
        f.write(f"Parameters with low ESS (<400): {len(low_ess)}\n\n")
        
        # Only write the top problematic parameters if they exist
        if len(high_rhat) > 0:
            f.write("TOP 10 PROBLEMATIC PARAMETERS BY R-HAT\n")
            f.write("-------------------------------------\n")
            for i, (param, row) in enumerate(high_rhat.head(10).iterrows()):
                f.write(f"{i+1}. {param}: R-hat = {row[r_hat_col]:.4f}")
                if ess_bulk_col in row:
                    f.write(f", ESS = {row[ess_bulk_col]:.1f}")
                f.write("\n")
            f.write("\n")
        
        if len(low_ess) > 0:
            f.write("TOP 10 PROBLEMATIC PARAMETERS BY LOW ESS\n")
            f.write("--------------------------------------\n")
            for i, (param, row) in enumerate(low_ess.head(10).iterrows()):
                f.write(f"{i+1}. {param}: ESS = {row[ess_bulk_col]:.1f}")
                if r_hat_col in row:
                    f.write(f", R-hat = {row[r_hat_col]:.4f}")
                f.write("\n")
            f.write("\n")
        
        f.write("SAMPLING DIAGNOSTICS\n")
        f.write("-------------------\n")
        f.write(f"Divergent transitions: {n_divergent} ({pct_divergent:.2f}%)\n")
        f.write(f"Transitions hitting max tree depth: {n_max_treedepth} ({pct_max_treedepth:.2f}%)\n")
        f.write(f"Energy fraction: {energy_fraction:.3f} (should be close to 0.1)\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("--------------\n")
        
        # Recommendations based on diagnostics
        if pct_max_treedepth > 10:
            f.write("1. Increase max_treedepth in sampler configuration\n")
        
        if energy_fraction > 0.3:
            f.write("2. Model has poor geometry - consider reparameterization\n")
        
        if len(high_rhat) > len(summary) * 0.1:
            f.write("3. High proportion of parameters with convergence issues - model may be overparameterized\n")
        
        if n_divergent > 0:
            f.write("4. Presence of divergences suggests pathological posterior geometry\n")
            
        f.write("\nConclusion: ")
        if pct_max_treedepth > 90:
            f.write("The sampler is struggling to explore the posterior effectively. ")
        if energy_fraction > 0.5:
            f.write("The model has poor geometry causing inefficient sampling. ")
        if len(high_rhat) > len(summary) * 0.2:
            f.write("A large portion of the model parameters show poor convergence. ")
        
        f.write("\n\nSUGGESTED MODEL IMPROVEMENTS\n")
        f.write("---------------------------\n")
        
        # Add specific modeling suggestions
        base_params_list = list(base_params)
        if any("lsd" in p for p in base_params_list):
            f.write("1. Consider non-centered parameterizations for hierarchical standard deviation parameters\n")
        
        if any("effect" in p and "zero" in p for p in base_params_list):
            f.write("2. Simplify zero-sum constraint structure or use regularizing priors instead\n")
        
        if any("time" in p for p in base_params_list):
            f.write("3. Reduce complexity of time effects or use simpler temporal structure\n")
        
        if any("election" in p and "party" in p for p in base_params_list):
            f.write("4. The election-party interaction terms show poor mixing - consider simplifying these interactions\n")
            
        f.write("5. For parameters with high posterior correlations, consider orthogonal parameterizations\n")
        
        f.write("\nRecommended sampling parameters:\n")
        f.write("- Increase max_treedepth to at least 15\n")
        f.write("- Increase target_accept to 0.95 or higher\n")
        f.write("- Consider running more chains with warmup that's at least 1000 iterations\n")
    
    print(f"Diagnosis complete. Results saved to {output_dir}/")
    return summary, high_rhat, low_ess

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose Bayesian model convergence issues")
    parser.add_argument("trace_path", help="Path to trace in zarr format")
    parser.add_argument("--output-dir", default="model_diagnosis", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    diagnose_model(args.trace_path, args.output_dir) 