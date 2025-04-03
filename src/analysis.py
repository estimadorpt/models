import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from joblib import Parallel, delayed
from scipy.special import softmax
from src.models.election_model import ElectionModel
import time
import os

# Define the refined color palette for Portuguese parties
party_colors = {
    'PS': '#FF85A2',  # Soft rose pink
    'CH': '#7C7C7C',  # Muted dark gray
    'IL': '#5BC0EB',  # Sky blue
    'BE': '#FF6B6B',  # Coral red
    'CDU': '#4EA855',  # Forest green
    'PAN': '#9BC53D',  # Lime green
    'L': '#FDE74C',  # Sunshine yellow
    'AD': '#FA9F42',  # Soft orange
    'Volatility': '#D3D3D3'  # Light grey
}

def compute_diagnostics_parallel(idata):
    var_names = list(idata.posterior.data_vars)
    
    def compute_single_diagnostic(var):
        r_hat = az.rhat(idata, var_names=[var])
        ess_bulk = az.ess(idata, var_names=[var], method="bulk")
        ess_tail = az.ess(idata, var_names=[var], method="tail")
        return var, {
            "r_hat": r_hat[var].values,
            "ess_bulk": ess_bulk[var].values,
            "ess_tail": ess_tail[var].values,
        }

    diagnostics = Parallel(n_jobs=1)(delayed(compute_single_diagnostic)(var) for var in var_names)
    return dict(diagnostics)

def run_diagnostics(idata, prior, posterior, model, polls_train, polls_test, results_mult, election_dates, government_parties):
    start_time = time.time()
    print("Running model diagnostics...")
    
    # # Compute diagnostics in parallel for all variables
    # diagnostics_start_time = time.time()
    # diagnostics = compute_diagnostics_parallel(idata)
    # diagnostics_end_time = time.time()
    # print(f"Diagnostics computation took {diagnostics_end_time - diagnostics_start_time:.2f} seconds")
    
    # # Write diagnostics to a file
    # with open("diagnostics_summary.txt", "w") as f:
    #     f.write("Convergence and ESS Summary:\n")
    #     for var, diag in diagnostics.items():
    #         f.write(f"{var}: r_hat={diag['r_hat']}, ess_bulk={diag['ess_bulk']}, ess_tail={diag['ess_tail']}\n")

    #     # Highlight variables with potential issues
    #     problematic_vars = {var: diag for var, diag in diagnostics.items() if (diag["r_hat"] > 1.01).any() or (diag["ess_bulk"] < 400).any() or (diag["ess_tail"] < 400).any()}
    #     if problematic_vars:
    #         f.write("\nVariables with potential issues:\n")
    #         for var, diag in problematic_vars.items():
    #             f.write(f"{var}: r_hat={diag['r_hat']}, ess_bulk={diag['ess_bulk']}, ess_tail={diag['ess_tail']}\n")
    #     else:
    #         f.write("\nNo variables with obvious convergence or ESS issues.\n")

    #     # Write mean of the posterior of each parameter
    #     f.write("\nMean of the Posterior of Each Parameter:\n")
    #     for var in idata.posterior.data_vars:
    #         mean_value = idata.posterior[var].mean().values
    #         f.write(f"{var}: mean={mean_value}\n")

    # # Prior predictive checks
    # prior_ppc_start_time = time.time()
    # print("\nPerforming prior predictive checks...")
    # az.plot_ppc(prior, kind="cumulative", group="prior")
    # plt.title("Prior Predictive Check")
    # plt.savefig("prior_predictive_check.png")
    # plt.close()
    # prior_ppc_end_time = time.time()
    # print(f"Prior predictive checks took {prior_ppc_end_time - prior_ppc_start_time:.2f} seconds")

    # # Posterior predictive checks
    # posterior_ppc_start_time = time.time()
    # print("Performing posterior predictive checks...")
    # az.plot_ppc(posterior, kind="cumulative", group="posterior")
    # plt.title("Posterior Predictive Check")
    # plt.savefig("posterior_predictive_check.png")
    # plt.close()
    # posterior_ppc_end_time = time.time()
    # print(f"Posterior predictive checks took {posterior_ppc_end_time - posterior_ppc_start_time:.2f} seconds")

    # # Posterior distribution insights
    # posterior_insights_start_time = time.time()
    # print("\nGenerating posterior distribution insights...")
    # interesting_params = ["party_baseline", "poll_bias", "concentration_polls", "concentration_results"]
    # for param in interesting_params:
    #     az.plot_posterior(idata, var_names=[param])
    #     plt.title(f"Posterior Distribution of {param}")
    #     plt.savefig(f"posterior_{param}.png")
    #     plt.close()
    # posterior_insights_end_time = time.time()
    # print(f"Posterior distribution insights took {posterior_insights_end_time - posterior_insights_start_time:.2f} seconds")

    # # Trace plots for key parameters
    # trace_plots_start_time = time.time()
    # az.plot_trace(idata, var_names=interesting_params)
    # plt.savefig("trace_plots.png")
    # plt.close()
    # trace_plots_end_time = time.time()
    # print(f"Trace plots took {trace_plots_end_time - trace_plots_start_time:.2f} seconds")

    # Evolution of latent popularity over time
    latent_popularity_start_time = time.time()
    print("\nPlotting evolution of latent popularity...")
    plot_latent_popularity(idata, polls_train, results_mult)
    plot_latent_popularity_last_year(idata, polls_train, results_mult)
    latent_popularity_end_time = time.time()
    print(f"Plotting evolution of latent popularity took {latent_popularity_end_time - latent_popularity_start_time:.2f} seconds")

    # Additional Portuguese election insights
    additional_insights_start_time = time.time()
    print("\nAnalyzing Portuguese election insights...")
    #plot_government_opposition_trends(idata, polls_train, election_dates, government_parties)
    plot_party_volatility(idata, polls_train)
    plot_party_correlations(idata)
    analyze_polling_accuracy(idata, polls_train)
    additional_insights_end_time = time.time()
    print(f"Additional Portuguese election insights took {additional_insights_end_time - additional_insights_start_time:.2f} seconds")

    # Summarize house effects
    house_effects_start_time = time.time()
    print("\nSummarizing house effects...")
    summarize_house_effects(idata, polls_train)
    house_effects_end_time = time.time()
    print(f"Summarizing house effects took {house_effects_end_time - house_effects_start_time:.2f} seconds")

    # Plot and save visualizations for all variables
    plot_variables_start_time = time.time()
    print("\nPlotting variables...")
    plot_all_variables(idata, polls_train)
    plot_variables_end_time = time.time()
    print(f"Plotting variables took {plot_variables_end_time - plot_variables_start_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total diagnostics run took {total_time:.2f} seconds")

def summarize_house_effects(idata, polls_train):
    house_effects = idata.posterior['house_effects'].mean(dim=['chain', 'draw'])
    pollsters = polls_train['pollster'].unique()
    parties = idata.posterior.coords['parties_complete'].values

    # Flatten the house_effects array and create a DataFrame
    house_effects_flat = house_effects.values.flatten()
    pollsters_repeated = np.repeat(pollsters, len(parties))
    parties_tiled = np.tile(parties, len(pollsters))

    df = pd.DataFrame({
        'Pollster': pollsters_repeated,
        'Party': parties_tiled,
        'House Effect': house_effects_flat
    })

    # Create a plot for each pollster
    for pollster in pollsters:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Party', 
            y='House Effect', 
            data=df[df['Pollster'] == pollster], 
            palette=party_colors,
            hue='Party',
            legend=False
        )
        plt.title(f'House Effects for {pollster}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Create a valid filename by replacing slashes with underscores
        safe_pollster = pollster.replace('/', '_')
        filename = f"house_effects_{safe_pollster}.png"
        filepath = os.path.join(os.getcwd(), filename)
        plt.savefig(filepath)
        plt.close()

    print(f"House effects plots saved in {os.getcwd()}")

def plot_latent_popularity(idata, polls_train, results_mult):
    # Extract the mean and credible intervals for latent popularity
    latent_popularity_mean = idata.posterior["latent_popularity"].mean(("chain", "draw"))
    latent_popularity_hdi = az.hdi(idata.posterior["latent_popularity"], hdi_prob=0.95)
    
    parties = latent_popularity_mean.coords["parties_complete"].values

    plt.figure(figsize=(12, 8))
    for i, party in enumerate(parties):
        mean_values = latent_popularity_mean.sel(parties_complete=party).values
        
        # Calculate HDI for the specific party
        hdi = latent_popularity_hdi.sel(parties_complete=party)
        
        # Access lower and upper bounds for the specific party
        hdi_lower = hdi.sel(hdi='lower').to_dataarray().values.flatten()
        hdi_upper = hdi.sel(hdi='higher').to_dataarray().values.flatten()

        # Plot latent popularity
        plt.plot(polls_train["date"], mean_values, label=f"{party} (Latent)", color=party_colors[party])
        plt.fill_between(polls_train["date"], hdi_lower, hdi_upper, alpha=0.3, color=party_colors[party])

        # Plot observed values
        observed_values = polls_train[party] / polls_train['sample_size']
        plt.scatter(polls_train["date"], observed_values, alpha=0.5, s=20, label=f"{party} (Observed)", color=party_colors[party])

    # Add vertical lines for election dates
    for _, row in results_mult.iterrows():
        date = row['date']
        plt.axvline(x=pd.to_datetime(date), color='gray', linestyle='--', alpha=0.5)
        
        # Plot actual election results
        for party in parties:
            if party in row.index and not pd.isna(row[party]):
                result = row[party] / row['sample_size']  # Calculate percentage
                plt.scatter(pd.to_datetime(date), result, marker='D', s=100, color=party_colors[party], 
                            edgecolor='black', linewidth=1.5, zorder=5)

    plt.title("Evolution of Latent Party Popularity, Observed Poll Results, and Election Results Over Time")
    plt.xlabel("Date")
    plt.ylabel("Popularity / Poll Share")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("latent_popularity_evolution_with_observed_and_results.png", bbox_inches='tight')
    plt.close()

def plot_latent_popularity_last_year(idata, polls_train, results_mult):
    # Extract the mean and credible intervals for latent popularity
    latent_popularity_mean = idata.posterior["latent_popularity"].mean(("chain", "draw"))
    latent_popularity_hdi = az.hdi(idata.posterior["latent_popularity"], hdi_prob=0.95)
    
    parties = latent_popularity_mean.coords["parties_complete"].values

    # Calculate the date one year ago from the most recent date in polls_train
    last_year_start = pd.to_datetime(polls_train["date"].max()) - pd.DateOffset(years=1)

    plt.figure(figsize=(12, 8))
    for i, party in enumerate(parties):
        mean_values = latent_popularity_mean.sel(parties_complete=party).values
        
        # Calculate HDI for the specific party
        hdi = latent_popularity_hdi.sel(parties_complete=party)
        
        # Access lower and upper bounds for the specific party
        hdi_lower = hdi.sel(hdi='lower').to_dataarray().values.flatten()
        hdi_upper = hdi.sel(hdi='higher').to_dataarray().values.flatten()

        # Plot latent popularity
        plt.plot(polls_train["date"], mean_values, label=f"{party} (Latent)", color=party_colors[party])
        plt.fill_between(polls_train["date"], hdi_lower, hdi_upper, alpha=0.3, color=party_colors[party])

        # Plot observed values
        observed_values = polls_train[party] / polls_train['sample_size']
        plt.scatter(polls_train["date"], observed_values, alpha=0.5, s=20, label=f"{party} (Observed)", color=party_colors[party])

    # Add vertical lines for election dates and plot actual results
    for _, row in results_mult.iterrows():
        date = pd.to_datetime(row['date'])
        if date >= last_year_start:
            plt.axvline(x=date, color='gray', linestyle='--', alpha=0.5)
        
            # Plot actual election results
            for party in parties:
                if party in row.index and not pd.isna(row[party]):
                    result = row[party] / row['sample_size']  # Calculate percentage
                    plt.scatter(date, result, marker='D', s=100, color=party_colors[party], 
                                edgecolor='black', linewidth=1.5, zorder=5)

    plt.title("Evolution of Latent Party Popularity (Last Year)")
    plt.xlabel("Date")
    plt.ylabel("Popularity / Poll Share")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.xlim(last_year_start, polls_train["date"].max())
    plt.tight_layout()
    plt.savefig("latent_popularity_evolution_last_year.png", bbox_inches='tight')
    plt.close()


def plot_government_opposition_trends(idata, polls_train, election_dates, government_parties):
    latent_popularity = idata.posterior["latent_popularity"].mean(("chain", "draw"))
    parties = latent_popularity.coords["parties_complete"].values

    plt.figure(figsize=(12, 8))
    for date in election_dates:
        plt.axvline(pd.to_datetime(date), color='gray', linestyle='--', alpha=0.5)

    for party in parties:
        is_government = [party in gov_parties for gov_parties in government_parties.values()]
        color = 'red' if any(is_government) else 'blue'
        plt.plot(polls_train["date"], latent_popularity.sel(parties_complete=party), 
                 label=party, color=party_colors[party], alpha=0.7)

    plt.title("Government vs Opposition Party Trends")
    plt.xlabel("Date")
    plt.ylabel("Latent Popularity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("government_opposition_trends.png")
    plt.close()

def plot_party_volatility(idata, polls_train):
    latent_popularity = idata.posterior["latent_popularity"]
    parties = latent_popularity.coords["parties_complete"].values

    volatility = latent_popularity.std(("chain", "draw"))

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=volatility.to_dataframe().melt(var_name='Party', value_name='Volatility'), palette=party_colors)
    plt.title("Party Popularity Volatility")
    plt.ylabel("Volatility (Standard Deviation)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("party_volatility.png")
    plt.close()

def plot_party_correlations(idata):
    latent_popularity = idata.posterior["latent_popularity"].mean(("chain", "draw"))
    parties = latent_popularity.coords["parties_complete"].values

    # Debug print to inspect dimensions
    print("Dimensions of latent_popularity:", latent_popularity.dims)

    # Adjust stacking based on actual dimensions
    if "elections" in latent_popularity.dims and "countdown" in latent_popularity.dims:
        latent_popularity_2d = latent_popularity.stack(observations=("elections", "countdown")).values
    elif "observations" in latent_popularity.dims:
        latent_popularity_2d = latent_popularity.values
    else:
        raise ValueError("Unexpected dimensions in latent_popularity")

    corr_matrix = np.corrcoef(latent_popularity_2d.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=parties, yticklabels=parties)
    plt.title("Party Popularity Correlations")
    plt.tight_layout()
    plt.savefig("party_correlations.png")
    plt.close()

def analyze_polling_accuracy(idata, polls_train):
    latent_popularity = idata.posterior["latent_popularity"].mean(("chain", "draw"))
    parties = latent_popularity.coords["parties_complete"].values

    poll_errors = []
    for party in parties:
        observed = polls_train[party] / polls_train['sample_size']
        predicted = latent_popularity.sel(parties_complete=party)
        error = (observed - predicted).abs().mean()
        poll_errors.append(error)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=parties, y=poll_errors, hue=parties, palette=party_colors, legend=False)
    plt.title("Average Polling Error by Party")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("polling_accuracy.png")
    plt.close()

def plot_all_variables(idata, polls_train):
    variables = [
        "party_baseline",
        "election_party_baseline",
        "poll_bias",
        "house_effects",
        "house_election_effects",
        "party_time_effect_weighted",
        "latent_popularity",
        "noisy_popularity",
        "election_party_time_effect_weighted",
        "N_approve",
        "R"
    ]

    for var in variables:
        if var in idata.posterior:
            plot_variable(idata, var, polls_train)

def plot_variable(idata, variable, polls_train):
    data = idata.posterior[variable]
    print(f"Plotting {variable}. Dimensions: {data.dims}, Shape: {data.shape}")
    print(f"Coordinates: {data.coords}")
    print(f"Data type: {type(data)}")
    print(f"Data values shape: {data.values.shape}")
    print(f"Sample data value (first chain, first draw):")
    print(data.isel(chain=0, draw=0).values)
    
    if "parties_complete" in data.dims:
        if "observations" in data.dims or variable.endswith("_effect"):
            plot_time_effect_variable(data, variable, polls_train)
        elif "elections" in data.dims:
            plot_party_election_variable(data, variable)
        else:
            plot_party_variable(data, variable)
    elif "pollster" in data.dims:
        plot_pollster_variable(data, variable)
    elif "elections" in data.dims:
        plot_election_variable(data, variable)
    else:
        plot_general_variable(data, variable)

def plot_time_effect_variable(data, variable, polls_train):
    mean_data = data.mean(("chain", "draw"))
    parties = mean_data.coords["parties_complete"].values
    
    # Convert 'observations' to actual dates
    dates = polls_train['date'].values
    
    fig, axes = plt.subplots(len(parties), 1, figsize=(12, 4*len(parties)), sharex=True)
    fig.suptitle(f"{variable.replace('_', ' ').title()} by Party over Time")

    for i, party in enumerate(parties):
        ax = axes[i] if len(parties) > 1 else axes
        party_data = mean_data.sel(parties_complete=party)
        
        ax.plot(dates, party_data, color=party_colors[party], label=party)
        ax.set_title(f"{party}")
        ax.set_ylabel("Effect")
        ax.legend()

    axes[-1].set_xlabel("Date")
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    plt.tight_layout()
    plt.savefig(f"{variable}_by_party_over_time.png", bbox_inches='tight')
    plt.close()

def plot_party_variable(data, variable):
    mean_data = data.mean(("chain", "draw"))
    parties = mean_data.coords["parties_complete"].values

    # Check if mean_data has more than one dimension
    if mean_data.ndim > 1:
        # If it has multiple dimensions, we'll plot each dimension separately
        other_dims = [dim for dim in mean_data.dims if dim != "parties_complete"]
        
        for dim in other_dims:
            plt.figure(figsize=(12, 8))
            for i, value in enumerate(mean_data[dim]):
                plt.bar(parties, mean_data.isel({dim: i}), 
                        color=[party_colors[party] for party in parties], 
                        alpha=0.7, label=f"{dim}={value.item()}")
            
            plt.title(f"{variable.replace('_', ' ').title()} by Party and {dim}")
            plt.xlabel("Party")
            plt.ylabel("Value")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{variable}_by_party_and_{dim}.png", bbox_inches='tight')
            plt.close()
    else:
        # If it's 1-dimensional, plot as before
        plt.figure(figsize=(12, 8))
        plt.bar(parties, mean_data, color=[party_colors[party] for party in parties])
        plt.title(f"{variable.replace('_', ' ').title()} by Party")
        plt.xlabel("Party")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{variable}_by_party.png", bbox_inches='tight')
        plt.close()

def plot_party_election_variable(data, variable):
    mean_data = data.mean(("chain", "draw"))
    parties = mean_data.coords["parties_complete"].values
    elections = mean_data.coords["elections"].values

    fig, axes = plt.subplots(len(elections), 1, figsize=(12, 6*len(elections)), sharex=True)
    fig.suptitle(f"{variable.replace('_', ' ').title()} by Party and Election")

    for i, election in enumerate(elections):
        election_data = mean_data.sel(elections=election)
        
        # Ensure axes is always treated as an array
        ax = axes[i] if len(elections) > 1 else axes

        # Create x-axis positions for the bars
        x = np.arange(len(parties))
        
        # Plot bars for each party
        for j, party in enumerate(parties):
            party_data = election_data.sel(parties_complete=party)
            if party_data.size > 1:
                # If there are multiple values, plot them as separate bars
                for k, value in enumerate(party_data.values.flatten()):
                    ax.bar(x[j] + k*0.1, value, width=0.1, color=party_colors[party], alpha=0.7)
            else:
                # If there's only one value, plot it as a single bar
                ax.bar(x[j], party_data.item(), color=party_colors[party])
        
        ax.set_title(f"Election: {election}")
        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(parties, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"{variable}_by_party_and_election.png", bbox_inches='tight')
    plt.close()

def plot_pollster_variable(data, variable):
    mean_data = data.mean(("chain", "draw"))
    pollsters = mean_data.coords["pollster"].values

    if "parties_complete" in mean_data.dims:
        parties = mean_data.coords["parties_complete"].values
        fig, axes = plt.subplots(len(parties), 1, figsize=(12, 6*len(parties)), sharex=True)
        fig.suptitle(f"{variable.replace('_', ' ').title()} by Pollster and Party")

        for i, party in enumerate(parties):
            party_data = mean_data.sel(parties_complete=party)
            axes[i].bar(pollsters, party_data)
            axes[i].set_title(f"Party: {party}")
            axes[i].set_ylabel("Value")
            axes[i].tick_params(axis='x', rotation=90)

        plt.tight_layout()
        plt.savefig(f"{variable}_by_pollster_and_party.png", bbox_inches='tight')
    else:
        plt.figure(figsize=(12, 8))
        plt.bar(pollsters, mean_data)
        plt.title(f"{variable.replace('_', ' ').title()} by Pollster")
        plt.xlabel("Pollster")
        plt.ylabel("Value")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{variable}_by_pollster.png", bbox_inches='tight')
    plt.close()

def plot_election_variable(data, variable):
    mean_data = data.mean(("chain", "draw"))
    elections = mean_data.coords["elections"].values

    plt.figure(figsize=(12, 8))
    plt.bar(elections, mean_data)
    plt.title(f"{variable.replace('_', ' ').title()} by Election")
    plt.xlabel("Election")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{variable}_by_election.png", bbox_inches='tight')
    plt.close()

def plot_general_variable(data, variable):
    mean_data = data.mean(("chain", "draw"))

    plt.figure(figsize=(10, 6))
    if mean_data.ndim == 0:
        plt.bar(variable, mean_data.values)
    else:
        sns.histplot(mean_data.values.flatten(), kde=True)
    plt.title(f"Distribution of {variable.replace('_', ' ').title()}")
    plt.xlabel("Value")
    plt.ylabel("Frequency" if mean_data.ndim > 0 else "Value")
    plt.tight_layout()
    plt.savefig(f"{variable}_distribution.png", bbox_inches='tight')
    plt.close()

def main():
    # Load data from zarr files
    posterior = az.from_zarr("posterior.zarr")
    prior = az.from_zarr("prior.zarr")
    trace = az.from_zarr("trace.zarr")

    model = ElectionModel('2024-03-10')
    # Extract necessary data from idata
    polls_train = model.polls_train
    polls_test = model.polls_test
    results_mult = model.results_mult
    election_dates = model.election_dates
    government_parties = model.government_parties

    run_diagnostics(trace, prior, posterior, model, polls_train, polls_test, results_mult, election_dates, government_parties)
    requests.post("https://ntfy.sh/bc-estimador",
        data="Finished analysis".encode(encoding='utf-8'))

if __name__ == "__main__":
    main()