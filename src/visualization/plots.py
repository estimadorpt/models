import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from typing import List
import matplotlib.dates as mdates
import os


def set_plot_style():
    """Set consistent plot style for the application"""
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette(as_cmap=True)
    return colors


def retrodictive_plot(
    trace: arviz.InferenceData,
    posterior,
    parties_complete: List[str],
    polls_train: pd.DataFrame,
    group: str = "posterior",
):
    """
    Create a retrodictive plot comparing model predictions with historical poll data.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        The inference data containing model traces
    posterior : arviz.InferenceData
        The posterior samples
    parties_complete : List[str]
        List of political parties
    polls_train : pd.DataFrame
        The training poll data
    group : str
        Whether to use posterior or prior. Options: "posterior", "prior"
    """
    colors = set_plot_style()
    
    if len(parties_complete) % 2 == 0:
        fig, axes = plt.subplots(
            len(parties_complete) // 2, 2, figsize=(12, 15), sharey=True
        )
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(
            len(parties_complete) // 2 + 1, 2, figsize=(12, 15), sharey=True
        )
        axes = axes.ravel()
        axes[-1].remove()

    N = trace.constant_data["observed_N"]
    if group == "posterior":
        pp = posterior.posterior_predictive
        POST_MEDIANS = pp["latent_popularity"].median(("chain", "draw"))
        STACKED_POP = pp["latent_popularity"].stack(sample=("chain", "draw"))

    elif group == "prior":
        prior = trace.prior
        pp = trace.prior_predictive
        POST_MEDIANS = prior["latent_popularity"].median(("chain", "draw"))
        STACKED_POP = prior["latent_popularity"].stack(sample=("chain", "draw"))

    POST_MEDIANS_MULT = (pp["N_approve"] / N).median(("chain", "draw"))
    HDI = arviz.hdi(pp)["N_approve"] / N
    SAMPLES = np.random.choice(range(len(STACKED_POP.sample)), size=1000)

    for i, p in enumerate(parties_complete):
        if group == "posterior":
            axes[i].plot(
                polls_train["date"],
                polls_train[p] / N,
                "o",
                color=colors[i],
                alpha=0.4,
            )
        for sample in SAMPLES:
            axes[i].plot(
                polls_train["date"],
                STACKED_POP.sel(parties_complete=p).isel(sample=sample),
                color=colors[i],
                alpha=0.05,
            )
        axes[i].fill_between(
            polls_train["date"],
            HDI.sel(parties_complete=p, hdi="lower"),
            HDI.sel(parties_complete=p, hdi="higher"),
            color=colors[i],
            alpha=0.4,
        )
        axes[i].plot(
            polls_train["date"],
            POST_MEDIANS_MULT.sel(parties_complete=p),
            color="black",
            ls="--",
            lw=3,
            label="Noisy Popularity",
        )
        axes[i].plot(
            polls_train["date"],
            POST_MEDIANS.sel(parties_complete=p),
            color="grey",
            lw=3,
            label="Latent Popularity",
        )
        axes[i].tick_params(axis="x", labelrotation=45, labelsize=10)
        axes[i].set(title=p.title())
        axes[i].legend(fontsize=9, ncol=2)
        
    plt.suptitle(f"{group.title()} Predictive Check", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def predictive_plot(
    idata: arviz.InferenceData,
    parties_complete: List[str],
    election_date: str,
    polls_train: pd.DataFrame,
    polls_test: pd.DataFrame,
    results: pd.DataFrame = None,
    hdi: bool = False,
):
    """
    Create a predictive plot for future election results
    
    Parameters:
    -----------
    idata : arviz.InferenceData
        The inference data for predictions
    parties_complete : List[str]
        List of political parties
    election_date : str
        Target election date
    polls_train : pd.DataFrame
        Training poll data
    polls_test : pd.DataFrame 
        Test poll data
    results : pd.DataFrame
        Actual election results, if available
    hdi : bool
        Whether to display highest density intervals
    """
    colors = set_plot_style()
    
    election_date = pd.to_datetime(election_date)
    
    # First get the predictions data
    try:
        predictions = idata.posterior_predictive
    except Exception as e:
        print(f"Error accessing posterior_predictive: {e}")
        return plt.figure()  # Return empty figure
    
    # Try to get observations from various places
    try:
        if hasattr(idata, 'constant_data') and 'observations' in idata.constant_data.data_vars:
            # Get from constant_data
            new_dates = idata.constant_data["observations"].values
        elif 'observations' in predictions.coords:
            # Get from prediction coords
            new_dates = predictions.coords["observations"].values
        else:
            # Default to election date - 100 days
            print("Warning: No observations found, using default dates")
            start_date = election_date - pd.Timedelta("100d")
            new_dates = pd.date_range(start=start_date, periods=100, freq="D")
    except Exception as e:
        print(f"Error getting observations: {e}")
        # Default to election date - 100 days
        start_date = election_date - pd.Timedelta("100d")
        new_dates = pd.date_range(start=start_date, periods=100, freq="D")
    
    # Try to filter to just election year
    try:
        # Convert to pandas DatetimeIndex if not already
        if not isinstance(new_dates, pd.DatetimeIndex):
            new_dates = pd.DatetimeIndex(new_dates)
            
        election_year_dates = new_dates[new_dates.year == election_date.year]
        
        if len(election_year_dates) > 0:
            # Filter predictions to just this year if possible
            try:
                predictions = predictions.sel(
                    observations=predictions.observations.isin(election_year_dates)
                )
            except:
                print("Could not filter predictions to election year")
        else:
            print("No dates found in election year, using all dates")
    except Exception as e:
        print(f"Error filtering to election year: {e}")
        # Just use all predictions

    if len(parties_complete) % 2 == 0:
        fig, axes = plt.subplots(len(parties_complete) // 2, 2, figsize=(12, 15))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(len(parties_complete) // 2 + 1, 2, figsize=(12, 15))
        axes = axes.ravel()
        try:
            axes[-1].remove()
        except:
            pass  # If removal fails, just continue

    # Try to compute statistics for the prediction
    try:
        POST_MEDIANS = predictions["latent_popularity"].median(("chain", "draw"))
        STACKED_POP = predictions["latent_popularity"].stack(sample=("chain", "draw"))
        HDI_POP_83 = arviz.hdi(predictions, hdi_prob=0.83)["latent_popularity"]
        
        try:
            sample_size = min(1000, len(STACKED_POP.sample))
            SAMPLES = np.random.choice(range(len(STACKED_POP.sample)), size=sample_size)
        except:
            # Default to a small number of samples
            SAMPLES = range(10) 
            
        POST_MEDIANS_MULT = predictions["noisy_popularity"].median(("chain", "draw"))
    except Exception as e:
        print(f"Error computing prediction statistics: {e}")
        # Return the figure with just the axes set up
        for i, p in enumerate(parties_complete):
            if i < len(axes):
                axes[i].set(title=p, ylim=(-0.01, 0.4))
                axes[i].tick_params(axis="x", labelrotation=45, labelsize=10)
                
        plt.tight_layout()
        return fig

    # Plot for each party
    for i, p in enumerate(parties_complete):
        if i >= len(axes):
            continue  # Skip if we have more parties than axes
            
        try:
            if hdi:
                axes[i].fill_between(
                    predictions.coords["observations"],
                    HDI_POP_83.sel(parties_complete=p, hdi="lower"),
                    HDI_POP_83.sel(parties_complete=p, hdi="higher"),
                    color=colors[i],
                    alpha=0.5,
                    label="5 in 6 chance",
                )
            else:
                for sample in SAMPLES:
                    try:
                        axes[i].plot(
                            predictions.coords["observations"],
                            STACKED_POP.sel(parties_complete=p).isel(sample=sample),
                            color=colors[i],
                            alpha=0.05,
                        )
                    except Exception as e:
                        print(f"Error plotting sample {sample} for party {p}: {e}")
                    
            axes[i].plot(
                predictions.coords["observations"],
                POST_MEDIANS.sel(parties_complete=p),
                lw=3,
                color="black",
                label="Latent Popularity",
            )
            axes[i].plot(
                predictions.coords["observations"],
                POST_MEDIANS_MULT.sel(parties_complete=p),
                ls="--",
                color="grey",
                label="Noisy Popularity",
            )
        except Exception as e:
            print(f"Error plotting predictions for party {p}: {e}")
            
        try:
            axes[i].plot(
                polls_train["date"],
                polls_train[p] / polls_train["sample_size"],
                "o",
                color="black",
                alpha=0.4,
                label="Observed polls",
            )
        except Exception as e:
            print(f"Error plotting train polls for party {p}: {e}")
        
        try:
            if polls_test is not None and not polls_test.empty:
                axes[i].plot(
                    polls_test["date"],
                    polls_test[p] / polls_test["sample_size"],
                    "x",
                    color="black",
                    alpha=0.4,
                    label="Unobserved polls",
                )
        except Exception as e:
            print(f"Error plotting test polls for party {p}: {e}")
            
        try:
            axes[i].axvline(
                x=election_date,
                ymin=-0.01,
                ymax=1.0,
                ls=":",
                c="k",
                alpha=0.6,
                label="Election Day",
            )
        except Exception as e:
            print(f"Error plotting election date for party {p}: {e}")
        
        # Plot historical average if possible
        try:
            axes[i].axhline(
                y=arviz.apply_ufunc(softmax, predictions["party_baseline"].mean(("chain", "draw"))).sel(
                    parties_complete=p
                ),
                xmin=-0.01,
                xmax=1.0,
                ls="-.",
                c=colors[i],
                label="Historical Average",
            )
        except Exception as e:
            print(f"Error plotting historical average for party {p}: {e}")
            
        axes[i].tick_params(axis="x", labelrotation=45, labelsize=10)
        axes[i].set(title=p, ylim=(-0.01, 0.4))
        try:
            axes[i].legend(fontsize=9, ncol=3)
        except:
            pass
        
    plt.tight_layout()
    return fig


def plot_house_effects(idata, pollster, parties_complete):
    """
    Plot house effects for a specific pollster
    
    Parameters:
    -----------
    idata : arviz.InferenceData
        The inference data containing house effects
    pollster : str
        The pollster to plot
    parties_complete : List[str]
        List of political parties
    """
    colors = set_plot_style()
    
    # Extract house effects for the specific pollster
    pollster_idx = np.where(idata.posterior.pollsters.values == pollster)[0][0]
    house_effects = idata.posterior.house_effects[:, :, pollster_idx, :]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, party in enumerate(parties_complete):
        party_idx = np.where(idata.posterior.parties_complete.values == party)[0][0]
        party_effects = house_effects[:, :, party_idx].values.flatten()
        
        # Plot density
        sns.kdeplot(party_effects, label=party, color=colors[i], ax=ax)
        
        # Plot mean and credible interval
        mean_effect = np.mean(party_effects)
        hdi_effect = arviz.hdi(party_effects)
        ax.axvline(mean_effect, color=colors[i], linestyle='--', alpha=0.7)
        ax.axvspan(hdi_effect['lower'], hdi_effect['higher'], color=colors[i], alpha=0.2)
    
    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_title(f'House Effects for {pollster}')
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Density')
    ax.legend()
    
    return fig


def plot_party_correlations(idata, parties_complete):
    """
    Plot correlations between party vote shares
    
    Parameters:
    -----------
    idata : arviz.InferenceData
        The inference data
    parties_complete : List[str]
        List of political parties
    """
    colors = set_plot_style()
    
    # Extract latent popularity samples
    latent_pop = idata.posterior_predictive.latent_popularity
    latent_pop_flat = latent_pop.stack(sample=("chain", "draw")).transpose("sample", "observations", "parties_complete")
    
    # Calculate correlation matrix
    corr_matrix = np.zeros((len(parties_complete), len(parties_complete)))
    
    for i, party1 in enumerate(parties_complete):
        for j, party2 in enumerate(parties_complete):
            if i <= j:  # Only compute upper triangle (including diagonal)
                party1_data = latent_pop_flat.sel(parties_complete=party1).values
                party2_data = latent_pop_flat.sel(parties_complete=party2).values
                
                # Calculate correlation across all samples and observations
                corr = np.corrcoef(party1_data.flatten(), party2_data.flatten())[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Mirror for the lower triangle
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        xticklabels=parties_complete,
        yticklabels=parties_complete,
        vmin=-1, 
        vmax=1, 
        center=0,
        ax=ax
    )
    ax.set_title("Correlations Between Party Vote Shares")
    
    return fig


def plot_predictive_accuracy(idata, observed_polls, parties_complete):
    """
    Plot predictive accuracy of the model
    
    Parameters:
    -----------
    idata : arviz.InferenceData
        The inference data
    observed_polls : pd.DataFrame
        Observed poll data
    parties_complete : List[str]
        List of political parties
    """
    colors = set_plot_style()
    
    # Prepare data
    predicted = idata.posterior_predictive.noisy_popularity.mean(("chain", "draw")).values
    actual = np.zeros_like(predicted)
    
    for i, party in enumerate(parties_complete):
        actual[:, i] = observed_polls[party].values / observed_polls['sample_size'].values
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, party in enumerate(parties_complete):
        ax.scatter(
            actual[:, i], 
            predicted[:, i], 
            label=party,
            color=colors[i],
            alpha=0.7,
            edgecolors='w',
            s=50
        )
    
    # Add 45-degree line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    
    ax.set_aspect('equal')
    ax.set_xlabel('Observed Vote Share')
    ax.set_ylabel('Predicted Vote Share')
    ax.set_title('Model Predictive Accuracy')
    ax.legend()
    
    return fig


def plot_latent_trajectories(
    prediction_data,
    coords=None,
    dims=None,
    polls_train=None,
    polls_test=None,
    election_date=None
):
    """
    Plot latent trajectories for all parties.
    
    Parameters:
    -----------
    prediction_data : dict or InferenceData
        The prediction data, either as a raw dictionary or InferenceData object
    coords : dict, optional
        Coordinates for the prediction data if using raw dictionary
    dims : dict, optional
        Dimensions for the prediction data if using raw dictionary
    polls_train : pd.DataFrame, optional
        Training poll data
    polls_test : pd.DataFrame, optional
        Test poll data
    election_date : str, optional
        The target election date
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import arviz as az
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    try:
        # Handle both raw dictionary and InferenceData formats
        if isinstance(prediction_data, az.InferenceData):
            if hasattr(prediction_data, 'posterior_predictive'):
                post_pred = prediction_data.posterior_predictive
            else:
                raise ValueError("InferenceData object does not have posterior_predictive group")
        else:
            # Use raw dictionary data
            post_pred = prediction_data
            
            # Create a simple namespace object that mimics InferenceData structure
            class SimpleNamespace:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
            
            # Create a simple namespace object with the data
            post_pred = SimpleNamespace(
                data_vars=prediction_data.keys(),
                observations=pd.to_datetime(coords.get('observations', [])),
            )
            
            # Add the prediction arrays as attributes
            for key, value in prediction_data.items():
                setattr(post_pred, key, value)
        
        # Get the latent popularity data
        if hasattr(post_pred, 'latent_popularity'):
            latent_pop = post_pred.latent_popularity
        else:
            raise ValueError("No latent_popularity found in prediction data")
        
        # Get dates and parties
        if hasattr(post_pred, 'observations'):
            dates = pd.to_datetime(post_pred.observations)
        else:
            dates = pd.to_datetime(coords['observations'])
        
        # Make sure parties is always a list
        parties = coords['parties_complete'] if coords else post_pred.parties_complete
        if not isinstance(parties, list):
            parties = list(parties)
        
        # Calculate mean and credible intervals
        if isinstance(latent_pop, np.ndarray):
            mean_pop = np.mean(latent_pop, axis=(0, 1))  # Average over chains and draws
            lower_bound = np.percentile(latent_pop, 2.5, axis=(0, 1))
            upper_bound = np.percentile(latent_pop, 97.5, axis=(0, 1))
        else:
            # Handle xarray DataArray
            mean_pop = latent_pop.mean(("chain", "draw")).values
            lower_bound = latent_pop.quantile(0.025, dim=("chain", "draw")).values
            upper_bound = latent_pop.quantile(0.975, dim=("chain", "draw")).values
        
        # Plot for each party
        for i, party in enumerate(parties):
            ax.plot(dates, mean_pop[:, i], label=party)
            ax.fill_between(dates, lower_bound[:, i], upper_bound[:, i], alpha=0.2)
        
        # Add historical polls if available
        if polls_train is not None:
            for party in parties:
                if party in polls_train.columns:
                    ax.scatter(polls_train['date'], 
                             polls_train[party] / polls_train['sample_size'],
                             alpha=0.3, marker='o', s=20)
        
        # Add test polls if available
        if polls_test is not None:
            for party in parties:
                if party in polls_test.columns:
                    ax.scatter(polls_test['date'],
                             polls_test[party] / polls_test['sample_size'],
                             alpha=0.5, marker='x', s=30, color='red')
        
        # Add election date line
        if election_date:
            election_date = pd.to_datetime(election_date)
            ax.axvline(x=election_date, color='k', linestyle='--', alpha=0.5)
            ax.text(election_date, ax.get_ylim()[1], 'Election',
                   rotation=90, verticalalignment='top')
        
        # Customize the plot
        ax.set_title('Latent Party Trajectories')
        ax.set_xlabel('Date')
        ax.set_ylabel('Vote Share')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to fit the legend
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_latent_trajectories: {e}")
        import traceback
        traceback.print_exc()
        raise


def plot_party_trajectory(
    prediction_data,
    coords=None,
    dims=None,
    party=None,
    polls_train=None,
    polls_test=None,
    election_date=None
):
    """
    Plot trajectory for a specific party.
    
    Parameters:
    -----------
    prediction_data : dict or InferenceData
        The prediction data, either as a raw dictionary or InferenceData object
    coords : dict, optional
        Coordinates for the prediction data if using raw dictionary
    dims : dict, optional
        Dimensions for the prediction data if using raw dictionary
    party : str
        The party to plot
    polls_train : pd.DataFrame, optional
        Training poll data
    polls_test : pd.DataFrame, optional
        Test poll data
    election_date : str, optional
        The target election date
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import arviz as az
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    try:
        # Handle both raw dictionary and InferenceData formats
        if isinstance(prediction_data, az.InferenceData):
            if hasattr(prediction_data, 'posterior_predictive'):
                post_pred = prediction_data.posterior_predictive
            else:
                raise ValueError("InferenceData object does not have posterior_predictive group")
        else:
            # Use raw dictionary data
            post_pred = prediction_data
            
            # Create a simple namespace object that mimics InferenceData structure
            class SimpleNamespace:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
            
            # Create a simple namespace object with the data
            post_pred = SimpleNamespace(
                data_vars=prediction_data.keys(),
                observations=pd.to_datetime(coords.get('observations', [])),
            )
            
            # Add the prediction arrays as attributes
            for key, value in prediction_data.items():
                setattr(post_pred, key, value)
        
        # Get the latent popularity data
        if hasattr(post_pred, 'latent_popularity'):
            latent_pop = post_pred.latent_popularity
        else:
            raise ValueError("No latent_popularity found in prediction data")
        
        # Get dates and parties
        if hasattr(post_pred, 'observations'):
            dates = pd.to_datetime(post_pred.observations)
        else:
            dates = pd.to_datetime(coords['observations'])
        
        # Make sure parties is always a list
        parties = coords['parties_complete'] if coords else post_pred.parties_complete
        if not isinstance(parties, list):
            parties = list(parties)
        
        # Find party index - use list index method for safety
        if party in parties:
            party_idx = parties.index(party)
        else:
            raise ValueError(f"Party '{party}' not found in the prediction data. Available parties: {parties}")
        
        # Calculate mean and credible intervals
        if isinstance(latent_pop, np.ndarray):
            mean_pop = np.mean(latent_pop, axis=(0, 1))  # Average over chains and draws
            lower_bound = np.percentile(latent_pop, 2.5, axis=(0, 1))
            upper_bound = np.percentile(latent_pop, 97.5, axis=(0, 1))
        else:
            # Handle xarray DataArray
            mean_pop = latent_pop.mean(("chain", "draw")).values
            lower_bound = latent_pop.quantile(0.025, dim=("chain", "draw")).values
            upper_bound = latent_pop.quantile(0.975, dim=("chain", "draw")).values
        
        # Plot the trajectory
        ax.plot(dates, mean_pop[:, party_idx], label='Mean')
        ax.fill_between(dates, lower_bound[:, party_idx], upper_bound[:, party_idx],
                       alpha=0.2, label='95% Credible Interval')
        
        # Add historical polls if available
        if polls_train is not None and party in polls_train.columns:
            ax.scatter(polls_train['date'],
                      polls_train[party] / polls_train['sample_size'],
                      alpha=0.3, marker='o', s=20, label='Historical Polls')
        
        # Add test polls if available
        if polls_test is not None and party in polls_test.columns:
            ax.scatter(polls_test['date'],
                      polls_test[party] / polls_test['sample_size'],
                      alpha=0.5, marker='x', s=30, color='red', label='Test Polls')
        
        # Add election date line
        if election_date:
            election_date = pd.to_datetime(election_date)
            ax.axvline(x=election_date, color='k', linestyle='--', alpha=0.5)
            ax.text(election_date, ax.get_ylim()[1], 'Election',
                   rotation=90, verticalalignment='top')
        
        # Customize the plot
        ax.set_title(f'Trajectory for {party}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Vote Share')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to fit the legend
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_party_trajectory: {e}")
        import traceback
        traceback.print_exc()
        raise 

def save_plots(elections_model, output_dir):
    """
    Save various plots from the model
    
    Parameters:
    -----------
    elections_model : ElectionsFacade
        The elections model to generate plots from
    output_dir : str
        Directory to save plots in
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    plot_functions = [
        ("retrodictive_check", lambda: elections_model.plot_retrodictive_check(), "latent_popularity_evolution_with_observed_and_results.png"),
        ("forecast", lambda: elections_model.plot_forecast(), None),  # Special handling for forecast plots
        ("party_correlations", lambda: elections_model.plot_party_correlations(), "party_correlations.png"),
        ("predictive_accuracy", lambda: elections_model.plot_predictive_accuracy(), "polling_accuracy.png")
    ]
    
    for name, plot_func, filename in plot_functions:
        try:
            result = plot_func()
            
            # Special handling for forecast plots which can return a list
            if name == "forecast":
                if isinstance(result, list):
                    for i, fig in enumerate(result):
                        fig.savefig(os.path.join(plots_dir, f"forecast_plot_{i}.png"))
                        plt.close(fig)
                else:
                    result.savefig(os.path.join(plots_dir, "latent_popularity_evolution_last_year.png"))
                    plt.close(result)
            else:
                result.savefig(os.path.join(plots_dir, filename))
                plt.close(result)
                
        except Exception as e:
            print(f"Error saving {name} plot: {e}")
    
    # Save house effects for each pollster
    house_effects_dir = os.path.join(plots_dir, "house_effects")
    if not os.path.exists(house_effects_dir):
        os.makedirs(house_effects_dir)
        
    for pollster in elections_model.dataset.unique_pollsters:
        try:
            house_fig = elections_model.plot_house_effects(pollster)
            house_fig.savefig(os.path.join(house_effects_dir, f"house_effects_{pollster.replace('/', '_')}.png"))
            plt.close(house_fig)
        except Exception as e:
            print(f"Error plotting house effects for {pollster}: {e}")
    
    # Plot individual model components
    components_dir = os.path.join(plots_dir, "components")
    if not os.path.exists(components_dir):
        os.makedirs(components_dir)
        
    try:
        plot_model_components(elections_model, components_dir)
    except Exception as e:
        print(f"Error plotting model components: {e}")


def plot_model_components(elections_model, output_dir):
    """
    Plot various model components
    
    Parameters:
    -----------
    elections_model : ElectionsFacade
        The elections model to generate plots from
    output_dir : str
        Directory to save plots in
    """
    trace = elections_model.trace
    
    component_plots = [
        # Component name, data extraction function, plot function, filename
        ("party_baseline", 
         lambda: trace.posterior.party_baseline.mean(("chain", "draw")),
         lambda df: plt.bar(df["parties_complete"], df["value"]),
         "party_baseline_by_party.png"),
         
        ("election_party_baseline", 
         lambda: trace.posterior.election_party_baseline.mean(("chain", "draw")),
         lambda df: sns.lineplot(data=df, x="elections", y="value", hue="parties_complete", marker="o"),
         "election_party_baseline_by_party_and_election.png"),
         
        ("poll_bias", 
         lambda: trace.posterior.poll_bias.mean(("chain", "draw")),
         lambda df: plt.bar(df["parties_complete"], df["value"]),
         "poll_bias_by_party.png"),
         
        ("party_time_effect_weighted", 
         lambda: trace.posterior.party_time_effect_weighted.mean(("chain", "draw")),
         lambda df: sns.lineplot(data=df, x="countdown", y="value", hue="parties_complete"),
         "party_time_effect_weighted_by_party_and_countdown.png"),
         
        ("latent_popularity", 
         lambda: trace.posterior.latent_popularity.mean(("chain", "draw")),
         lambda df: sns.lineplot(data=df, x="observations", y="value", hue="parties_complete"),
         "latent_popularity_by_party_over_time.png"),
         
        ("noisy_popularity", 
         lambda: trace.posterior.noisy_popularity.mean(("chain", "draw")),
         lambda df: sns.lineplot(data=df, x="observations", y="value", hue="parties_complete"),
         "noisy_popularity_by_party_over_time.png"),
    ]
    
    for name, data_func, plot_func, filename in component_plots:
        try:
            # Extract data
            data = data_func()
            
            # Convert to dataframe for plotting
            df = data.to_dataframe(name="value").reset_index()
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plot_func(df)
            plt.title(f"{name.replace('_', ' ').title()}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
        except Exception as e:
            print(f"Error plotting {name}: {e}")
    
    # Handle special cases with more complex plotting requirements
    try:
        # House effects heatmap
        house_effects = trace.posterior.house_effects.mean(("chain", "draw"))
        df = house_effects.to_dataframe(name="value").reset_index()
        pivot_df = df.pivot(index="pollsters", columns="parties_complete", values="value")
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0)
        plt.title("House Effects by Party and Pollsters")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "house_effects_by_party_and_pollsters.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting house effects heatmap: {e}")
    
    try:
        # House election effects - multiple plots
        house_election_effects = trace.posterior.house_election_effects.mean(("chain", "draw"))
        df = house_election_effects.to_dataframe(name="value").reset_index()
        
        # Group by election and create separate plots
        for election in df["elections"].unique():
            election_df = df[df["elections"] == election]
            pivot_df = election_df.pivot(index="pollsters", columns="parties_complete", values="value")
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0)
            plt.title(f"House Election Effects - Election {election}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"house_election_effects_election_{election}.png"))
            plt.close()
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x="elections", y="value", hue="parties_complete")
        plt.title("House Election Effects by Party and Election")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "house_election_effects_by_party_and_election.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting house election effects: {e}")
    
    try:
        # Election party time effect weighted
        election_party_time_effect_weighted = trace.posterior.election_party_time_effect_weighted.mean(("chain", "draw"))
        eptew_at_zero = election_party_time_effect_weighted.isel(countdown=0)
        df = eptew_at_zero.to_dataframe(name="value").reset_index()
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x="elections", y="value", hue="parties_complete", marker="o")
        plt.title("Election Party Time Effect Weighted by Party and Election (Countdown=0)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "election_party_time_effect_weighted_by_party_and_election.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting election party time effect weighted: {e}") 