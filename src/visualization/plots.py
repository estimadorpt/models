import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from typing import List
import matplotlib.dates as mdates


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


def plot_latent_trajectories(idata, dates=None, parties=None, polls_train=None, polls_test=None, election_date=None):
    """
    Plot the evolution of latent trajectories for multiple parties.
    
    Parameters:
    -----------
    idata : arviz.InferenceData
        The inference data object containing the posterior samples
    dates : array-like, optional
        The dates corresponding to the samples
    parties : list, optional
        The list of party names to plot
    polls_train : pandas.DataFrame, optional
        The training polls data to overlay on the plot
    polls_test : pandas.DataFrame, optional
        The test polls data to overlay on the plot
    election_date : str, optional
        The election date to mark with a vertical line
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure with the latent trajectories
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Define party colors
    party_colors = {
        "Socialist Party": "#E4001C",  # Red
        "Labour Party": "#BB0022",  # Dark red
        "Social Democratic Party": "#CC0000",  # Another shade of red
        "People's Party": "#0047AB",  # Blue
        "Liberal Party": "#5F9EA0",  # Cadet blue
        "Conservative Party": "#0C2577",  # Dark blue
        "Christian Democratic Union": "#000000",  # Black
        "Green Party": "#009E60",  # Green
        "Left Party": "#800080",  # Purple
        "National Rally": "#8B4513",  # Brown
        "Freedom Party": "#8B4513",  # Brown
        "Others": "#808080"  # Gray
    }
    
    # Get the posterior predictive samples
    posterior_pred = idata.posterior_predictive
    
    if "latent_popularity" not in posterior_pred:
        raise ValueError("No latent_popularity variable in posterior_predictive")
    
    # Get the latent popularity samples
    latent_pop = posterior_pred["latent_popularity"].values
    
    # Use provided dates or create default dates
    if dates is None:
        try:
            dates = posterior_pred.coords["observations"].values
        except (KeyError, AttributeError):
            # Create default dates based on the number of observations
            n_obs = latent_pop.shape[2]
            if election_date is not None:
                end_date = pd.to_datetime(election_date)
                start_date = end_date - pd.Timedelta(days=100)
                dates = pd.date_range(start=start_date, end=end_date, freq="D")
            else:
                dates = pd.date_range(end=pd.Timestamp.now(), periods=n_obs, freq="D")
    
    # Use provided party names or create default names
    if parties is None:
        try:
            parties = posterior_pred.coords["parties_complete"].values
        except (KeyError, AttributeError):
            # Create default party names
            n_parties = latent_pop.shape[3]
            parties = [f"Party {i+1}" for i in range(n_parties)]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get a colormap with distinct colors
    cmap = plt.cm.get_cmap('tab10', len(parties))
    
    # Calculate mean trajectories
    mean_latent = latent_pop.mean(axis=(0, 1))
    
    # Plot sample trajectories for uncertainty visualization (pollsposition style)
    # Reduce the number of samples to decrease visual clutter
    n_samples = min(50, latent_pop.shape[0] * latent_pop.shape[1])
    samples_per_chain = n_samples // latent_pop.shape[0]
    for i, party in enumerate(parties):
        # Reshape to get all samples
        party_samples = latent_pop[:, :, :, i].reshape(-1, latent_pop.shape[2])
        
        # Randomly select some samples to plot
        sample_indices = np.random.choice(party_samples.shape[0], size=n_samples, replace=False)
        
        # Plot sample trajectories with low alpha for uncertainty visualization
        for idx in sample_indices:
            ax.plot(dates, party_samples[idx], color=cmap(i), alpha=0.02, linewidth=1)
        
        # Plot mean trajectory on top
        ax.plot(dates, mean_latent[:, i], label=party, linewidth=2.5, color=cmap(i))
    
    # Add poll data points if available with jittering to reduce overlap
    marker_size = 30  # Reduced marker size
    if polls_train is not None:
        for i, party in enumerate(parties):
            if party in polls_train.columns:
                # Try different column names for sample size
                sample_size_col = None
                for col in ['sample_size', 'samplesize', 'N']:
                    if col in polls_train.columns:
                        sample_size_col = col
                        break
                
                if sample_size_col is None:
                    print(f"Warning: Could not find sample size column in polls_train. Available columns: {polls_train.columns.tolist()}")
                    continue
                
                # Calculate party support in polls
                party_support = polls_train[party].values / polls_train[sample_size_col].values
                
                # Group polls by date to apply jittering
                grouped_polls = polls_train.groupby('date')
                
                for j, (date, group) in enumerate(grouped_polls):
                    # Apply jitter to avoid overplotting
                    x_jitter = date + pd.Timedelta(hours=np.random.uniform(-12, 12))
                    
                    # Get the party support for this date group
                    date_idx = polls_train.index[polls_train['date'] == date].tolist()
                    if date_idx:
                        y_values = party_support[date_idx]
                        
                        # Get a list of keys from the grouped_polls.groups
                        group_keys = list(grouped_polls.groups.keys())
                        first_date = group_keys[0] if group_keys else None
                        
                        # Plot the polls with small markers
                        ax.scatter(
                            [x_jitter] * len(y_values), 
                            y_values,
                            color=party_colors.get(party, f"C{i}"),
                            s=15,  # Small marker size
                            alpha=0.5,
                            marker="o",
                            label=f"{party} polls (train)" if date == first_date else ""
                        )
    
    # Add test polls with X markers
    if polls_test is not None and len(polls_test) > 0:
        for i, party in enumerate(parties):
            if party in polls_test.columns:
                # Find sample size column
                sample_size_col = None
                for col in ['sample_size', 'samplesize', 'N']:
                    if col in polls_test.columns:
                        sample_size_col = col
                        break
                
                if sample_size_col is None:
                    continue
                
                # Calculate party support in test polls
                party_support = polls_test[party].values / polls_test[sample_size_col].values
                
                # Group polls by date to apply jittering
                grouped_polls = polls_test.groupby('date')
                
                for date, group in grouped_polls:
                    if date in dates:
                        # Get this party's support for this date's polls
                        date_polls = group[party].values / group[sample_size_col].values
                        
                        # Apply small horizontal jitter for polls on the same date
                        jitter = np.random.normal(0, 0.5, size=len(date_polls))  # in days
                        jittered_dates = [pd.Timestamp(date) + pd.Timedelta(days=j) for j in jitter]
                        
                        # Plot with X markers for test polls
                        ax.scatter(
                            jittered_dates, 
                            date_polls, 
                            s=marker_size, 
                            color=cmap(i), 
                            alpha=0.6,
                            marker='x',
                            linewidths=1.5,
                            label=f"{party} polls (test)" if date == list(grouped_polls.groups.keys())[0] else ""
                        )
    
    # Add election date vertical line
    if election_date is not None:
        ax.axvline(
            x=pd.to_datetime(election_date),
            linestyle='--',
            color='black',
            alpha=0.8,
            linewidth=1.5,
            label='Election Day'
        )
    
    # Add historical averages if party_baseline is available
    try:
        if 'party_baseline' in posterior_pred:
            party_baselines = posterior_pred["party_baseline"].mean(("chain", "draw")).values
            # Use softmax to convert to probabilities
            baseline_probs = softmax(party_baselines)
            
            for i, party in enumerate(parties):
                ax.axhline(
                    y=baseline_probs[i],
                    xmin=-0.01,
                    xmax=1.0,
                    ls="-.",
                    c=cmap(i),
                    alpha=0.5,
                    linewidth=1.5
                )
    except Exception as e:
        print(f"Could not add historical averages: {e}")
    
    # Set labels and title
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Support (%)", fontsize=12)
    ax.set_title("Forecasted Party Support", fontsize=16, fontweight='bold')
    
    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Use month-year format for cleaner labels
    
    # Use fewer date ticks to avoid overcrowding
    date_range = pd.date_range(start=min(dates), end=max(dates), freq='MS')  # Monthly ticks
    if len(date_range) > 12:  # If too many months, use quarters instead
        date_range = pd.date_range(start=min(dates), end=max(dates), freq='QS')
    ax.set_xticks(date_range)
    
    # Rotate date labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add a grid
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits with some padding, handling NaN or Inf values
    try:
        max_val = np.nanmax(mean_latent) if np.any(np.isfinite(mean_latent)) else 0.5
        if np.isfinite(max_val):
            ax.set_ylim(0, min(max(max_val * 1.2, 0.5), 1.0))
        else:
            # Default if max is not finite
            ax.set_ylim(0, 0.5)
    except (ValueError, TypeError):
        # Default if anything goes wrong
        ax.set_ylim(0, 0.5)
        
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Add a legend with neat formatting - use a separate box to avoid overlap with plot
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Clean up duplicate labels and place the legend to the right of the figure
    ax.legend(
        by_label.values(), 
        by_label.keys(), 
        loc='upper left', 
        bbox_to_anchor=(1.02, 1), 
        fontsize=10, 
        framealpha=0.7
    )
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    return fig


def plot_party_trajectory(idata, party=None, dates=None, polls_train=None, polls_test=None, election_date=None):
    """
    Plot the trajectory of a specific party's popularity over time.
    
    Parameters:
    -----------
    idata : arviz.InferenceData
        The inference data object containing the posterior samples
    party : str
        The name of the party to plot
    dates : array-like, optional
        The dates corresponding to the samples
    polls_train : pandas.DataFrame, optional
        The training polls data to overlay on the plot
    polls_test : pandas.DataFrame, optional
        The test polls data to overlay on the plot
    election_date : str, optional
        The election date to mark with a vertical line
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure with the party trajectory
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Define party colors
    party_colors = {
        "Socialist Party": "#E4001C",  # Red
        "Labour Party": "#BB0022",  # Dark red
        "Social Democratic Party": "#CC0000",  # Another shade of red
        "People's Party": "#0047AB",  # Blue
        "Liberal Party": "#5F9EA0",  # Cadet blue
        "Conservative Party": "#0C2577",  # Dark blue
        "Christian Democratic Union": "#000000",  # Black
        "Green Party": "#009E60",  # Green
        "Left Party": "#800080",  # Purple
        "National Rally": "#8B4513",  # Brown
        "Freedom Party": "#8B4513",  # Brown
        "Others": "#808080"  # Gray
    }
    
    # Get the posterior predictive samples
    posterior_pred = idata.posterior_predictive
    
    # Check if party is provided
    if party is None:
        raise ValueError("Must specify a party to plot")
    
    # Get dates from posterior predictive
    try:
        dates = posterior_pred.coords["observations"].values
    except (KeyError, AttributeError):
        raise ValueError("No observations coordinate found in posterior predictive")
    
    # Get party names
    try:
        parties = posterior_pred.coords["parties_complete"].values
    except (KeyError, AttributeError):
        raise ValueError("No parties_complete coordinate found in posterior predictive")
    
    # Find the index of the specified party
    try:
        party_index = np.where(parties == party)[0][0]
    except:
        raise ValueError(f"Party '{party}' not found in posterior predictive. Available parties: {parties}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get the latent popularity for this party
    if "latent_popularity" in posterior_pred:
        latent_pop = posterior_pred["latent_popularity"].sel(parties_complete=party).values
    else:
        raise ValueError("No latent_popularity variable found in posterior predictive")
    
    # Calculate mean and credible intervals
    mean_latent = latent_pop.mean(axis=(0, 1))
    
    # Calculate HDI interval (highest density interval)
    lower_bound = np.percentile(latent_pop, 2.5, axis=(0, 1))
    upper_bound = np.percentile(latent_pop, 97.5, axis=(0, 1))
    
    # Plot mean and credible intervals
    ax.plot(dates, mean_latent, label=f"{party} Mean", color=party_colors.get(party, f"C0"), linewidth=2)
    ax.fill_between(dates, lower_bound, upper_bound, alpha=0.2, color=party_colors.get(party, f"C0"))
    
    # Plot observed polls if available
    if polls_train is not None and party in polls_train.columns:
        # Look for the sample size column
        sample_size_col = None
        for col in ['sample_size', 'samplesize', 'N']:
            if col in polls_train.columns:
                sample_size_col = col
                break
        
        if sample_size_col is not None:
            # Calculate proportions
            party_prop_train = polls_train[party].values / polls_train[sample_size_col].values
            
            # Apply jitter to dates for better visualization
            jitter = pd.Timedelta(hours=np.random.uniform(-12, 12))
            jittered_dates = polls_train["date"].apply(lambda d: d + jitter)
            
            # Plot with small marker size for better visibility
            ax.scatter(jittered_dates, party_prop_train, color=party_colors.get(party, f"C0"), alpha=0.5, s=20, label=f"{party} Polls (Train)")
    
    # Plot test polls if available
    if polls_test is not None and party in polls_test.columns:
        # Look for the sample size column
        sample_size_col = None
        for col in ['sample_size', 'samplesize', 'N']:
            if col in polls_test.columns:
                sample_size_col = col
                break
        
        if sample_size_col is not None:
            # Calculate proportions
            party_prop_test = polls_test[party].values / polls_test[sample_size_col].values
            
            # Apply jitter to dates for better visualization
            jitter = pd.Timedelta(hours=np.random.uniform(-12, 12))
            jittered_dates = polls_test["date"].apply(lambda d: d + jitter)
            
            # Plot with small marker size for better visibility
            ax.scatter(jittered_dates, party_prop_test, color=party_colors.get(party, f"C0"), alpha=0.8, s=30, marker="x", label=f"{party} Polls (Test)")
    
    # Add election date if provided
    if election_date is not None:
        election_date = pd.to_datetime(election_date)
        ax.axvline(x=election_date, color='gray', linestyle='--', alpha=0.7, label='Election Date')
    
    # Format x-axis with dates
    plt.gcf().autofmt_xdate()
    
    # Add labels and title
    ax.set_ylabel("Popularity (%)")
    ax.set_title(f"Forecast Trajectory for {party}")
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Adjust layout to fit the legend
    plt.tight_layout()
    
    return fig 