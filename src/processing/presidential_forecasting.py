"""
Presidential Election Forecasting Processing.

This module provides functions for generating forecasts, visualizations,
and analysis outputs for Portuguese presidential election models.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import arviz as az


# Candidate colors for visualization
CANDIDATE_COLORS = {
    'Gouveia e Melo': '#4A90D9',      # Blue (independent/military)
    'Marques Mendes': '#FF8C00',       # Orange (PSD)
    'António José Seguro': '#FF69B4',  # Pink (PS)
    'André Ventura': '#8B0000',        # Dark red (CH)
    'Cotrim Figueiredo': '#00CED1',    # Cyan (IL)
    'Catarina Martins': '#DC143C',     # Crimson (BE)
    'António Filipe': '#228B22',       # Green (CDU)
    'Others': '#808080',               # Gray
}


def get_candidate_color(candidate: str) -> str:
    """Get color for a candidate, with fallback."""
    return CANDIDATE_COLORS.get(candidate, '#666666')


def plot_support_trajectory(
    idata: az.InferenceData,
    candidates: List[str],
    calendar_time: pd.DatetimeIndex,
    polls_df: Optional[pd.DataFrame] = None,
    election_date: Optional[pd.Timestamp] = None,
    title: str = "Presidential Election Forecast",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the support trajectory for all candidates.

    Args:
        idata: InferenceData with posterior samples
        candidates: List of candidate names
        calendar_time: DatetimeIndex of time points
        polls_df: Optional DataFrame with raw poll data for overlay
        election_date: Election date for vertical line
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get posterior probabilities
    probs = idata.posterior['national_probs_calendar']
    calendar_dates = pd.to_datetime(calendar_time)

    for i, candidate in enumerate(candidates):
        color = get_candidate_color(candidate)
        candidate_probs = probs.isel(candidates=i)

        # Calculate statistics
        mean = candidate_probs.mean(dim=['chain', 'draw']).values
        q05 = candidate_probs.quantile(0.05, dim=['chain', 'draw']).values
        q95 = candidate_probs.quantile(0.95, dim=['chain', 'draw']).values
        q25 = candidate_probs.quantile(0.25, dim=['chain', 'draw']).values
        q75 = candidate_probs.quantile(0.75, dim=['chain', 'draw']).values

        # Plot credible intervals
        ax.fill_between(calendar_dates, q05 * 100, q95 * 100,
                       color=color, alpha=0.15)
        ax.fill_between(calendar_dates, q25 * 100, q75 * 100,
                       color=color, alpha=0.25)
        ax.plot(calendar_dates, mean * 100, color=color, linewidth=2,
               label=f'{candidate}')

    # Overlay poll data if provided
    if polls_df is not None:
        for i, candidate in enumerate(candidates):
            if candidate in polls_df.columns:
                color = get_candidate_color(candidate)
                # Get poll proportions (if stored as counts, convert back)
                if 'sample_size' in polls_df.columns:
                    poll_props = polls_df[candidate] / polls_df['sample_size'] * 100
                else:
                    poll_props = polls_df[candidate] * 100
                ax.scatter(polls_df['date'], poll_props,
                          color=color, alpha=0.4, s=20, marker='o')

    # Add election day line
    if election_date is not None:
        ax.axvline(election_date, color='black', linestyle='--',
                  linewidth=1.5, alpha=0.7, label='Election Day')

    # Add 50% threshold line (for first round win)
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5)

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Support (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.set_ylim(0, None)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")

    return fig


def plot_forecast_bars(
    forecast_df: pd.DataFrame,
    title: str = "Election Day Forecast",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot bar chart of election day forecasts with uncertainty.

    Args:
        forecast_df: DataFrame from model.get_forecast()
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    candidates = forecast_df['candidate'].tolist()
    means = forecast_df['mean'].values * 100
    colors = [get_candidate_color(c) for c in candidates]

    # Error bars
    if 'ci_lower' in forecast_df.columns:
        lower_err = means - forecast_df['ci_lower'].values * 100
        upper_err = forecast_df['ci_upper'].values * 100 - means
        yerr = [lower_err, upper_err]
    else:
        yerr = None

    bars = ax.barh(candidates, means, color=colors, xerr=yerr,
                   capsize=3, error_kw={'elinewidth': 1.5})

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f'{mean:.1f}%', va='center', fontsize=10)

    # Add 50% line
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5,
              label='50% threshold')

    ax.set_xlabel('Support (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(means) * 1.3)
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved forecast bars to {save_path}")

    return fig


def plot_win_probabilities(
    win_probs_df: pd.DataFrame,
    title: str = "Win Probabilities",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot win probabilities for each candidate.

    Args:
        win_probs_df: DataFrame from model.get_win_probabilities()
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    candidates = win_probs_df['candidate'].tolist()
    leading_probs = win_probs_df['leading_prob'].values * 100
    colors = [get_candidate_color(c) for c in candidates]

    bars = ax.barh(candidates, leading_probs, color=colors)

    # Add value labels
    for bar, prob in zip(bars, leading_probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f'{prob:.1f}%', va='center', fontsize=10)

    ax.set_xlabel('Probability of Leading (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    # Add second round probability annotation
    second_round_prob = win_probs_df['second_round_prob'].iloc[0]
    ax.text(0.95, 0.02, f'P(Second Round): {second_round_prob:.1%}',
           transform=ax.transAxes, ha='right', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved win probabilities to {save_path}")

    return fig


def plot_house_effects(
    idata: az.InferenceData,
    candidates: List[str],
    pollsters: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot estimated house effects for each pollster-candidate pair.

    Args:
        idata: InferenceData with posterior samples
        candidates: List of candidate names
        pollsters: List of pollster names
        figsize: Figure size (auto-calculated if None)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    house_effects = idata.posterior['house_effects']

    n_pollsters = len(pollsters)
    n_candidates = len(candidates)

    if figsize is None:
        figsize = (max(10, n_candidates * 1.5), max(6, n_pollsters * 0.8))

    fig, axes = plt.subplots(n_pollsters, 1, figsize=figsize, sharex=True)
    if n_pollsters == 1:
        axes = [axes]

    for p_idx, (ax, pollster) in enumerate(zip(axes, pollsters)):
        effects = house_effects.isel(pollsters=p_idx)
        means = effects.mean(dim=['chain', 'draw']).values * 100
        q05 = effects.quantile(0.05, dim=['chain', 'draw']).values * 100
        q95 = effects.quantile(0.95, dim=['chain', 'draw']).values * 100

        colors = [get_candidate_color(c) for c in candidates]

        x = np.arange(n_candidates)
        ax.bar(x, means, color=colors, alpha=0.7)
        ax.errorbar(x, means, yerr=[means - q05, q95 - means],
                   fmt='none', color='black', capsize=3)

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_ylabel(pollster, fontsize=10)
        ax.set_ylim(-10, 10)

    axes[-1].set_xticks(np.arange(n_candidates))
    axes[-1].set_xticklabels(candidates, rotation=45, ha='right')
    axes[0].set_title('House Effects by Pollster (percentage points)', fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved house effects to {save_path}")

    return fig


def generate_forecast_report(
    forecast_df: pd.DataFrame,
    win_probs_df: pd.DataFrame,
    election_date: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a text summary of the forecast.

    Args:
        forecast_df: DataFrame from model.get_forecast()
        win_probs_df: DataFrame from model.get_win_probabilities()
        election_date: Election date string
        output_path: Path to save report (optional)

    Returns:
        Report string
    """
    lines = [
        "=" * 60,
        f"PRESIDENTIAL ELECTION FORECAST",
        f"Election Date: {election_date}",
        "=" * 60,
        "",
        "PROJECTED VOTE SHARES",
        "-" * 40,
    ]

    for _, row in forecast_df.iterrows():
        lines.append(
            f"  {row['candidate']:25s} {row['mean']*100:5.1f}% "
            f"[{row['ci_lower']*100:4.1f}% - {row['ci_upper']*100:4.1f}%]"
        )

    lines.extend([
        "",
        "WIN PROBABILITIES",
        "-" * 40,
    ])

    for _, row in win_probs_df.iterrows():
        lines.append(
            f"  {row['candidate']:25s} {row['leading_prob']*100:5.1f}%"
        )

    second_round_prob = win_probs_df['second_round_prob'].iloc[0]
    lines.extend([
        "",
        f"Probability of Second Round: {second_round_prob:.1%}",
        "",
        "=" * 60,
    ])

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Saved report to {output_path}")

    return report


def save_forecast_results(
    model,
    output_dir: str,
    include_plots: bool = True,
) -> Dict[str, str]:
    """
    Save all forecast results to files.

    Args:
        model: PresidentialElectionModel with completed sampling
        output_dir: Directory to save results
        include_plots: Whether to generate and save plots

    Returns:
        Dictionary mapping output type to file path
    """
    os.makedirs(output_dir, exist_ok=True)

    output_files = {}

    # Get forecasts
    forecast_df = model.get_forecast()
    win_probs_df = model.get_win_probabilities()

    # Save CSVs
    forecast_path = os.path.join(output_dir, 'forecast.csv')
    forecast_df.to_csv(forecast_path, index=False)
    output_files['forecast_csv'] = forecast_path

    win_probs_path = os.path.join(output_dir, 'win_probabilities.csv')
    win_probs_df.to_csv(win_probs_path, index=False)
    output_files['win_probs_csv'] = win_probs_path

    # Save report
    report = generate_forecast_report(
        forecast_df, win_probs_df,
        model.dataset.election_date,
        os.path.join(output_dir, 'forecast_report.txt')
    )
    output_files['report'] = os.path.join(output_dir, 'forecast_report.txt')

    # Save inference data
    trace_path = os.path.join(output_dir, 'trace.nc')
    model.trace.to_netcdf(trace_path)
    output_files['trace'] = trace_path

    if include_plots:
        # Trajectory plot
        calendar_time = pd.to_datetime(model.coords['calendar_time'])
        plot_support_trajectory(
            model.trace,
            model.dataset.candidates,
            calendar_time,
            polls_df=model.dataset.polls_raw,
            election_date=model.dataset.election_date_dt,
            save_path=os.path.join(output_dir, 'trajectory.png')
        )
        output_files['trajectory_plot'] = os.path.join(output_dir, 'trajectory.png')

        # Forecast bars
        plot_forecast_bars(
            forecast_df,
            save_path=os.path.join(output_dir, 'forecast_bars.png')
        )
        output_files['forecast_bars'] = os.path.join(output_dir, 'forecast_bars.png')

        # Win probabilities
        plot_win_probabilities(
            win_probs_df,
            save_path=os.path.join(output_dir, 'win_probabilities.png')
        )
        output_files['win_probs_plot'] = os.path.join(output_dir, 'win_probabilities.png')

        # House effects
        plot_house_effects(
            model.trace,
            model.dataset.candidates,
            list(model.coords['pollsters']),
            save_path=os.path.join(output_dir, 'house_effects.png')
        )
        output_files['house_effects_plot'] = os.path.join(output_dir, 'house_effects.png')

    print(f"\nSaved all results to {output_dir}")
    return output_files
