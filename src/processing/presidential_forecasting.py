"""
Presidential Election Forecasting Processing.

This module provides functions for generating forecasts, visualizations,
and analysis outputs for Portuguese presidential election models.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import arviz as az

# Import shared export utilities
from .dashboard_exports import (
    get_contestant_color,
    PRESIDENTIAL_CANDIDATE_COLORS,
    generate_all_dashboard_files,
    save_trace_zarr,
    save_json,
    build_forecast_json,
    build_trends_json,
    build_trajectories_json,
    build_house_effects_json,
    build_polls_json,
    format_float,
    format_date,
)

# Re-export for backwards compatibility
CANDIDATE_COLORS = PRESIDENTIAL_CANDIDATE_COLORS


def get_candidate_color(candidate: str) -> str:
    """Get color for a candidate, with fallback."""
    return get_contestant_color(candidate, election_type='presidential')


def plot_support_trajectory(
    idata: az.InferenceData,
    candidates: List[str],
    calendar_time: pd.DatetimeIndex,
    polls_df: Optional[pd.DataFrame] = None,
    election_date: Optional[pd.Timestamp] = None,
    title: str = "Presidential Election Forecast",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    limit_to_polls: bool = True,
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
        limit_to_polls: If True, limit x-axis to poll date range only

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
        q25 = candidate_probs.quantile(0.25, dim=['chain', 'draw']).values
        q75 = candidate_probs.quantile(0.75, dim=['chain', 'draw']).values

        # Plot only 50% credible interval (narrower bands)
        ax.fill_between(calendar_dates, q25 * 100, q75 * 100,
                       color=color, alpha=0.2)
        ax.plot(calendar_dates, mean * 100, color=color, linewidth=2.5,
               label=f'{candidate}')

    # Overlay poll data with clear markers
    if polls_df is not None:
        for i, candidate in enumerate(candidates):
            if candidate in polls_df.columns:
                color = get_candidate_color(candidate)
                # Get poll proportions (if stored as counts, convert back)
                if 'sample_size' in polls_df.columns:
                    poll_props = polls_df[candidate] / polls_df['sample_size'] * 100
                else:
                    poll_props = polls_df[candidate] * 100
                # Larger markers with black edge for visibility
                ax.scatter(polls_df['date'], poll_props,
                          color=color, alpha=0.8, s=60, marker='o',
                          edgecolors='black', linewidths=0.5, zorder=10)

    # Add election day line (only if not limiting to polls)
    if election_date is not None and not limit_to_polls:
        ax.axvline(election_date, color='black', linestyle='--',
                  linewidth=1.5, alpha=0.7, label='Election Day')

    # Add 50% threshold line (for first round win)
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5)

    # Limit x-axis to poll date range if requested
    if limit_to_polls and polls_df is not None and 'date' in polls_df.columns:
        poll_dates = pd.to_datetime(polls_df['date'])
        min_date = poll_dates.min() - pd.Timedelta(days=3)
        max_date = poll_dates.max() + pd.Timedelta(days=3)
        ax.set_xlim(min_date, max_date)

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Support (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.set_ylim(0, None)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
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
        "NOTE: This forecast models DECLARED voting intention only.",
        "Undecided voters (~35%) represent additional uncertainty",
        "that cannot be reliably modeled without external data.",
        "",
        "PROJECTED VOTE SHARES (among decided voters)",
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


def generate_dashboard_json(
    model,
    output_dir: str,
    n_trajectory_samples: int = 100,
) -> Dict[str, str]:
    """
    Generate JSON files for the web dashboard.
    
    Uses shared export utilities from dashboard_exports.py for consistent
    format across election types.

    Args:
        model: PresidentialElectionModel with completed sampling
        output_dir: Directory to save JSON files
        n_trajectory_samples: Number of simulation paths for spaghetti plot

    Returns:
        Dictionary mapping output type to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data from model
    forecast_df = model.get_forecast()
    win_probs_df = model.get_win_probabilities()
    candidates = model.dataset.candidates
    calendar_time = pd.to_datetime(model.coords['calendar_time'])
    probs = model.trace.posterior['national_probs_calendar']
    pollsters = list(model.coords['pollsters'])
    
    # Use shared export function
    output_files = generate_all_dashboard_files(
        output_dir=output_dir,
        election_type='presidential',
        election_date=model.dataset.election_date,
        forecast_df=forecast_df,
        posterior_trends=probs,
        time_coords=calendar_time,
        contestant_names=candidates,
        house_effects=model.trace.posterior['house_effects'],
        pollster_names=pollsters,
        polls_df=model.dataset.polls_raw,
        win_probabilities_df=win_probs_df,
        contestant_column='candidate',
        contestant_dim='candidates',
        include_trajectories=True,
        n_trajectory_samples=n_trajectory_samples,
        file_prefix='presidential_',
    )
    
    # Convert output keys to legacy format for backwards compatibility
    legacy_files = {}
    key_mapping = {
        'forecast': 'forecast_json',
        'win_probabilities': 'win_probs_json',
        'trends': 'trends_json',
        'trajectories': 'trajectories_json',
        'house_effects': 'house_effects_json',
        'polls': 'polls_json',
    }
    for new_key, legacy_key in key_mapping.items():
        if new_key in output_files:
            legacy_files[legacy_key] = output_files[new_key]
    
    return legacy_files


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

    # Save inference data as Zarr (standardized format, shared utility)
    output_files['trace'] = save_trace_zarr(model.trace, output_dir, 'trace.zarr')

    # Generate dashboard JSON files
    dashboard_files = generate_dashboard_json(model, output_dir)
    output_files.update(dashboard_files)

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


# =============================================================================
# HEAD-TO-HEAD PROBABILITY
# =============================================================================

def build_head_to_head_json(
    posterior_probs: "xr.DataArray",
    candidates: List[str],
    time_coords: pd.DatetimeIndex,
    election_date: str,
    candidate_a: Optional[str] = None,
    candidate_b: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build JSON structure for head-to-head probability visualization.
    
    Computes P(candidate_a > candidate_b) at each time point.
    If candidates not specified, uses the top two candidates by mean support.
    
    Args:
        posterior_probs: DataArray with posterior samples (chain, draw, time, candidates)
        candidates: List of candidate names
        time_coords: Time coordinates
        election_date: Election date string
        candidate_a: First candidate (optional, defaults to leader)
        candidate_b: Second candidate (optional, defaults to second place)
        
    Returns:
        Dictionary with head-to-head probability over time
    """
    import xarray as xr
    
    # If candidates not specified, determine top 2 by mean at last time point
    if candidate_a is None or candidate_b is None:
        last_time_probs = posterior_probs.isel(calendar_time=-1)
        means = last_time_probs.mean(dim=['chain', 'draw']).values
        top2_idx = np.argsort(means)[-2:][::-1]
        candidate_a = candidates[top2_idx[0]]
        candidate_b = candidates[top2_idx[1]]
    
    idx_a = candidates.index(candidate_a)
    idx_b = candidates.index(candidate_b)
    
    probs_a = posterior_probs.isel(candidates=idx_a)
    probs_b = posterior_probs.isel(candidates=idx_b)
    
    # Compute P(A > B) at each time point
    # Flatten chains and draws into single sample dimension
    n_times = len(time_coords)
    probabilities = []
    
    for t in range(n_times):
        samples_a = probs_a.isel(calendar_time=t).values.flatten()
        samples_b = probs_b.isel(calendar_time=t).values.flatten()
        p_a_leads = np.mean(samples_a > samples_b)
        probabilities.append(format_float(p_a_leads))
    
    return {
        'election_date': election_date,
        'candidate_a': candidate_a,
        'candidate_b': candidate_b,
        'color_a': get_candidate_color(candidate_a),
        'color_b': get_candidate_color(candidate_b),
        'dates': [format_date(d) for d in time_coords],
        'probability_a_leads': probabilities,
    }


# =============================================================================
# RUNOFF PAIR PROBABILITIES
# =============================================================================

def compute_runoff_pair_probabilities(
    election_day_probs: np.ndarray,
    candidates: List[str],
) -> pd.DataFrame:
    """
    Compute the probability of each pair of candidates going to the second round.
    
    For each posterior sample, determines which two candidates finish 1st and 2nd,
    then counts how often each pair appears across all samples.
    
    Args:
        election_day_probs: Array of shape (n_samples, n_candidates) with 
                           election day vote share probabilities
        candidates: List of candidate names
        
    Returns:
        DataFrame with columns: candidate_a, candidate_b, probability
        Sorted by probability descending
    """
    n_samples = election_day_probs.shape[0]
    n_candidates = len(candidates)
    
    # Initialize pair counts
    from collections import Counter
    pair_counts = Counter()
    
    # For each sample, determine 1st and 2nd place
    for i in range(n_samples):
        sample = election_day_probs[i]
        # Get indices of top 2 candidates
        top2_idx = np.argsort(sample)[-2:][::-1]  # Descending order
        first_idx, second_idx = top2_idx[0], top2_idx[1]
        
        # Create canonical pair (alphabetically sorted for consistency)
        pair = tuple(sorted([candidates[first_idx], candidates[second_idx]]))
        pair_counts[pair] += 1
    
    # Convert to DataFrame
    results = []
    for pair, count in pair_counts.items():
        results.append({
            'candidate_a': pair[0],
            'candidate_b': pair[1],
            'probability': count / n_samples
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('probability', ascending=False).reset_index(drop=True)
    return df


def build_runoff_pairs_json(
    election_day_probs: np.ndarray,
    candidates: List[str],
    election_date: str,
) -> Dict[str, Any]:
    """
    Build JSON structure for runoff pair probabilities visualization.
    
    Args:
        election_day_probs: Array of shape (n_samples, n_candidates)
        candidates: List of candidate names
        election_date: Election date string
        
    Returns:
        Dictionary with runoff pair data for both bar chart and matrix
    """
    runoff_df = compute_runoff_pair_probabilities(election_day_probs, candidates)
    
    # Build pair list for bar chart (top pairs)
    pairs = []
    for _, row in runoff_df.iterrows():
        pairs.append({
            'candidate_a': row['candidate_a'],
            'candidate_b': row['candidate_b'],
            'probability': format_float(row['probability']),
            'color_a': get_candidate_color(row['candidate_a']),
            'color_b': get_candidate_color(row['candidate_b']),
        })
    
    # Build probability matrix for heatmap
    # Only include top candidates (those with >1% chance of being in runoff)
    candidates_in_runoff = set()
    for _, row in runoff_df.iterrows():
        if row['probability'] >= 0.01:  # At least 1% chance
            candidates_in_runoff.add(row['candidate_a'])
            candidates_in_runoff.add(row['candidate_b'])
    
    # Order by mean support (approximate from pairs)
    candidate_order = [c for c in candidates if c in candidates_in_runoff]
    
    # Build matrix
    n = len(candidate_order)
    matrix = [[0.0] * n for _ in range(n)]
    
    for _, row in runoff_df.iterrows():
        if row['candidate_a'] in candidate_order and row['candidate_b'] in candidate_order:
            i = candidate_order.index(row['candidate_a'])
            j = candidate_order.index(row['candidate_b'])
            matrix[i][j] = format_float(row['probability'])
            matrix[j][i] = format_float(row['probability'])  # Symmetric
    
    return {
        'election_date': election_date,
        'pairs': pairs,
        'matrix': {
            'candidates': candidate_order,
            'colors': [get_candidate_color(c) for c in candidate_order],
            'probabilities': matrix,
        }
    }


# =============================================================================
# DASHBOARD EXPORT FROM SAVED TRACE
# =============================================================================

def export_dashboard_from_trace(
    trace_path: str,
    output_dir: str,
    election_date: str,
    polls_df: Optional[pd.DataFrame] = None,
    n_trajectory_samples: int = 100,
) -> Dict[str, str]:
    """
    Export dashboard JSON files from a saved zarr trace.
    
    This function loads a previously saved model trace and generates all
    the JSON files needed for the web dashboard.
    
    Args:
        trace_path: Path to the saved zarr trace directory
        output_dir: Directory to save JSON files
        election_date: Election date string (e.g., '2026-01-18')
        polls_df: Optional DataFrame with raw poll data
        n_trajectory_samples: Number of samples for trajectory JSON
        
    Returns:
        Dictionary mapping output type to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = {}
    
    # Load the trace
    print(f"Loading trace from {trace_path}...")
    idata = az.from_zarr(trace_path)
    
    # Extract coordinates from trace
    candidates = list(idata.posterior.coords['candidates'].values)
    pollsters = list(idata.posterior.coords['pollsters'].values)
    calendar_time = pd.to_datetime(idata.posterior.coords['calendar_time'].values)
    
    print(f"Found {len(candidates)} candidates: {candidates}")
    print(f"Found {len(pollsters)} pollsters: {pollsters}")
    print(f"Calendar time range: {calendar_time.min()} to {calendar_time.max()}")
    
    # Extract posterior data
    probs = idata.posterior['national_probs_calendar']
    election_day_probs = idata.posterior['election_day_probs']
    house_effects = idata.posterior['house_effects']
    
    # Compute forecast statistics from election day probs
    forecast_data = []
    for i, candidate in enumerate(candidates):
        candidate_probs = election_day_probs.isel(candidates=i).values.flatten()
        forecast_data.append({
            'candidate': candidate,
            'mean': np.mean(candidate_probs),
            'median': np.median(candidate_probs),
            'ci_lower': np.percentile(candidate_probs, 2.5),
            'ci_upper': np.percentile(candidate_probs, 97.5),
            'ci_10': np.percentile(candidate_probs, 10),
            'ci_90': np.percentile(candidate_probs, 90),
        })
    forecast_df = pd.DataFrame(forecast_data)
    forecast_df = forecast_df.sort_values('mean', ascending=False).reset_index(drop=True)
    
    # Compute win probabilities
    ed_probs_flat = election_day_probs.values.reshape(-1, len(candidates))
    n_samples = ed_probs_flat.shape[0]
    
    leading_counts = np.zeros(len(candidates))
    first_round_wins = np.zeros(len(candidates))
    second_round_needed = 0
    
    for i in range(n_samples):
        sample = ed_probs_flat[i]
        leader_idx = np.argmax(sample)
        leading_counts[leader_idx] += 1
        
        if sample[leader_idx] > 0.5:
            first_round_wins[leader_idx] += 1
        else:
            second_round_needed += 1
    
    win_probs_data = []
    for i, candidate in enumerate(candidates):
        win_probs_data.append({
            'candidate': candidate,
            'leading_prob': leading_counts[i] / n_samples,
            'first_round_win_prob': first_round_wins[i] / n_samples,
            'mean_support': forecast_df[forecast_df['candidate'] == candidate]['mean'].iloc[0],
            'second_round_prob': second_round_needed / n_samples,
        })
    win_probs_df = pd.DataFrame(win_probs_data)
    win_probs_df = win_probs_df.sort_values('leading_prob', ascending=False).reset_index(drop=True)
    
    # Generate standard dashboard files using shared utilities
    print("Generating standard dashboard files...")
    standard_files = generate_all_dashboard_files(
        output_dir=output_dir,
        election_type='presidential',
        election_date=election_date,
        forecast_df=forecast_df,
        posterior_trends=probs,
        time_coords=calendar_time,
        contestant_names=candidates,
        house_effects=house_effects,
        pollster_names=pollsters,
        polls_df=polls_df,
        win_probabilities_df=win_probs_df,
        contestant_column='candidate',
        contestant_dim='candidates',
        include_trajectories=True,
        n_trajectory_samples=n_trajectory_samples,
        file_prefix='presidential_',
    )
    output_files.update(standard_files)
    
    # Generate runoff pairs JSON
    print("Generating runoff pairs data...")
    runoff_data = build_runoff_pairs_json(ed_probs_flat, candidates, election_date)
    runoff_path = save_json(runoff_data, output_dir, 'runoff_pairs.json', 'presidential_')
    output_files['runoff_pairs'] = runoff_path
    
    # Generate head-to-head JSON (for top 2 candidates)
    print("Generating head-to-head data...")
    head_to_head_data = build_head_to_head_json(
        probs, candidates, calendar_time, election_date
    )
    head_to_head_path = save_json(head_to_head_data, output_dir, 'head_to_head.json', 'presidential_')
    output_files['head_to_head'] = head_to_head_path
    
    print(f"\nExported {len(output_files)} files to {output_dir}")
    return output_files
