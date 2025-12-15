"""
Unified Dashboard Export Functions.

This module provides shared export functions and consistent data formats
for generating web dashboard JSON files across different election types
(presidential, parliamentary).
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az


# =============================================================================
# SHARED COLOR CONFIGURATIONS
# =============================================================================

# Party colors for parliamentary elections
PARTY_COLORS = {
    'PS': '#FF69B4',      # Pink
    'PSD': '#FF8C00',     # Orange
    'AD': '#FF8C00',      # Orange (coalition)
    'CH': '#8B0000',      # Dark Red
    'IL': '#00CED1',      # Cyan
    'BE': '#DC143C',      # Crimson
    'CDU': '#228B22',     # Green
    'PCP': '#228B22',     # Green
    'PAN': '#2E8B57',     # Sea Green
    'L': '#90EE90',       # Light Green
    'Others': '#808080',  # Gray
}

# Candidate colors for presidential elections
PRESIDENTIAL_CANDIDATE_COLORS = {
    'Gouveia e Melo': '#4A90D9',      # Blue
    'Marques Mendes': '#FF8C00',       # Orange
    'António José Seguro': '#FF69B4',  # Pink
    'André Ventura': '#8B0000',        # Dark Red
    'Cotrim Figueiredo': '#00CED1',    # Cyan
    'Catarina Martins': '#DC143C',     # Crimson
    'António Filipe': '#228B22',       # Green
    'Others': '#808080',               # Gray
}


def get_contestant_color(
    name: str,
    election_type: str = 'parliamentary'
) -> str:
    """
    Get color for a party or candidate.
    
    Args:
        name: Name of party or candidate
        election_type: 'parliamentary' or 'presidential'
        
    Returns:
        Hex color string
    """
    if election_type == 'presidential':
        return PRESIDENTIAL_CANDIDATE_COLORS.get(name, '#666666')
    else:
        return PARTY_COLORS.get(name, '#666666')


# =============================================================================
# SHARED DATA FORMATTING UTILITIES
# =============================================================================

def format_date(date: Union[datetime, pd.Timestamp, np.datetime64, str]) -> str:
    """Convert various date formats to ISO string."""
    if isinstance(date, str):
        return date
    return pd.to_datetime(date).strftime('%Y-%m-%d')


def format_float(value: float, decimals: int = 4) -> float:
    """Round float for JSON export."""
    return round(float(value), decimals)


def compute_quantiles(
    data: xr.DataArray,
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
    dims: List[str] = ['chain', 'draw']
) -> Dict[str, np.ndarray]:
    """
    Compute quantiles from posterior samples.
    
    Args:
        data: DataArray with posterior samples
        quantiles: List of quantile values to compute
        dims: Dimensions to reduce over
        
    Returns:
        Dictionary mapping quantile names to arrays
    """
    result = {}
    for q in quantiles:
        key = f'q{int(q*100):02d}'
        result[key] = data.quantile(q, dim=dims).values
    result['mean'] = data.mean(dim=dims).values
    return result


# =============================================================================
# SHARED JSON EXPORT STRUCTURE BUILDERS
# =============================================================================

def build_forecast_json(
    forecast_df: pd.DataFrame,
    election_date: str,
    election_type: str = 'parliamentary',
    contestant_column: str = 'candidate',
) -> Dict[str, Any]:
    """
    Build standardized forecast JSON structure.
    
    Works for both presidential candidates and parliamentary parties.
    
    Args:
        forecast_df: DataFrame with forecast data (mean, ci_lower, ci_upper, etc.)
        election_date: Election date string
        election_type: 'presidential' or 'parliamentary'
        contestant_column: Name of the column containing party/candidate names
        
    Returns:
        Standardized forecast dictionary
    """
    data = {
        "election_type": election_type,
        "election_date": election_date,
        "updated_at": pd.Timestamp.now().isoformat(),
        "candidates": []
    }
    
    for _, row in forecast_df.iterrows():
        name = row[contestant_column]
        contestant = {
            "name": name,
            "color": get_contestant_color(name, election_type),
            "mean": format_float(row['mean']),
            "ci_lower": format_float(row.get('ci_lower', row.get('hdi_low', 0))),
            "ci_upper": format_float(row.get('ci_upper', row.get('hdi_high', 0))),
        }
        
        # Add optional fields if present
        if 'median' in row:
            contestant['median'] = format_float(row['median'])
        if 'ci_10' in row:
            contestant['ci_10'] = format_float(row['ci_10'])
        if 'ci_90' in row:
            contestant['ci_90'] = format_float(row['ci_90'])
            
        data["candidates"].append(contestant)
    
    return data


def build_trends_json(
    posterior: xr.DataArray,
    time_coords: pd.DatetimeIndex,
    contestant_names: List[str],
    election_date: str,
    election_type: str = 'parliamentary',
    contestant_dim: str = 'parties_complete',
    time_dim: str = 'calendar_time',
) -> Dict[str, Any]:
    """
    Build standardized time series trends JSON.
    
    Args:
        posterior: DataArray with posterior samples (chain, draw, time, contestants)
        time_coords: Time coordinates
        contestant_names: List of party/candidate names
        election_date: Election date
        election_type: 'presidential' or 'parliamentary'
        contestant_dim: Name of contestant dimension
        time_dim: Name of time dimension
        
    Returns:
        Standardized trends dictionary
    """
    data = {
        "election_type": election_type,
        "election_date": election_date,
        "dates": [format_date(d) for d in time_coords],
        "candidates": {}
    }
    
    for i, name in enumerate(contestant_names):
        contestant_data = posterior.isel({contestant_dim: i})
        
        mean = contestant_data.mean(dim=['chain', 'draw']).values
        q05 = contestant_data.quantile(0.05, dim=['chain', 'draw']).values
        q95 = contestant_data.quantile(0.95, dim=['chain', 'draw']).values
        q25 = contestant_data.quantile(0.25, dim=['chain', 'draw']).values
        q75 = contestant_data.quantile(0.75, dim=['chain', 'draw']).values
        
        data["candidates"][name] = {
            "color": get_contestant_color(name, election_type),
            "mean": [format_float(v) for v in mean],
            "ci_05": [format_float(v) for v in q05],
            "ci_95": [format_float(v) for v in q95],
            "ci_25": [format_float(v) for v in q25],
            "ci_75": [format_float(v) for v in q75],
        }
    
    return data


def build_snapshot_probabilities_json(
    posterior: xr.DataArray,
    time_coords: pd.DatetimeIndex,
    contestant_names: List[str],
    election_date: str,
    election_type: str = 'parliamentary',
    contestant_dim: str = 'parties_complete',
    time_dim: str = 'calendar_time',
) -> Dict[str, Any]:
    """
    Build "snapshot" (as-of-date) leader probabilities from joint posterior draws.

    This computes, for each date t, the probability that each contestant is
    the leader (highest share) in the first round *if the election were held at t*.

    Critically, we compute this from the joint posterior samples (not marginal
    Normal approximations), preserving the zero-sum / negative correlation structure.
    """
    data: Dict[str, Any] = {
        "election_type": election_type,
        "election_date": election_date,
        "dates": [format_date(d) for d in time_coords],
        "candidates": {},
        "metric": "first_round_leader_probability",
    }

    # Ensure consistent dimension ordering then materialize a numpy view
    # Shape: (chain, draw, time, contestant)
    arr = posterior.transpose('chain', 'draw', time_dim, contestant_dim).values
    n_chains, n_draws, n_times, n_contestants = arr.shape
    flat = arr.reshape(n_chains * n_draws, n_times, n_contestants)

    # Argmax along contestants for each (sample, time)
    leaders = np.argmax(flat, axis=2)  # (n_samples, n_times)

    for i, name in enumerate(contestant_names):
        probs = (leaders == i).mean(axis=0)
        data["candidates"][name] = {
            "color": get_contestant_color(name, election_type),
            "leading_probability": [format_float(v) for v in probs],
        }

    return data


def build_trajectories_json(
    posterior: xr.DataArray,
    time_coords: pd.DatetimeIndex,
    contestant_names: List[str],
    election_date: str,
    election_type: str = 'parliamentary',
    contestant_dim: str = 'parties_complete',
    n_samples: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Build trajectories JSON for spaghetti plots.
    
    Args:
        posterior: DataArray with posterior samples
        time_coords: Time coordinates
        contestant_names: List of party/candidate names
        election_date: Election date
        election_type: 'presidential' or 'parliamentary'
        contestant_dim: Name of contestant dimension
        n_samples: Number of trajectory samples to include
        seed: Random seed for reproducibility
        
    Returns:
        Trajectories dictionary
    """
    n_chains = posterior.sizes['chain']
    n_draws = posterior.sizes['draw']
    total_samples = n_chains * n_draws
    
    np.random.seed(seed)
    sample_indices = np.random.choice(
        total_samples, 
        min(n_samples, total_samples), 
        replace=False
    )
    
    data = {
        "election_type": election_type,
        "election_date": election_date,
        "dates": [format_date(d) for d in time_coords],
        "n_samples": len(sample_indices),
        "candidates": {}
    }
    
    for i, name in enumerate(contestant_names):
        contestant_probs = posterior.isel({contestant_dim: i}).values
        flat_probs = contestant_probs.reshape(total_samples, -1)
        
        data["candidates"][name] = {
            "color": get_contestant_color(name, election_type),
            "trajectories": [
                [format_float(v) for v in flat_probs[idx]] 
                for idx in sample_indices
            ]
        }
    
    return data


def build_house_effects_json(
    house_effects: xr.DataArray,
    pollster_names: List[str],
    contestant_names: List[str],
    election_type: str = 'parliamentary',
    pollster_dim: str = 'pollsters',
    contestant_dim: str = 'parties_complete',
    active_pollsters: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build standardized house effects JSON.
    
    Args:
        house_effects: DataArray with house effects posterior
        pollster_names: List of pollster names
        contestant_names: List of party/candidate names
        election_type: 'presidential' or 'parliamentary'
        pollster_dim: Name of pollster dimension
        contestant_dim: Name of contestant dimension
        active_pollsters: Optional list to filter only active pollsters
        
    Returns:
        House effects dictionary
    """
    data = {
        "election_type": election_type,
        "pollsters": [],
        "candidates": contestant_names,
        "effects": {}
    }
    
    for p_idx, pollster in enumerate(pollster_names):
        # Skip inactive pollsters if filter provided
        if active_pollsters is not None and pollster not in active_pollsters:
            continue
            
        data["pollsters"].append(pollster)
        effects = house_effects.isel({pollster_dim: p_idx})
        data["effects"][pollster] = {}
        
        for c_idx, contestant in enumerate(contestant_names):
            contestant_effects = effects.isel({contestant_dim: c_idx})
            mean = float(contestant_effects.mean(dim=['chain', 'draw']).values)
            ci_lower = float(contestant_effects.quantile(0.05, dim=['chain', 'draw']).values)
            ci_upper = float(contestant_effects.quantile(0.95, dim=['chain', 'draw']).values)
            
            data["effects"][pollster][contestant] = {
                "mean": format_float(mean),
                "ci_lower": format_float(ci_lower),
                "ci_upper": format_float(ci_upper),
                "color": get_contestant_color(contestant, election_type),
            }
    
    return data


def build_polls_json(
    polls_df: pd.DataFrame,
    contestant_names: List[str],
    election_type: str = 'parliamentary',
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Build polls JSON for visualization overlay.
    
    Polls are normalized to sum to 1 (excluding undecided voters) to match
    the model's posterior trends, which are also normalized via softmax.
    
    Args:
        polls_df: DataFrame with poll data
        contestant_names: List of party/candidate names to include
        election_type: 'presidential' or 'parliamentary'
        normalize: If True, normalize values to sum to 1 (default True)
        
    Returns:
        Polls dictionary with normalized values
    """
    data = {
        "election_type": election_type,
        "polls": []
    }
    
    for _, row in polls_df.iterrows():
        poll_entry = {
            "date": format_date(row['date']),
            "pollster": row['pollster'],
        }
        
        # Add sample size if available
        if 'sample_size' in row and pd.notna(row['sample_size']):
            poll_entry["sample_size"] = int(row['sample_size'])
        
        # Calculate sum of all contestants for normalization
        if normalize:
            total = 0.0
            for contestant in contestant_names:
                if contestant in row and pd.notna(row[contestant]):
                    total += float(row[contestant])
            norm_factor = 1.0 / total if total > 0 else 1.0
        else:
            norm_factor = 1.0
        
        # Add contestant values (normalized if requested)
        for contestant in contestant_names:
            if contestant in row and pd.notna(row[contestant]):
                value = float(row[contestant]) * norm_factor
                poll_entry[contestant] = format_float(value)
                
        data["polls"].append(poll_entry)
    
    return data


# =============================================================================
# UNIFIED FILE EXPORT FUNCTION
# =============================================================================

def save_json(
    data: Dict[str, Any],
    output_dir: str,
    filename: str,
    prefix: str = '',
) -> str:
    """
    Save data as JSON file.
    
    Args:
        data: Dictionary to save
        output_dir: Output directory
        filename: Base filename (e.g., 'forecast.json')
        prefix: Optional prefix (e.g., 'presidential_')
        
    Returns:
        Full path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    full_filename = f"{prefix}{filename}" if prefix else filename
    output_path = os.path.join(output_dir, full_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {full_filename}")
    return output_path


def generate_all_dashboard_files(
    output_dir: str,
    election_type: str,
    election_date: str,
    forecast_df: pd.DataFrame,
    posterior_trends: xr.DataArray,
    time_coords: pd.DatetimeIndex,
    contestant_names: List[str],
    house_effects: Optional[xr.DataArray] = None,
    pollster_names: Optional[List[str]] = None,
    polls_df: Optional[pd.DataFrame] = None,
    win_probabilities_df: Optional[pd.DataFrame] = None,
    contestant_column: str = 'candidate',
    contestant_dim: str = 'candidates',
    include_trajectories: bool = True,
    n_trajectory_samples: int = 100,
    file_prefix: str = '',
) -> Dict[str, str]:
    """
    Generate all dashboard JSON files with consistent format.
    
    This is the main entry point for generating web dashboard exports.
    
    Args:
        output_dir: Directory to save files
        election_type: 'presidential' or 'parliamentary'
        election_date: Election date string
        forecast_df: Forecast DataFrame
        posterior_trends: Posterior DataArray for trends
        time_coords: Time coordinates
        contestant_names: List of party/candidate names
        house_effects: Optional house effects DataArray
        pollster_names: Optional list of pollster names
        polls_df: Optional raw polls DataFrame
        win_probabilities_df: Optional win probabilities (presidential only)
        contestant_column: Column name for contestants in DataFrames
        contestant_dim: Dimension name for contestants in DataArrays
        include_trajectories: Whether to generate trajectory JSON
        n_trajectory_samples: Number of samples for spaghetti plot
        file_prefix: Prefix for all output files (e.g., 'presidential_')
        
    Returns:
        Dictionary mapping output type to file path
    """
    output_files = {}
    
    # 1. Forecast JSON
    forecast_data = build_forecast_json(
        forecast_df, election_date, election_type, contestant_column
    )
    output_files['forecast'] = save_json(
        forecast_data, output_dir, 'forecast.json', file_prefix
    )
    
    # 2. Win Probabilities JSON (presidential specific)
    if win_probabilities_df is not None:
        # For presidential: includes second round probability
        win_probs_data = {
            "election_type": election_type,
            "election_date": election_date,
            "candidates": []
        }
        
        # Check if second_round_prob exists
        if 'second_round_prob' in win_probabilities_df.columns:
            win_probs_data["second_round_probability"] = format_float(
                win_probabilities_df['second_round_prob'].iloc[0]
            )
        
        for _, row in win_probabilities_df.iterrows():
            name = row[contestant_column]
            entry = {
                "name": name,
                "color": get_contestant_color(name, election_type),
            }
            # Add any probability columns present
            for col in ['leading_prob', 'leading_probability', 'win_prob', 
                        'first_round_win_prob', 'mean_support']:
                if col in row:
                    entry[col.replace('_prob', '_probability')] = format_float(row[col])
                    
            win_probs_data["candidates"].append(entry)
            
        output_files['win_probabilities'] = save_json(
            win_probs_data, output_dir, 'win_probabilities.json', file_prefix
        )
    
    # 3. Trends JSON
    trends_data = build_trends_json(
        posterior_trends, time_coords, contestant_names,
        election_date, election_type, contestant_dim
    )
    output_files['trends'] = save_json(
        trends_data, output_dir, 'trends.json', file_prefix
    )

    # 3b. Snapshot probabilities JSON (joint posterior leader probabilities)
    snapshot_probs_data = build_snapshot_probabilities_json(
        posterior_trends, time_coords, contestant_names,
        election_date, election_type, contestant_dim
    )
    output_files['snapshot_probabilities'] = save_json(
        snapshot_probs_data, output_dir, 'snapshot_probabilities.json', file_prefix
    )
    
    # 4. Trajectories JSON
    if include_trajectories:
        trajectories_data = build_trajectories_json(
            posterior_trends, time_coords, contestant_names,
            election_date, election_type, contestant_dim,
            n_samples=n_trajectory_samples
        )
        output_files['trajectories'] = save_json(
            trajectories_data, output_dir, 'trajectories.json', file_prefix
        )
    
    # 5. House Effects JSON
    if house_effects is not None and pollster_names is not None:
        pollster_dim = 'pollsters'
        house_effects_data = build_house_effects_json(
            house_effects, pollster_names, contestant_names,
            election_type, pollster_dim, contestant_dim
        )
        output_files['house_effects'] = save_json(
            house_effects_data, output_dir, 'house_effects.json', file_prefix
        )
    
    # 6. Polls JSON
    if polls_df is not None:
        polls_data = build_polls_json(polls_df, contestant_names, election_type)
        output_files['polls'] = save_json(
            polls_data, output_dir, 'polls.json', file_prefix
        )
    
    print(f"\nGenerated {len(output_files)} dashboard files in {output_dir}")
    return output_files


# =============================================================================
# TRACE EXPORT (Zarr format - standardized)
# =============================================================================

def save_trace_zarr(
    trace: az.InferenceData,
    output_dir: str,
    filename: str = 'trace.zarr',
) -> str:
    """
    Save inference trace in Zarr format (standardized).
    
    Args:
        trace: ArviZ InferenceData object
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to saved trace
    """
    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, filename)
    trace.to_zarr(trace_path)
    print(f"Saved trace to {trace_path}")
    return trace_path



