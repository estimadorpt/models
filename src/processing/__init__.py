"""
Processing module for election model outputs.

This module provides functions for:
- Seat prediction and allocation
- Dashboard JSON exports (unified across election types)
- Presidential election forecasting
- Electoral system calculations
"""

from .dashboard_exports import (
    # Color configurations
    PARTY_COLORS,
    PRESIDENTIAL_CANDIDATE_COLORS,
    get_contestant_color,
    
    # Export utilities
    format_date,
    format_float,
    compute_quantiles,
    save_json,
    save_trace_zarr,
    
    # JSON builders
    build_forecast_json,
    build_trends_json,
    build_trajectories_json,
    build_house_effects_json,
    build_polls_json,
    
    # Main export function
    generate_all_dashboard_files,
)

__all__ = [
    # Color configurations
    'PARTY_COLORS',
    'PRESIDENTIAL_CANDIDATE_COLORS', 
    'get_contestant_color',
    
    # Export utilities
    'format_date',
    'format_float',
    'compute_quantiles',
    'save_json',
    'save_trace_zarr',
    
    # JSON builders
    'build_forecast_json',
    'build_trends_json',
    'build_trajectories_json',
    'build_house_effects_json',
    'build_polls_json',
    
    # Main export function
    'generate_all_dashboard_files',
]





