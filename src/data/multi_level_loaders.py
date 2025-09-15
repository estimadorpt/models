"""
Multi-level data loaders for Portuguese election results.

This module loads parish-level election data and provides flexible aggregation
to municipality, district, or national levels for different election types.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Literal
from src.config import DATA_DIR
from src.data.geographic_mapping import (
    create_geographic_mapping_table,
    validate_aggregation_consistency
)


def load_parish_level_results(
    election_dates: List[str], 
    political_families: List[str],
    data_dir: str = None
) -> pd.DataFrame:
    """
    Load raw parish-level election results for multiple elections.
    
    Args:
        election_dates: List of election dates (YYYY-MM-DD format)
        political_families: List of party names to include
        data_dir: Directory containing election parquet files
        
    Returns:
        DataFrame with parish-level results across all elections
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    all_results = []
    
    # Load geographic mapping once
    mapping_file = os.path.join(data_dir, 'geographic_mappings.csv')
    if os.path.exists(mapping_file):
        geographic_mapping = pd.read_csv(mapping_file)
    else:
        # Create mappings if they don't exist
        latest_file = os.path.join(data_dir, 'legislativas_2024.parquet')
        geographic_mapping = create_geographic_mapping_table(latest_file)
    
    for election_date in election_dates:
        # Convert date to election file pattern
        year = election_date.split('-')[0]
        election_file = os.path.join(data_dir, f'legislativas_{year}.parquet')
        
        if not os.path.exists(election_file):
            print(f"Warning: Election file not found: {election_file}")
            continue
            
        print(f"Loading parish-level data for {election_date} from {election_file}")
        
        # Load raw parish data
        parish_df = pd.read_parquet(election_file)
        
        # Add election date
        parish_df['election_date'] = pd.to_datetime(election_date)
        
        # Standardize column names first
        parish_df = parish_df.rename(columns={
            'territoryCode': 'territory_code',
            'territoryName': 'territory_name'
        })
        
        # Merge with geographic mapping to get consistent geographic info
        parish_df = parish_df.merge(
            geographic_mapping[['territory_code', 'district_code', 'municipality_code', 
                               'parish_code', 'district_name', 'full_municipality_id', 'full_parish_id']],
            on='territory_code',
            how='left'
        )
        
        # Map party columns to standard political families
        parish_df = map_party_columns(parish_df, political_families, year)
        
        all_results.append(parish_df)
    
    if not all_results:
        return pd.DataFrame()
    
    # Combine all elections
    combined_results = pd.concat(all_results, ignore_index=True)
    
    print(f"Loaded parish-level results: {len(combined_results)} parish-election combinations")
    return combined_results


def map_party_columns(df: pd.DataFrame, political_families: List[str], year: str) -> pd.DataFrame:
    """
    Map raw election data column names to standard political family names.
    
    Different elections may have different coalition representations
    (e.g., "PPD/PSD.CDS-PP" in 2024 vs separate parties in earlier years).
    """
    df = df.copy()
    
    print(f"  Mapping party columns for {year}. Available columns: {[col for col in df.columns if any(party in col for party in ['PS', 'PSD', 'CH', 'IL', 'BE', 'PCP', 'PAN', 'L'])]}")
    
    # Create mapping based on election year and known coalition patterns
    if year >= '2024':
        # AD coalition represented as "PPD/PSD.CDS-PP" or variants
        coalition_columns = [col for col in df.columns if 'PPD/PSD' in col and 'CDS' in col]
        if coalition_columns:
            # Use the first match as AD
            df['AD'] = df[coalition_columns[0]].fillna(0)
            print(f"    Mapped {coalition_columns[0]} -> AD")
    
    # Map other parties with flexible matching
    party_mappings = {
        'PS': ['PS'],
        'CH': ['CH', 'CHEGA'],
        'IL': ['IL', 'INICIATIVA LIBERAL'],
        'BE': ['B.E.', 'BE', 'BLOCO DE ESQUERDA'],
        'CDU': ['PCP-PEV', 'CDU', 'PCP'],
        'PAN': ['PAN'],
        'L': ['L', 'LIVRE'],
    }
    
    for standard_name, possible_names in party_mappings.items():
        if standard_name not in df.columns:
            # Find matching column
            mapped = False
            for possible_name in possible_names:
                matching_cols = [col for col in df.columns if possible_name == col or possible_name in col]
                if matching_cols:
                    df[standard_name] = df[matching_cols[0]].fillna(0)
                    print(f"    Mapped {matching_cols[0]} -> {standard_name}")
                    mapped = True
                    break
            
            # If still not found, fill with zeros
            if not mapped:
                df[standard_name] = 0.0
                print(f"    Created {standard_name} with zeros (no match found)")
    
    # Ensure all political families are present
    for party in political_families:
        if party not in df.columns:
            df[party] = 0.0
    
    return df


def aggregate_to_level(
    parish_df: pd.DataFrame,
    level: Literal['municipality', 'district', 'national'],
    political_families: List[str]
) -> pd.DataFrame:
    """
    Aggregate parish-level results to municipality, district, or national level.
    
    Args:
        parish_df: Parish-level results DataFrame
        level: Aggregation level ('municipality', 'district', or 'national')  
        political_families: List of party names to aggregate
        
    Returns:
        Aggregated results DataFrame
    """
    if parish_df.empty:
        return pd.DataFrame()
    
    # Define grouping columns based on aggregation level
    if level == 'municipality':
        group_cols = ['election_date', 'full_municipality_id', 'district_name']
        geo_id_col = 'full_municipality_id'
    elif level == 'district':
        group_cols = ['election_date', 'district_name']  
        geo_id_col = 'district_name'
    elif level == 'national':
        group_cols = ['election_date']
        geo_id_col = None
    else:
        raise ValueError(f"Invalid aggregation level: {level}")
    
    # Aggregate numeric columns
    numeric_cols = ['number_voters', 'subscribed_voters', 'null_votes', 'blank_votes'] + political_families
    agg_dict = {col: 'sum' for col in numeric_cols if col in parish_df.columns}
    
    # Group and aggregate
    aggregated = parish_df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Add geographic identifier column for consistency
    if geo_id_col:
        aggregated['geographic_id'] = aggregated[geo_id_col]
    else:
        aggregated['geographic_id'] = 'national'
    
    # Calculate sample size equivalent (total valid votes)
    valid_vote_cols = [col for col in political_families if col in aggregated.columns]
    if valid_vote_cols:
        aggregated['sample_size'] = aggregated[valid_vote_cols].sum(axis=1)
    
    # Add date column for compatibility
    aggregated['date'] = aggregated['election_date']
    
    return aggregated


def load_multi_level_results(
    election_dates: List[str],
    political_families: List[str], 
    aggregate_to: Literal['parish', 'municipality', 'district', 'national'] = 'district',
    validate_totals: bool = True
) -> pd.DataFrame:
    """
    Load election results at specified aggregation level with validation.
    
    This is the main entry point for loading results at any geographic level.
    
    Args:
        election_dates: List of election dates  
        political_families: List of party names
        aggregate_to: Target aggregation level
        validate_totals: Whether to validate aggregation consistency
        
    Returns:
        DataFrame with results at requested aggregation level
    """
    print(f"Loading results aggregated to {aggregate_to} level...")
    
    # Load raw parish data
    parish_results = load_parish_level_results(election_dates, political_families)
    
    if parish_results.empty:
        print("No parish data loaded")
        return pd.DataFrame()
    
    # Return parish level directly if requested
    if aggregate_to == 'parish':
        return parish_results
    
    # Aggregate to requested level
    aggregated_results = aggregate_to_level(parish_results, aggregate_to, political_families)
    
    # Validate aggregation if requested
    if validate_totals and not aggregated_results.empty:
        validation = validate_aggregation_consistency(
            parish_results, aggregated_results, None, aggregate_to
        )
        
        if not validation['vote_total_matches']:
            print(f"Warning: Vote totals don't match after {aggregate_to} aggregation")
            for error in validation['aggregation_errors'][:3]:  # Show first 3 errors
                print(f"  {error['party']}: {error['difference']} vote difference")
        else:
            print(f"✓ Vote totals validated for {aggregate_to} aggregation")
    
    print(f"Loaded {len(aggregated_results)} {aggregate_to}-level results")
    return aggregated_results


def get_available_geographic_levels(election_dates: List[str]) -> Dict[str, int]:
    """
    Get count of available geographic divisions for given elections.
    
    Returns:
        Dict with counts for each geographic level
    """
    parish_results = load_parish_level_results(election_dates, ['PS'])  # Just load one party for counting
    
    if parish_results.empty:
        return {}
    
    return {
        'parishes': parish_results['territory_code'].nunique(),
        'municipalities': parish_results['full_municipality_id'].nunique(),
        'districts': parish_results['district_name'].nunique()
    }


def create_geographic_coordinates(
    aggregation_level: Literal['parish', 'municipality', 'district'],
    election_dates: List[str]
) -> Dict[str, List[str]]:
    """
    Create coordinate lists for PyMC models at different geographic levels.
    
    Args:
        aggregation_level: Geographic level for modeling
        election_dates: Elections to include
        
    Returns:
        Dict with coordinate lists for model building
    """
    # Load sample data to get geographic divisions
    sample_results = load_parish_level_results(election_dates[:1], ['PS'])
    
    if sample_results.empty:
        return {}
    
    if aggregation_level == 'parish':
        geo_coords = sorted(sample_results['territory_code'].unique())
        coord_name = 'parishes'
    elif aggregation_level == 'municipality':
        geo_coords = sorted(sample_results['full_municipality_id'].unique())
        coord_name = 'municipalities' 
    elif aggregation_level == 'district':
        geo_coords = sorted(sample_results['district_name'].unique())
        coord_name = 'districts'
    else:
        raise ValueError(f"Invalid aggregation level: {aggregation_level}")
    
    return {
        coord_name: geo_coords,
        'elections': election_dates
    }


if __name__ == "__main__":
    # Test the multi-level loading system
    test_elections = ['2024-03-10', '2022-01-30']
    test_parties = ['PS', 'AD', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L']
    
    print("=== Testing Multi-Level Data Loading ===")
    
    # Test different aggregation levels
    for level in ['national', 'district', 'municipality']:
        print(f"\n--- {level.upper()} Level ---")
        results = load_multi_level_results(
            test_elections, 
            test_parties, 
            aggregate_to=level,
            validate_totals=True
        )
        if not results.empty:
            print(f"Shape: {results.shape}")
            print("Sample data:")
            print(results[['election_date', 'geographic_id'] + test_parties[:3]].head(3))
    
    # Test geographic level availability
    print(f"\n--- Geographic Level Availability ---")
    levels = get_available_geographic_levels(test_elections)
    for level, count in levels.items():
        print(f"{level}: {count}")