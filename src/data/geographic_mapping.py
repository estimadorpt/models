"""
Geographic mapping utilities for Portuguese administrative divisions.

This module creates and manages mappings between parishes (freguesias), 
municipalities (concelhos), and districts (distritos) for election data processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from src.config import DATA_DIR


def parse_territory_code(territory_code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse Portuguese territory code into district, municipality, and parish components.
    
    Territory codes follow pattern: LOCAL-DDMMPP where:
    - DD = District code (01-18 mainland + 31,32,41-49 islands)
    - MM = Municipality code within district  
    - PP = Parish code within municipality
    
    Args:
        territory_code: Territory code like "LOCAL-010203"
        
    Returns:
        Tuple of (district_code, municipality_code, parish_code) or (None, None, None) if invalid
    """
    if not isinstance(territory_code, str) or not territory_code.startswith('LOCAL-'):
        return None, None, None
        
    try:
        parts = territory_code.split('-')
        if len(parts) != 2:
            return None, None, None
            
        code = parts[1]
        if len(code) < 4:
            return None, None, None
            
        district_code = code[:2]
        municipality_code = code[2:4]
        parish_code = code[4:] if len(code) > 4 else '00'
        
        return district_code, municipality_code, parish_code
        
    except Exception:
        return None, None, None


def create_geographic_mapping_table(election_data_file: str) -> pd.DataFrame:
    """
    Create comprehensive geographic mapping table from election data.
    
    Args:
        election_data_file: Path to parquet file with parish-level election data
        
    Returns:
        DataFrame with columns: territory_code, territory_name, district_code, 
        municipality_code, parish_code, district_name, full_municipality_id, full_parish_id
    """
    # Load raw election data
    df = pd.read_parquet(election_data_file)
    
    # Load district mapping
    district_map = pd.read_csv(os.path.join(DATA_DIR, 'legislativas_2024_codigos_circulos.csv'))
    district_dict = dict(zip(district_map['Codigo'].astype(str).str.zfill(2), district_map['Circulo']))
    
    # Create base mapping from territory codes
    mappings = []
    for _, row in df.iterrows():
        territory_code = row['territoryCode']
        territory_name = row['territoryName']
        
        district_code, municipality_code, parish_code = parse_territory_code(territory_code)
        
        if district_code is not None:
            # Look up district name
            district_name = district_dict.get(district_code, f'Unknown District {district_code}')
            
            # Create hierarchical IDs
            full_municipality_id = f"{district_code}-{municipality_code}"
            full_parish_id = f"{district_code}-{municipality_code}-{parish_code}"
            
            mappings.append({
                'territory_code': territory_code,
                'territory_name': territory_name,
                'district_code': district_code,
                'municipality_code': municipality_code, 
                'parish_code': parish_code,
                'district_name': district_name,
                'full_municipality_id': full_municipality_id,
                'full_parish_id': full_parish_id
            })
    
    mapping_df = pd.DataFrame(mappings)
    
    # Remove duplicates and sort
    mapping_df = mapping_df.drop_duplicates().sort_values(['district_code', 'municipality_code', 'parish_code'])
    
    return mapping_df


def create_municipality_aggregation_map(mapping_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Create municipality aggregation mapping from parishes.
    
    Args:
        mapping_df: Geographic mapping DataFrame
        
    Returns:
        Dict mapping municipality_id to list of territory_codes (parishes)
    """
    municipality_map = {}
    
    for _, row in mapping_df.iterrows():
        municipality_id = row['full_municipality_id']
        territory_code = row['territory_code']
        
        if municipality_id not in municipality_map:
            municipality_map[municipality_id] = []
        municipality_map[municipality_id].append(territory_code)
    
    return municipality_map


def create_district_aggregation_map(mapping_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Create district aggregation mapping from parishes.
    
    Args:
        mapping_df: Geographic mapping DataFrame
        
    Returns:
        Dict mapping district_name to list of territory_codes (parishes)
    """
    district_map = {}
    
    for _, row in mapping_df.iterrows():
        district_name = row['district_name']
        territory_code = row['territory_code']
        
        if district_name not in district_map:
            district_map[district_name] = []
        district_map[district_name].append(territory_code)
    
    return district_map


def get_municipality_name_mapping(mapping_df: pd.DataFrame) -> Dict[str, str]:
    """
    Create mapping from municipality IDs to municipality names.
    
    Args:
        mapping_df: Geographic mapping DataFrame
        
    Returns:
        Dict mapping full_municipality_id to inferred municipality name
    """
    # Group by municipality and get most common parish prefix or find pattern
    municipality_names = {}
    
    for municipality_id in mapping_df['full_municipality_id'].unique():
        municipality_parishes = mapping_df[mapping_df['full_municipality_id'] == municipality_id]
        
        # Try to infer municipality name from parish names
        parish_names = municipality_parishes['territory_name'].tolist()
        
        # Simple heuristic: if parishes share common prefixes or contain municipality name
        # For now, use the district + municipality code as identifier
        # TODO: This could be enhanced with real municipality name lookup
        district_name = municipality_parishes['district_name'].iloc[0]
        municipality_code = municipality_parishes['municipality_code'].iloc[0]
        municipality_names[municipality_id] = f"{district_name}_Municipality_{municipality_code}"
        
    return municipality_names


def validate_aggregation_consistency(
    parish_df: pd.DataFrame, 
    aggregated_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    aggregation_level: str
) -> Dict[str, any]:
    """
    Validate that aggregation preserves vote totals.
    
    Args:
        parish_df: Original parish-level data
        aggregated_df: Aggregated data (municipality or district level)
        mapping_df: Geographic mapping
        aggregation_level: 'municipality' or 'district'
        
    Returns:
        Dict with validation results and statistics
    """
    results = {
        'total_parishes': len(parish_df),
        'total_aggregated': len(aggregated_df),
        'vote_total_matches': True,
        'missing_parishes': [],
        'aggregation_errors': []
    }
    
    # Check vote total consistency - only validate numeric party columns
    exclude_columns = [
        'territoryCode', 'territoryName', 'territory_code', 'territory_name',
        'number_voters', 'percentage_voters', 'subscribed_voters', 'null_votes', 'blank_votes',
        'election_date', 'date', 'district_code', 'municipality_code', 'parish_code',
        'district_name', 'full_municipality_id', 'full_parish_id', 'geographic_id', 'sample_size'
    ]
    
    party_columns = [col for col in parish_df.columns 
                    if col not in exclude_columns 
                    and parish_df[col].dtype in ['int64', 'float64', 'int32', 'float32']
                    and not col.startswith('LOCAL-')]
    
    for party in party_columns:
        if party in parish_df.columns and party in aggregated_df.columns:
            parish_total = parish_df[party].fillna(0).sum()
            agg_total = aggregated_df[party].fillna(0).sum()
            
            if abs(parish_total - agg_total) > 1:  # Allow small rounding differences
                results['vote_total_matches'] = False
                results['aggregation_errors'].append({
                    'party': party,
                    'parish_total': parish_total,
                    'aggregated_total': agg_total,
                    'difference': parish_total - agg_total
                })
    
    return results


def save_geographic_mappings(mapping_df: pd.DataFrame, output_dir: str = None) -> None:
    """
    Save geographic mapping tables to files for reuse.
    
    Args:
        mapping_df: Geographic mapping DataFrame
        output_dir: Directory to save mappings (default: data/)
    """
    if output_dir is None:
        output_dir = DATA_DIR
    
    # Save main mapping table
    mapping_file = os.path.join(output_dir, 'geographic_mappings.csv')
    mapping_df.to_csv(mapping_file, index=False)
    
    # Save aggregation maps as separate files for easy loading
    municipality_map = create_municipality_aggregation_map(mapping_df)
    district_map = create_district_aggregation_map(mapping_df)
    municipality_names = get_municipality_name_mapping(mapping_df)
    
    # Save as JSON for easy loading
    import json
    
    with open(os.path.join(output_dir, 'municipality_aggregation_map.json'), 'w') as f:
        json.dump(municipality_map, f, indent=2)
        
    with open(os.path.join(output_dir, 'district_aggregation_map.json'), 'w') as f:
        json.dump(district_map, f, indent=2)
        
    with open(os.path.join(output_dir, 'municipality_names.json'), 'w') as f:
        json.dump(municipality_names, f, indent=2)
    
    print(f"Geographic mappings saved to {output_dir}")
    print(f"  - {len(mapping_df)} parish mappings")
    print(f"  - {len(municipality_map)} municipalities") 
    print(f"  - {len(district_map)} districts")


if __name__ == "__main__":
    # Create and save mappings from 2024 election data
    election_file = os.path.join(DATA_DIR, 'legislativas_2024.parquet')
    
    print("Creating geographic mapping tables...")
    mapping_df = create_geographic_mapping_table(election_file)
    
    print("Saving mapping tables...")
    save_geographic_mappings(mapping_df)
    
    print("Sample mappings:")
    print(mapping_df.head(10))