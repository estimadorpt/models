"""
Flexible data loaders that provide backward compatibility while enabling multi-level aggregation.

This module acts as a bridge between the existing system and the new multi-level
geographic aggregation capabilities, ensuring no breaking changes.
"""

import pandas as pd
from typing import Dict, List, Optional, Literal
from src.data.multi_level_loaders import load_multi_level_results
from src.data.loaders import load_district_config as original_load_district_config


def load_election_results_flexible(
    election_dates: List[str], 
    political_families: List[str], 
    aggregate_national: bool = True,
    aggregation_level: Optional[Literal['parish', 'municipality', 'district', 'national']] = None,
    election_type: str = 'parliamentary',
    coalition_manager = None
) -> pd.DataFrame:
    """
    Load election results with flexible geographic aggregation.
    
    This function provides backward compatibility with the existing load_election_results
    while adding support for parish and municipality-level data.
    
    Args:
        election_dates: List of election dates
        political_families: List of party names
        aggregate_national: If True and no aggregation_level specified, returns national results
                          If False and no aggregation_level specified, returns district results
        aggregation_level: Explicit aggregation level ('parish', 'municipality', 'district', 'national')
                          Overrides aggregate_national parameter
        election_type: Type of election ('parliamentary', 'municipal', etc.)
        coalition_manager: Coalition manager for intelligent party mapping
        
    Returns:
        DataFrame with election results at specified aggregation level
    """
    # Determine aggregation level
    if aggregation_level is None:
        aggregation_level = 'national' if aggregate_national else 'district'
    
    print(f"Loading election results with flexible aggregation: {aggregation_level} level")
    
    # Use new multi-level loading system
    results_df = load_multi_level_results(
        election_dates=election_dates,
        political_families=political_families,
        aggregate_to=aggregation_level,
        validate_totals=True,
        election_type=election_type,
        coalition_manager=coalition_manager
    )
    
    if results_df.empty:
        return results_df
    
    # Add backward compatibility columns for existing system
    if aggregation_level == 'district':
        # Add 'Circulo' column for backward compatibility
        results_df['Circulo'] = results_df['geographic_id']
        
    elif aggregation_level == 'municipality':
        # Add municipality-specific columns
        results_df['municipality_id'] = results_df['geographic_id']
        
    elif aggregation_level == 'parish':
        # Add parish-specific columns  
        results_df['territory_code'] = results_df['geographic_id']
    
    # Ensure required columns exist for compatibility
    if 'sample_size' not in results_df.columns:
        # Calculate sample size as sum of valid votes
        valid_vote_cols = [col for col in political_families if col in results_df.columns]
        if valid_vote_cols:
            results_df['sample_size'] = results_df[valid_vote_cols].sum(axis=1)
        else:
            results_df['sample_size'] = 0
    
    return results_df


def load_geographic_config(
    level: Literal['district', 'municipality', 'parish'] = 'district'
) -> Dict[str, int]:
    """
    Load geographic division configuration (seats, population, etc.).
    
    Args:
        level: Geographic level to load configuration for
        
    Returns:
        Dict mapping geographic division names to seat counts or other config
    """
    if level == 'district':
        # Use existing district configuration for backward compatibility
        return original_load_district_config()
    
    elif level == 'municipality':
        # For municipalities, we need to implement seat allocation
        # This would vary by municipality size, type of election, etc.
        # For now, return a placeholder that assumes 1 mayor per municipality
        
        # Load municipality data to get list
        sample_results = load_multi_level_results(
            election_dates=['2024-03-10'], 
            political_families=['PS'], 
            aggregate_to='municipality'
        )
        
        if not sample_results.empty:
            municipalities = sample_results['geographic_id'].unique()
            # For municipal elections: 1 mayor per municipality
            # Council seats would vary by municipality size (TODO: implement proper lookup)
            return {municipality: 1 for municipality in municipalities}
        else:
            return {}
            
    elif level == 'parish':
        # Parishes don't typically have separate seat allocations
        # They're just geographic units for data collection
        return {}
    
    else:
        raise ValueError(f"Invalid geographic level: {level}")


def create_district_mapping_compatibility() -> Dict[str, str]:
    """
    Create mapping from old district system to new geographic IDs.
    
    Returns:
        Dict mapping old district names to new geographic IDs for compatibility
    """
    # Load district results to create mapping
    district_results = load_multi_level_results(
        election_dates=['2024-03-10'],
        political_families=['PS'],
        aggregate_to='district'
    )
    
    if district_results.empty:
        return {}
    
    # Create mapping from district name to geographic_id (which is also district name)
    return dict(zip(district_results['geographic_id'], district_results['geographic_id']))


class GeographicLevelManager:
    """
    Manager class for handling different geographic levels in the election system.
    
    This class provides a unified interface for working with parishes, municipalities,
    and districts while maintaining backward compatibility.
    """
    
    def __init__(self, default_level: Literal['parish', 'municipality', 'district'] = 'district'):
        self.default_level = default_level
        self._cached_configs = {}
    
    def get_available_levels(self, election_dates: List[str]) -> Dict[str, int]:
        """Get count of geographic divisions available for given elections."""
        from src.data.multi_level_loaders import get_available_geographic_levels
        return get_available_geographic_levels(election_dates)
    
    def load_results(
        self, 
        election_dates: List[str], 
        political_families: List[str],
        level: Optional[str] = None
    ) -> pd.DataFrame:
        """Load results at specified geographic level."""
        if level is None:
            level = self.default_level
            
        return load_election_results_flexible(
            election_dates=election_dates,
            political_families=political_families,
            aggregation_level=level
        )
    
    def get_config(self, level: Optional[str] = None) -> Dict[str, int]:
        """Get configuration for specified geographic level."""
        if level is None:
            level = self.default_level
            
        if level not in self._cached_configs:
            self._cached_configs[level] = load_geographic_config(level)
            
        return self._cached_configs[level]
    
    def create_coordinates(self, election_dates: List[str], level: Optional[str] = None) -> Dict[str, List[str]]:
        """Create coordinate lists for PyMC models."""
        if level is None:
            level = self.default_level
            
        from src.data.multi_level_loaders import create_geographic_coordinates
        return create_geographic_coordinates(level, election_dates)


def demonstrate_flexibility():
    """Demonstrate the flexible aggregation capabilities."""
    
    test_elections = ['2024-03-10', '2022-01-30']
    test_parties = ['PS', 'AD', 'CH', 'IL']
    
    print("=== Flexible Geographic Aggregation Demo ===\n")
    
    # Test different aggregation levels
    levels = ['national', 'district', 'municipality', 'parish']
    
    for level in levels:
        print(f"--- {level.upper()} LEVEL ---")
        
        try:
            results = load_election_results_flexible(
                election_dates=test_elections,
                political_families=test_parties,
                aggregation_level=level
            )
            
            if not results.empty:
                print(f"Shape: {results.shape}")
                print(f"Geographic divisions: {results['geographic_id'].nunique()}")
                print("Sample data:")
                print(results[['election_date', 'geographic_id'] + test_parties[:2]].head(3))
                
                # Show vote totals for validation
                total_votes = results[test_parties].sum().sum()
                print(f"Total votes across all parties: {total_votes:,}")
            else:
                print("No data loaded")
                
        except Exception as e:
            print(f"Error loading {level} level: {e}")
        
        print()
    
    # Test geographic level manager
    print("--- GEOGRAPHIC LEVEL MANAGER ---")
    manager = GeographicLevelManager(default_level='municipality')
    
    available_levels = manager.get_available_levels(test_elections)
    print("Available geographic levels:")
    for level, count in available_levels.items():
        print(f"  {level}: {count:,} divisions")
    
    print("\nMunicipality configuration sample:")
    muni_config = manager.get_config('municipality')
    sample_munis = list(muni_config.items())[:5]
    for muni_id, seats in sample_munis:
        print(f"  {muni_id}: {seats} seats")


if __name__ == "__main__":
    demonstrate_flexibility()