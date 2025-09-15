"""
Coalition Management System for Portuguese Elections

This module handles coalition politics with a TARGET-ELECTION-DRIVEN approach
that ensures modeling consistency across time periods.

Key Principle: **Bayesian models require stable party coordinates**
- Start with target election's party structure
- Project that structure backwards through all historical data  
- Combine component parties retroactively for consistent time series
- No artificial discontinuities from coalition formation dates

Example: For 2026 election with AD coalition:
- 2026: AD (target structure)
- 2024: AD (from actual data) 
- 2022: AD = PSD + CDS (combined retroactively)
- 2011: AD = PSD + CDS (combined retroactively)

This creates stable AD time series for proper Bayesian trend modeling.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field
from datetime import datetime
from src.config import DATA_DIR


# Legacy compatibility alias for tests
@dataclass
class CoalitionDefinition:
    """Legacy coalition definition class for backward compatibility with tests."""
    coalition_name: str
    component_parties: List[str]
    election_types: List[str] = field(default_factory=list)
    geographic_scope: Optional[str] = None
    valid_from: str = '2000-01-01'
    valid_until: Optional[str] = None
    priority: int = 10
    
    def is_valid_for_context(self, election_type: str, election_date: str, 
                           geographic_id: Optional[str] = None) -> bool:
        """Check if coalition is valid for given context (legacy compatibility)."""
        # Election type check
        if self.election_types and election_type not in self.election_types and 'all' not in self.election_types:
            return False
        
        # Date range check
        if self.valid_from and election_date < self.valid_from:
            return False
        if self.valid_until and election_date > self.valid_until:
            return False
        
        # Geographic scope check
        if self.geographic_scope and geographic_id != self.geographic_scope:
            return False
        
        return True


@dataclass
class TargetElectionStructure:
    """Defines the party structure for the target election being modeled."""
    
    coalitions: Dict[str, List[str]]  # coalition_name -> component_parties
    election_type: str = 'parliamentary'  # 'parliamentary', 'municipal', etc.
    geographic_scope: Optional[str] = None  # None=national, or specific municipality
    
    def get_target_parties(self) -> List[str]:
        """Get all party names in the target structure (coalitions + individual parties)."""
        return list(self.coalitions.keys())
    
    def get_component_parties_for(self, coalition_name: str) -> List[str]:
        """Get component parties for a given coalition."""
        return self.coalitions.get(coalition_name, [coalition_name])
    
    def is_coalition(self, party_name: str) -> bool:
        """Check if a party name represents a coalition."""
        return party_name in self.coalitions and len(self.coalitions[party_name]) > 1


@dataclass 
class GeographicCoalitionOverride:
    """Defines coalition override for specific geographic areas."""
    
    geographic_id: str
    coalition_overrides: Dict[str, List[str]]  # coalition_name -> different_component_parties
    
    def applies_to(self, geographic_id: Optional[str]) -> bool:
        """Check if this override applies to the given geographic area."""
        return geographic_id == self.geographic_id


class CoalitionManager:
    """
    Manages coalition mapping for consistent Bayesian modeling.
    
    Uses TARGET-ELECTION-DRIVEN approach:
    - Accepts target election's party structure
    - Projects coalitions backwards through all historical data
    - Creates stable party coordinates for Bayesian models
    - Prioritizes modeling consistency over historical accuracy
    """
    
    def __init__(self, target_structure: Optional[TargetElectionStructure] = None):
        """Initialize coalition manager with target election structure."""
        self.target_structure = target_structure or self._get_default_structure()
        self.geographic_overrides: List[GeographicCoalitionOverride] = []
        self.party_aliases: Dict[str, List[str]] = {}
        
        # Load standard party aliases for flexible matching
        self._load_party_aliases()
    
    def _get_default_structure(self) -> TargetElectionStructure:
        """Get default Portuguese parliamentary structure for 2024+ elections."""
        return TargetElectionStructure(
            coalitions={
                'PS': ['PS'],
                'AD': ['PSD', 'CDS'],  # Key coalition for stable modeling
                'CH': ['CH'],
                'IL': ['IL'],
                'BE': ['BE'],
                'CDU': ['PCP', 'PEV'],  # Historical communist coalition
                'PAN': ['PAN'],
                'L': ['L']
            },
            election_type='parliamentary'
        )
    
    def _load_party_aliases(self):
        """Load standard party aliases for flexible matching across elections."""
        self.party_aliases = {
            'PS': ['PS', 'PARTIDO SOCIALISTA'],
            'PSD': ['PSD', 'PPD/PSD', 'PPD', 'PARTIDO SOCIAL DEMOCRATA'],
            'CDS': ['CDS-PP', 'CDS', 'CENTRO DEMOCRÁTICO SOCIAL'],
            'CH': ['CH', 'CHEGA'],
            'IL': ['IL', 'INICIATIVA LIBERAL'],
            'BE': ['BE', 'B.E.', 'BLOCO DE ESQUERDA'],
            'PCP': ['PCP', 'PARTIDO COMUNISTA PORTUGUÊS'],
            'PEV': ['PEV', 'PARTIDO ECOLOGISTA OS VERDES'],
            'PAN': ['PAN', 'PESSOAS-ANIMAIS-NATUREZA'],
            'L': ['L', 'LIVRE', 'L/TDA']
        }
    
    def add_geographic_override(self, override: GeographicCoalitionOverride):
        """Add a geographic-specific coalition override."""
        self.geographic_overrides.append(override)
    
    def get_target_parties(self, geographic_id: Optional[str] = None) -> List[str]:
        """Get target party names (coalitions + individual parties) for modeling."""
        # Check for geographic overrides first
        for override in self.geographic_overrides:
            if override.applies_to(geographic_id):
                # Merge base structure with overrides
                merged_coalitions = {**self.target_structure.coalitions, **override.coalition_overrides}
                return list(merged_coalitions.keys())
        
        return self.target_structure.get_target_parties()
    
    def get_component_parties_for_target(self, target_party: str, geographic_id: Optional[str] = None) -> List[str]:
        """Get component parties for a target party, considering geographic overrides."""
        # Check for geographic overrides first
        for override in self.geographic_overrides:
            if override.applies_to(geographic_id) and target_party in override.coalition_overrides:
                return override.coalition_overrides[target_party]
        
        return self.target_structure.get_component_parties_for(target_party)
    
    def resolve_historical_column_to_target(self, target_party: str, available_columns: List[str],
                                          geographic_id: Optional[str] = None) -> Optional[str]:
        """
        Resolve historical data columns to target party structure.
        
        This is the core method that projects target coalitions backwards through time.
        For AD (target coalition), it will find PSD+CDS, PPD/PSD.CDS-PP, etc.
        """
        # Get component parties for this target
        component_parties = self.get_component_parties_for_target(target_party, geographic_id)
        
        # If single party, look for direct matches via aliases
        if len(component_parties) == 1:
            return self._find_party_column(component_parties[0], available_columns)
        
        # If coalition, look for:
        # 1. Direct coalition column (e.g., "AD", "PPD/PSD.CDS-PP")
        coalition_column = self._find_coalition_column_by_components(component_parties, available_columns)
        if coalition_column:
            return coalition_column
        
        # 2. Individual component columns that need to be combined
        # This would require summing multiple columns - return None for now
        # TODO: Implement multi-column combination logic
        return None
    
    def map_election_data_columns(self, df: pd.DataFrame, 
                                 political_families: List[str],
                                 election_type: str = 'parliamentary',
                                 election_date: str = '2024-01-01',
                                 geographic_id: Optional[str] = None) -> pd.DataFrame:
        """
        Map raw election data to target party structure for consistent Bayesian modeling.
        
        Projects target election coalitions backwards through historical data.
        Example: AD target party matches PPD/PSD.CDS-PP in 2022 data.
        """
        df = df.copy()
        available_columns = df.columns.tolist()
        
        print(f"  Target-driven coalition mapping for {election_type} election on {election_date}")
        if geographic_id:
            print(f"    Geographic context: {geographic_id}")
        
        # Get target parties for this geographic context
        target_parties = self.get_target_parties(geographic_id)
        print(f"    Target party structure: {target_parties}")
        
        # Map each requested political family to target structure
        for family in political_families:
            if family in df.columns:
                continue  # Already exists
            
            # Check if this family is in our target structure
            if family not in target_parties:
                print(f"    Warning: {family} not in target structure, creating with zeros")
                df[family] = 0.0
                continue
                
            # Resolve historical data to this target party
            historical_column = self.resolve_historical_column_to_target(family, available_columns, geographic_id)
            if historical_column:
                df[family] = df[historical_column].fillna(0)
                print(f"    Mapped {historical_column} -> {family} (target projection)")
                continue
            
            # Create with zeros if no historical match found
            df[family] = 0.0
            print(f"    Created {family} with zeros (no historical data found)")
        
        return df
    
    def _find_party_column(self, party: str, available_columns: List[str]) -> Optional[str]:
        """Find column for individual party using aliases."""
        if party in self.party_aliases:
            for alias in self.party_aliases[party]:
                for col in available_columns:
                    if alias == col or alias in col:
                        return col
        return None
    
    def _find_coalition_column_by_components(self, component_parties: List[str], 
                                           available_columns: List[str]) -> Optional[str]:
        """Find column representing a coalition of specific component parties."""
        # Direct matches for coalition patterns
        coalition_patterns = [
            # Component parties joined with common separators
            '.'.join(component_parties),
            '-'.join(component_parties), 
            '/'.join(component_parties),
            '+'.join(component_parties),
            # Try reversed order too
            '.'.join(reversed(component_parties)),
            '-'.join(reversed(component_parties)),
            '/'.join(reversed(component_parties))
        ]
        
        for pattern in coalition_patterns:
            for col in available_columns:
                if pattern in col:
                    return col
        
        # Look for columns containing all component party names (using aliases)
        for col in available_columns:
            matched_parties = 0
            for component in component_parties:
                if component in self.party_aliases:
                    # Check if any alias of this component is in the column
                    if any(alias in col for alias in self.party_aliases[component]):
                        matched_parties += 1
                elif component in col:
                    matched_parties += 1
            
            # If all component parties are represented in this column
            if matched_parties == len(component_parties):
                return col
        
        return None
    
    def add_municipal_coalition_override(self, municipality_id: str, coalition_overrides: Dict[str, List[str]]):
        """Add municipality-specific coalition overrides to target structure."""
        override = GeographicCoalitionOverride(
            geographic_id=municipality_id,
            coalition_overrides=coalition_overrides
        )
        self.add_geographic_override(override)
    
    def load_configuration(self, config_file: str):
        """Load target election structure configuration from JSON file."""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Load target structure
        if 'target_structure' in config:
            self.target_structure = TargetElectionStructure(**config['target_structure'])
        
        # Load geographic overrides
        if 'geographic_overrides' in config:
            for override_data in config['geographic_overrides']:
                override = GeographicCoalitionOverride(**override_data)
                self.add_geographic_override(override)
        
        # Load additional party aliases
        if 'party_aliases' in config:
            for party, aliases in config['party_aliases'].items():
                if party in self.party_aliases:
                    self.party_aliases[party].extend(aliases)
                else:
                    self.party_aliases[party] = aliases
    
    def save_configuration(self, config_file: str):
        """Save current target structure configuration to JSON file."""
        config = {
            'target_structure': {
                'coalitions': self.target_structure.coalitions,
                'election_type': self.target_structure.election_type,
                'geographic_scope': self.target_structure.geographic_scope
            },
            'geographic_overrides': [
                {
                    'geographic_id': override.geographic_id,
                    'coalition_overrides': override.coalition_overrides
                }
                for override in self.geographic_overrides
            ],
            'party_aliases': self.party_aliases
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


def get_default_coalition_manager(target_structure: Optional[TargetElectionStructure] = None) -> CoalitionManager:
    """Get the default coalition manager instance."""
    manager = CoalitionManager(target_structure)
    
    # Try to load additional configuration if available
    config_file = os.path.join(DATA_DIR, 'coalition_config.json')
    if os.path.exists(config_file):
        manager.load_configuration(config_file)
    
    return manager


# Example usage and testing
if __name__ == "__main__":
    # Demonstrate target-election-driven coalition management
    print("=== Target-Election-Driven Coalition Management Demo ===\n")
    
    # Create 2026 election target structure
    target_2026 = TargetElectionStructure(
        coalitions={
            'PS': ['PS'],
            'AD': ['PSD', 'CDS'],  # Target coalition for consistent modeling
            'CH': ['CH'],
            'IL': ['IL'],
            'BE': ['BE'],
            'CDU': ['PCP', 'PEV'],
            'PAN': ['PAN'],
            'L': ['L']
        },
        election_type='parliamentary'
    )
    
    manager = CoalitionManager(target_2026)
    
    print(f"Target Election Structure (2026): {target_2026.get_target_parties()}")
    print("\nComponent parties for key coalitions:")
    for party in ['AD', 'CDU']:
        components = manager.get_component_parties_for_target(party)
        print(f"  {party}: {components}")
    
    # Demonstrate backward projection
    print("\n=== Backward Projection Examples ===")
    
    # Simulate 2022 election data with different column names
    sample_2022_columns = ['PS', 'PPD/PSD.CDS-PP', 'CH', 'IL', 'BE', 'PCP-PEV', 'PAN', 'L']
    print(f"\n2022 Election Data Columns: {sample_2022_columns}")
    
    for target_party in ['AD', 'CDU']:
        historical_match = manager.resolve_historical_column_to_target(target_party, sample_2022_columns)
        components = manager.get_component_parties_for_target(target_party)
        print(f"  {target_party} (target) <- {historical_match} (2022 data)")
        print(f"    Target components: {components}")
    
    # Add municipal override
    print("\n=== Municipal Override Example ===")
    manager.add_municipal_coalition_override('11-01', {  # Lisboa
        'LOCAL_ALLIANCE': ['PS', 'BE', 'L']  # Different coalition in Lisboa
    })
    
    # Show how target parties change by geography
    national_parties = manager.get_target_parties()
    lisboa_parties = manager.get_target_parties('11-01')
    
    print(f"National target parties: {national_parties}")
    print(f"Lisboa target parties: {lisboa_parties}")
    
    print("\n=== Bayesian Modeling Principle Demonstrated ===")
    print("Key insight: AD voters in 2026 are the same political family as:")
    print("  - PPD/PSD.CDS-PP voters in 2022")
    print("  - Combined PSD+CDS voters in 2011")
    print("  - This creates stable coordinates for time series modeling")