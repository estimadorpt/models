"""
Tests for the Coalition Management System

These tests verify the flexible coalition handling system works correctly
for both parliamentary and municipal elections with different coalition patterns.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.coalition_manager import CoalitionManager, CoalitionDefinition, TargetElectionStructure, GeographicCoalitionOverride


class TestCoalitionDefinition:
    """Test the CoalitionDefinition class."""
    
    def test_basic_coalition_definition(self):
        """Test basic coalition definition creation."""
        coalition = CoalitionDefinition(
            coalition_name='AD',
            component_parties=['PSD', 'CDS'],
            election_types=['parliamentary'],
            valid_from='2024-01-01'
        )
        
        assert coalition.coalition_name == 'AD'
        assert coalition.component_parties == ['PSD', 'CDS']
        assert coalition.election_types == ['parliamentary']
        assert coalition.valid_from == '2024-01-01'
    
    def test_coalition_validity_checks(self):
        """Test coalition validity checking logic."""
        coalition = CoalitionDefinition(
            coalition_name='AD',
            component_parties=['PSD', 'CDS'],
            election_types=['parliamentary'],
            valid_from='2024-01-01',
            valid_until='2026-12-31'
        )
        
        # Valid context
        assert coalition.is_valid_for_context('parliamentary', '2024-06-01')
        
        # Invalid election type
        assert not coalition.is_valid_for_context('municipal', '2024-06-01')
        
        # Invalid date range
        assert not coalition.is_valid_for_context('parliamentary', '2023-06-01')
        assert not coalition.is_valid_for_context('parliamentary', '2027-06-01')
    
    def test_geographic_scope_validation(self):
        """Test geographic scope validation."""
        # National coalition
        national_coalition = CoalitionDefinition(
            coalition_name='AD',
            component_parties=['PSD', 'CDS'],
            election_types=['parliamentary']
        )
        
        # Should apply everywhere if no geographic scope
        assert national_coalition.is_valid_for_context('parliamentary', '2024-01-01', None)
        assert national_coalition.is_valid_for_context('parliamentary', '2024-01-01', '01-01')
        
        # Municipal-specific coalition
        municipal_coalition = CoalitionDefinition(
            coalition_name='PSD_Local',
            component_parties=['PSD', 'LOCAL_PARTY'],
            election_types=['municipal'],
            geographic_scope='01-01'
        )
        
        # Should only apply to specific municipality
        assert municipal_coalition.is_valid_for_context('municipal', '2024-01-01', '01-01')
        assert not municipal_coalition.is_valid_for_context('municipal', '2024-01-01', '01-02')
        assert not municipal_coalition.is_valid_for_context('municipal', '2024-01-01', None)


class TestCoalitionManager:
    """Test the CoalitionManager class."""
    
    def setup_method(self):
        """Set up test coalition manager."""
        self.manager = CoalitionManager()
    
    def test_default_target_structure_loaded(self):
        """Test that default Portuguese target structure is loaded."""
        target_parties = self.manager.get_target_parties()
        
        assert 'AD' in target_parties
        assert 'CDU' in target_parties
        assert 'PS' in target_parties
    
    def test_party_aliases_work(self):
        """Test that party aliases are resolved correctly."""
        # Test basic alias resolution
        assert 'PS' in self.manager.party_aliases
        assert 'PARTIDO SOCIALISTA' in self.manager.party_aliases['PS']
        
        assert 'PSD' in self.manager.party_aliases
        assert 'PPD/PSD' in self.manager.party_aliases['PSD']
    
    def test_add_geographic_override(self):
        """Test adding geographic coalition overrides."""
        initial_overrides = len(self.manager.geographic_overrides)
        
        # Add a municipal coalition override
        self.manager.add_municipal_coalition_override('01-01', {
            'LOCAL_ALLIANCE': ['PSD', 'INDEPENDENT']
        })
        
        assert len(self.manager.geographic_overrides) == initial_overrides + 1
        
        # Test it's retrievable
        target_parties = self.manager.get_target_parties('01-01')
        assert 'LOCAL_ALLIANCE' in target_parties
    
    def test_component_party_resolution(self):
        """Test component party resolution for target coalitions."""
        # Test AD coalition components
        ad_components = self.manager.get_component_parties_for_target('AD')
        assert 'PSD' in ad_components
        assert 'CDS' in ad_components
        
        # Test CDU coalition components  
        cdu_components = self.manager.get_component_parties_for_target('CDU')
        assert 'PCP' in cdu_components
        assert 'PEV' in cdu_components
        
        # Test single party
        ps_components = self.manager.get_component_parties_for_target('PS')
        assert ps_components == ['PS']
    
    def test_historical_column_resolution(self):
        """Test resolving historical columns to target parties."""
        available_columns = ['PS', 'PPD/PSD.CDS-PP', 'CH', 'IL', 'BE', 'PCP-PEV']
        
        # Test direct single party match
        result = self.manager.resolve_historical_column_to_target('PS', available_columns)
        assert result == 'PS'
        
        # Test coalition resolution - CDU should match PCP-PEV
        result = self.manager.resolve_historical_column_to_target('CDU', available_columns)
        assert result == 'PCP-PEV'
        
        # Test coalition resolution - AD should match PPD/PSD.CDS-PP
        result = self.manager.resolve_historical_column_to_target('AD', available_columns)
        assert result == 'PPD/PSD.CDS-PP'
    
    def test_election_data_mapping(self):
        """Test mapping real election data columns."""
        # Create sample data similar to Portuguese election results
        sample_data = pd.DataFrame({
            'territoryCode': ['LOCAL-010101', 'LOCAL-010102'],
            'territoryName': ['Parish 1', 'Parish 2'],
            'PS': [1000, 800],
            'PPD/PSD.CDS-PP': [900, 700],
            'CH': [300, 250],
            'IL': [200, 150],
            'B.E.': [150, 100],
            'PCP-PEV': [100, 80],
            'PAN': [50, 40],
            'L': [30, 25]
        })
        
        political_families = ['PS', 'AD', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L']
        
        # Map using coalition manager
        mapped_data = self.manager.map_election_data_columns(
            sample_data,
            political_families,
            election_type='parliamentary',
            election_date='2024-03-10'
        )
        
        # Check all political families are present
        for party in political_families:
            assert party in mapped_data.columns
        
        # Check specific mappings
        assert mapped_data['AD'].iloc[0] == 900  # Mapped from PPD/PSD.CDS-PP
        assert mapped_data['BE'].iloc[0] == 150  # Mapped from B.E.
        assert mapped_data['CDU'].iloc[0] == 100  # Mapped from PCP-PEV
        assert mapped_data['PS'].iloc[0] == 1000  # Direct match
    
    def test_municipal_coalition_specificity(self):
        """Test that municipal coalition overrides work correctly."""
        # Add a municipality-specific coalition override
        self.manager.add_municipal_coalition_override('01-01', {
            'LOCAL_ALLIANCE': ['PSD', 'INDEPENDENT_GROUP']
        })
        
        # Test it applies to the specific municipality
        municipal_parties = self.manager.get_target_parties('01-01')
        assert 'LOCAL_ALLIANCE' in municipal_parties
        
        # Test component resolution
        components = self.manager.get_component_parties_for_target('LOCAL_ALLIANCE', '01-01')
        assert 'PSD' in components
        assert 'INDEPENDENT_GROUP' in components
        
        # Test it doesn't apply to other municipalities
        other_parties = self.manager.get_target_parties('01-02')
        assert 'LOCAL_ALLIANCE' not in other_parties
    
    def test_target_structure_consistency(self):
        """Test that target structure remains consistent regardless of time."""
        # In the new approach, target structure doesn't change with time
        # It's always the same for the target election being modeled
        
        # Test that target parties are consistent
        parties_2020 = self.manager.get_target_parties()
        parties_2024 = self.manager.get_target_parties()
        
        assert parties_2020 == parties_2024
        
        # Test that AD coalition always has same components for modeling consistency
        components_2020 = self.manager.get_component_parties_for_target('AD')
        components_2024 = self.manager.get_component_parties_for_target('AD')
        
        assert components_2020 == components_2024
        assert 'PSD' in components_2020
        assert 'CDS' in components_2020


class TestCoalitionIntegration:
    """Test integration with existing election data loading."""
    
    def test_backward_compatibility(self):
        """Test that existing AD coalition handling still works."""
        manager = CoalitionManager()
        
        # Test that AD coalition is still defined in target structure
        target_parties = manager.get_target_parties()
        assert 'AD' in target_parties
        
        # Test component parties
        ad_components = manager.get_component_parties_for_target('AD')
        assert 'PSD' in ad_components
        assert 'CDS' in ad_components
    
    def test_election_type_differentiation(self):
        """Test that geographic areas can have different coalition overrides."""
        manager = CoalitionManager()
        
        # Add different municipal coalition override
        manager.add_municipal_coalition_override('11-01', {  # Lisboa
            'LISBOA_ALLIANCE': ['PS', 'BE', 'L']
        })
        
        # National should have standard structure
        national_parties = manager.get_target_parties()
        assert 'AD' in national_parties
        assert 'LISBOA_ALLIANCE' not in national_parties
        
        # Lisboa should have local alliance added
        lisboa_parties = manager.get_target_parties('11-01')
        assert 'AD' in lisboa_parties  # Still has national parties
        assert 'LISBOA_ALLIANCE' in lisboa_parties  # Plus local override


if __name__ == "__main__":
    # Run the tests if executed directly
    pytest.main([__file__, '-v'])