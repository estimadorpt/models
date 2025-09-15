"""
Tests for the Coalition Management System

These tests verify the flexible coalition handling system works correctly
for both parliamentary and municipal elections with different coalition patterns.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.coalition_manager import (
    CoalitionManager, CoalitionDefinition, TargetElectionStructure, 
    GeographicCoalitionOverride, DisaggregationRule, MunicipalCoalitionStructure
)


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


class TestMunicipalDisaggregation:
    """Test municipal coalition disaggregation functionality."""
    
    def test_disaggregation_rule(self):
        """Test basic disaggregation of national coalitions."""
        rule = DisaggregationRule(
            national_coalition='AD',
            component_parties={'PSD': 0.85, 'CDS': 0.15}
        )
        
        # Test disaggregation
        result = rule.disaggregate(0.30)  # 30% AD support
        
        assert result['PSD'] == pytest.approx(0.255)  # 25.5%
        assert result['CDS'] == pytest.approx(0.045)  # 4.5%
    
    def test_municipal_coalition_structure_basic(self):
        """Test basic municipal coalition structure functionality."""
        ad_rule = DisaggregationRule('AD', {'PSD': 0.85, 'CDS': 0.15})
        
        # Aveiro structure (PSD vs CDS competing)
        aveiro = MunicipalCoalitionStructure(
            municipality_id='01-01',
            local_coalitions={
                'PSD': ['PSD'],
                'CDS': ['CDS'],
                'PS': ['PS'],
                'IL': ['IL']
            },
            disaggregation_rules=[ad_rule]
        )
        
        national_predictions = {'PS': 0.35, 'AD': 0.30, 'IL': 0.08}
        municipal_predictions = aveiro.propagate_national_predictions(national_predictions)
        
        # Check disaggregation worked
        assert 'AD' not in municipal_predictions  # AD should be disaggregated
        assert municipal_predictions['PSD'] == pytest.approx(0.255)
        assert municipal_predictions['CDS'] == pytest.approx(0.045)
        assert municipal_predictions['PS'] == pytest.approx(0.35)
        assert municipal_predictions['IL'] == pytest.approx(0.08)
    
    def test_municipal_coalition_structure_aggregation(self):
        """Test municipal coalition aggregation (combining parties)."""
        ad_rule = DisaggregationRule('AD', {'PSD': 0.85, 'CDS': 0.15})
        
        # Lisboa structure (PSD+IL coalition)
        lisboa = MunicipalCoalitionStructure(
            municipality_id='11-01',
            local_coalitions={
                'PSD_IL': ['PSD', 'IL'],  # Combine PSD and IL
                'PS': ['PS'],
                'CDS': ['CDS']
            },
            disaggregation_rules=[ad_rule]
        )
        
        national_predictions = {'PS': 0.35, 'AD': 0.30, 'IL': 0.08}
        municipal_predictions = lisboa.propagate_national_predictions(national_predictions)
        
        # Check aggregation worked
        assert municipal_predictions['PSD_IL'] == pytest.approx(0.335)  # 0.255 + 0.08
        assert municipal_predictions['PS'] == pytest.approx(0.35)
        assert municipal_predictions['CDS'] == pytest.approx(0.045)
        assert 'PSD' not in municipal_predictions  # Should be absorbed into coalition
        assert 'IL' not in municipal_predictions   # Should be absorbed into coalition
    
    def test_municipal_coalition_structure_recombination(self):
        """Test recombining disaggregated parties (rural AD pattern)."""
        ad_rule = DisaggregationRule('AD', {'PSD': 0.85, 'CDS': 0.15})
        
        # Rural structure (traditional AD coalition)
        rural = MunicipalCoalitionStructure(
            municipality_id='rural',
            local_coalitions={
                'AD': ['PSD', 'CDS'],  # Recombine after disaggregation
                'PS': ['PS'],
                'IL': ['IL']
            },
            disaggregation_rules=[ad_rule]
        )
        
        national_predictions = {'PS': 0.35, 'AD': 0.30, 'IL': 0.08}
        municipal_predictions = rural.propagate_national_predictions(national_predictions)
        
        # Check recombination worked
        assert municipal_predictions['AD'] == pytest.approx(0.30)  # Back to original
        assert municipal_predictions['PS'] == pytest.approx(0.35)
        assert municipal_predictions['IL'] == pytest.approx(0.08)
        assert 'PSD' not in municipal_predictions  # Should be absorbed back into AD
        assert 'CDS' not in municipal_predictions  # Should be absorbed back into AD


class TestCoalitionManagerMunicipal:
    """Test municipal functionality of CoalitionManager."""
    
    def setup_method(self):
        """Set up test coalition manager."""
        self.manager = CoalitionManager()
    
    def test_municipal_structure_loading(self):
        """Test that municipal structures are loaded correctly."""
        # Test Aveiro structure
        aveiro_structure = self.manager.get_municipal_structure('01-01')
        assert aveiro_structure.municipality_id == '01-01'
        assert 'PSD' in aveiro_structure.local_coalitions
        assert 'CDS' in aveiro_structure.local_coalitions
        
        # Test Lisboa structure
        lisboa_structure = self.manager.get_municipal_structure('11-01')
        assert 'PSD_IL' in lisboa_structure.local_coalitions
        
        # Test fallback to default
        unknown_structure = self.manager.get_municipal_structure('99-99')
        assert unknown_structure.municipality_id == 'default'
        assert 'AD' in unknown_structure.local_coalitions
    
    def test_end_to_end_propagation(self):
        """Test full national-to-municipal prediction propagation."""
        national_polls = {
            'PS': 0.35,
            'AD': 0.30,
            'IL': 0.08,
            'CH': 0.12
        }
        
        # Test Aveiro (competing PSD vs CDS)
        aveiro_predictions = self.manager.propagate_national_to_municipal(national_polls, '01-01')
        assert aveiro_predictions['PSD'] == pytest.approx(0.255)
        assert aveiro_predictions['CDS'] == pytest.approx(0.045)
        assert aveiro_predictions['IL'] == pytest.approx(0.08)
        assert 'AD' not in aveiro_predictions
        
        # Test Lisboa (PSD+IL coalition)
        lisboa_predictions = self.manager.propagate_national_to_municipal(national_polls, '11-01')
        assert lisboa_predictions['PSD_IL'] == pytest.approx(0.335)
        assert 'PSD' not in lisboa_predictions
        assert 'IL' not in lisboa_predictions
        
        # Test rural (traditional AD)
        rural_predictions = self.manager.propagate_national_to_municipal(national_polls, 'rural-99')
        assert rural_predictions['AD'] == pytest.approx(0.30)
        assert 'PSD' not in rural_predictions
        assert 'CDS' not in rural_predictions
    
    def test_municipal_vs_national_coordinate_spaces(self):
        """Test that municipal and national coordinate spaces are properly separated."""
        national_polls = {'PS': 0.35, 'AD': 0.30, 'IL': 0.08}
        
        # National space should have AD
        national_parties = self.manager.get_target_parties()
        assert 'AD' in national_parties
        
        # Municipal spaces should transform AD appropriately
        aveiro_predictions = self.manager.propagate_national_to_municipal(national_polls, '01-01')
        lisboa_predictions = self.manager.propagate_national_to_municipal(national_polls, '11-01')
        
        # Different municipal coordinate spaces
        assert set(aveiro_predictions.keys()) != set(lisboa_predictions.keys())
        
        # But vote shares should be conserved (approximately)
        aveiro_total = sum(aveiro_predictions.values())
        lisboa_total = sum(lisboa_predictions.values())
        national_total = sum(national_polls.values())
        
        assert aveiro_total == pytest.approx(national_total, abs=0.01)
        assert lisboa_total == pytest.approx(national_total, abs=0.01)


if __name__ == "__main__":
    # Run the tests if executed directly
    pytest.main([__file__, '-v'])