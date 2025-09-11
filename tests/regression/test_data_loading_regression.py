"""
Regression tests for data loading pipeline.

These tests ensure the data loading pipeline produces consistent, deterministic outputs.
They capture the current behavior as the "correct" behavior to prevent regressions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from data.dataset import ElectionDataset
from data.loaders import (
    load_marktest_polls,
    load_election_results,
    load_district_config
)


class TestDataLoadingRegression:
    """Test suite to ensure data loading behavior remains consistent."""
    
    def test_marktest_polls_structure(self):
        """Test that poll loading produces expected DataFrame structure."""
        polls = load_marktest_polls()
        
        # Test basic structure
        assert isinstance(polls, pd.DataFrame)
        assert len(polls) > 0, "Should load some poll data"
        
        # Test expected columns exist
        expected_columns = [
            'date', 'pollster', 'sample_size', 
            'PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD'
        ]
        for col in expected_columns:
            assert col in polls.columns, f"Missing expected column: {col}"
        
        # Test data types are consistent
        assert pd.api.types.is_datetime64_any_dtype(polls['date'])
        assert pd.api.types.is_numeric_dtype(polls['sample_size'])
        
        # Test party vote shares are numeric and within reasonable bounds
        party_columns = ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
        for party in party_columns:
            assert pd.api.types.is_numeric_dtype(polls[party])
            # Check bounds (allowing for some flexibility in data)
            assert polls[party].min() >= 0, f"{party} has negative values"
            assert polls[party].max() <= 1, f"{party} has values > 100%"
    
    def test_election_results_structure(self):
        """Test that election results loading produces expected structure."""
        historical_dates = [
            '2024-03-10', '2022-01-30', '2019-10-06', '2015-10-04', '2011-06-05'
        ]
        party_columns = ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
        
        # Test national aggregation
        results_national = load_election_results(
            historical_dates, party_columns, aggregate_national=True
        )
        
        assert isinstance(results_national, pd.DataFrame)
        assert len(results_national) > 0, "Should load election results"
        
        # Test expected columns
        expected_columns = ['election_date', 'date'] + party_columns + ['sample_size']
        for col in expected_columns:
            assert col in results_national.columns, f"Missing column: {col}"
        
        # Test district-level data
        results_district = load_election_results(
            historical_dates, party_columns, aggregate_national=False
        )
        
        assert isinstance(results_district, pd.DataFrame)
        assert len(results_district) > 0, "Should load district results"
        assert 'Circulo' in results_district.columns, "District results should have Circulo column"
    
    def test_district_config_loading(self):
        """Test that district configuration loads correctly."""
        district_config = load_district_config()
        
        assert isinstance(district_config, dict)
        assert len(district_config) > 0, "Should have district seat configurations"
        
        # Test that all values are positive integers (seats)
        for district, seats in district_config.items():
            assert isinstance(seats, int), f"Seats for {district} should be integer"
            assert seats > 0, f"Seats for {district} should be positive"
        
        # Test that we have reasonable number of districts
        assert len(district_config) >= 20, "Should have at least 20 districts"
        assert len(district_config) <= 25, "Should have at most 25 districts"
    
    def test_election_dataset_initialization(self):
        """Test ElectionDataset initialization produces consistent results."""
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        # Test basic attributes exist
        assert hasattr(dataset, 'polls_train')
        assert hasattr(dataset, 'polls_test')
        assert hasattr(dataset, 'results_national')
        assert hasattr(dataset, 'results_mult_district')
        assert hasattr(dataset, 'political_families')
        
        # Test political families are as expected
        expected_parties = ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
        assert dataset.political_families == expected_parties
        
        # Test data structures
        assert isinstance(dataset.polls_train, pd.DataFrame)
        assert isinstance(dataset.results_national, pd.DataFrame)
        
        # Test that polls have required structure
        if len(dataset.polls_train) > 0:
            required_poll_cols = ['date', 'pollster', 'sample_size', 'election_date', 'countdown']
            for col in required_poll_cols:
                assert col in dataset.polls_train.columns, f"Missing poll column: {col}"
    
    def test_data_consistency_checks(self):
        """Test basic data consistency requirements."""
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        # Test polls data consistency
        if len(dataset.polls_train) > 0:
            polls = dataset.polls_train
            
            # Sample sizes should be reasonable
            assert polls['sample_size'].min() >= 100, "Sample sizes too small"
            assert polls['sample_size'].max() <= 50000, "Sample sizes too large"  # Adjusted based on actual data
            
            # Vote counts should sum to sample size (data is multinomial counts)
            party_cols = dataset.political_families
            vote_sums = polls[party_cols].sum(axis=1)
            sample_sizes = polls['sample_size']
            # Should be exactly equal since votes are converted to counts
            assert (vote_sums == sample_sizes).all(), "Vote counts should equal sample sizes in multinomial data"
        
        # Test results data consistency
        if len(dataset.results_national) > 0:
            results = dataset.results_national
            
            # Should have data for known historical elections
            historical_dates = pd.to_datetime([
                '2024-03-10', '2022-01-30', '2019-10-06', '2015-10-04', '2011-06-05'
            ])
            result_dates = pd.to_datetime(results['election_date'])
            
            # Should have most historical elections (allowing for some missing)
            common_dates = set(result_dates).intersection(set(historical_dates))
            assert len(common_dates) >= 3, "Should have data for at least 3 historical elections"
    
    def test_coordinate_generation_deterministic(self):
        """Test that coordinate generation is deterministic."""
        dataset1 = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        dataset2 = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        # Test that repeated initialization gives same results
        assert dataset1.political_families == dataset2.political_families
        assert dataset1.historical_election_dates == dataset2.historical_election_dates
        assert dataset1.all_election_dates == dataset2.all_election_dates
        
        # Test that unique pollsters/elections are consistent
        if len(dataset1.polls_train) > 0 and len(dataset2.polls_train) > 0:
            # Should get same pollsters (order might differ)
            pollsters1 = set(dataset1.unique_pollsters)
            pollsters2 = set(dataset2.unique_pollsters)
            assert pollsters1 == pollsters2, "Pollster lists should be identical"


class TestCurrentCoalitionHandling:
    """Test current coalition handling behavior (hardcoded AD=PSD+CDS)."""
    
    def test_current_coalition_representation(self):
        """Document and test how coalitions are currently handled."""
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        # Test that AD is treated as a separate party in current system
        assert 'AD' in dataset.political_families
        assert 'PSD' not in dataset.political_families  # Should not be separate
        assert 'CDS' not in dataset.political_families  # Should not be separate
        
        # Test that data has AD values (not PSD+CDS separately)
        if len(dataset.polls_train) > 0:
            assert 'AD' in dataset.polls_train.columns
            # AD should have non-zero values for recent polls
            recent_polls = dataset.polls_train[
                dataset.polls_train['date'] >= pd.Timestamp('2024-01-01')
            ]
            if len(recent_polls) > 0:
                assert recent_polls['AD'].sum() > 0, "AD should have votes in recent polls"
    
    def test_current_geographic_aggregation(self):
        """Test current geographic aggregation behavior."""
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        # Test district structure
        if len(dataset.results_mult_district) > 0:
            districts = dataset.results_mult_district['Circulo'].unique()
            assert len(districts) >= 20, "Should have at least 20 districts"
            assert len(districts) <= 25, "Should have at most 25 districts"
            
            # Test that district results exist for historical elections
            election_dates = pd.to_datetime(dataset.results_mult_district['election_date']).unique()
            assert len(election_dates) >= 3, "Should have district data for multiple elections"