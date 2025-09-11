"""
Calculation validation tests for data processing pipeline.

These tests verify that data transformations and calculations produce expected results.
They capture specific values and relationships to detect any changes in core logic.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from data.dataset import ElectionDataset
from data.loaders import load_marktest_polls, load_election_results
from processing.electoral_systems import calculate_dhondt


class TestDataFrameContent:
    """Test actual DataFrame content and values to ensure calculations are correct."""
    
    def test_poll_data_content_validation(self):
        """Test that poll data contains expected values and relationships."""
        polls = load_marktest_polls()
        
        # Test specific known characteristics of the data
        assert len(polls) == 602, f"Expected 602 polls, got {len(polls)}"
        
        # Test date ranges are reasonable (based on actual data)
        min_date = polls['date'].min()
        max_date = polls['date'].max()
        assert pd.Timestamp('2009-01-01') < min_date < pd.Timestamp('2010-01-01'), "Earliest poll should be in 2009"
        assert pd.Timestamp('2025-01-01') < max_date < pd.Timestamp('2026-01-01'), "Latest poll should be in 2025"
        
        # Test party vote distributions look reasonable
        party_cols = ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
        
        # PS should have significant support across all polls (major party)
        # Data is already in percentage format (0.383 = 38.3%)
        ps_votes = polls['PS']  # Already percentages
        assert ps_votes.mean() > 0.15, "PS should average > 15% across polls"
        assert ps_votes.max() < 0.6, "PS should not exceed 60% in any poll"
        
        # AD should have votes (especially in recent polls)
        ad_votes = polls['AD']  # Already percentages
        assert ad_votes.mean() > 0.1, "AD should average > 10% across polls"
        
        # Minor parties should have lower support
        minor_parties = ['PAN', 'L', 'BE']
        for party in minor_parties:
            party_pct = polls[party]  # Already percentages
            assert party_pct.mean() < 0.15, f"{party} should average < 15%"
    
    def test_election_results_content_validation(self):
        """Test that election results contain expected values."""
        historical_dates = ['2024-03-10', '2022-01-30', '2019-10-06', '2015-10-04', '2011-06-05']
        party_cols = ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
        
        results = load_election_results(historical_dates, party_cols, aggregate_national=True)
        
        assert len(results) == 5, "Should have 5 historical elections"
        
        # Test specific election characteristics
        # 2024 election - AD won
        result_2024 = results[results['election_date'] == pd.Timestamp('2024-03-10')]
        assert len(result_2024) == 1, "Should have exactly one 2024 result"
        
        if len(result_2024) > 0:
            ad_2024 = result_2024['AD'].iloc[0] / result_2024['sample_size'].iloc[0]
            ps_2024 = result_2024['PS'].iloc[0] / result_2024['sample_size'].iloc[0]
            assert ad_2024 > ps_2024, "AD should have won 2024 election"
            assert ad_2024 > 0.25, "AD should have > 25% in 2024"
        
        # 2022 election - PS won
        result_2022 = results[results['election_date'] == pd.Timestamp('2022-01-30')]
        if len(result_2022) > 0:
            ps_2022 = result_2022['PS'].iloc[0] / result_2022['sample_size'].iloc[0]
            assert ps_2022 > 0.35, "PS should have > 35% in 2022 (they won)"
    
    def test_district_results_aggregation(self):
        """Test that district results aggregate correctly."""
        historical_dates = ['2024-03-10', '2022-01-30']  # Test recent elections
        party_cols = ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
        
        # Get both national and district results
        national_results = load_election_results(historical_dates, party_cols, aggregate_national=True)
        district_results = load_election_results(historical_dates, party_cols, aggregate_national=False)
        
        # Test that district results aggregate to national totals
        for date in historical_dates:
            date_ts = pd.Timestamp(date)
            
            national_row = national_results[national_results['election_date'] == date_ts]
            district_rows = district_results[district_results['election_date'] == date_ts]
            
            if len(national_row) > 0 and len(district_rows) > 0:
                # Sum district votes for each party
                district_totals = district_rows[party_cols].sum()
                national_votes = national_row[party_cols].iloc[0]
                
                # Should be approximately equal (allowing for rounding)
                for party in party_cols:
                    district_total = district_totals[party]
                    national_vote = national_votes[party]
                    
                    if national_vote > 0:  # Only test parties with votes
                        ratio = district_total / national_vote
                        assert 0.95 < ratio < 1.05, f"{party} {date}: District total {district_total} vs National {national_vote} (ratio: {ratio:.3f})"
    
    def test_coalition_handling_calculations(self):
        """Test that current coalition handling produces expected results."""
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        # Test that AD appears as unified party
        assert 'AD' in dataset.political_families
        assert 'PSD' not in dataset.political_families
        assert 'CDS' not in dataset.political_families
        
        # Test government status calculation
        gov_status = dataset.government_status
        assert gov_status is not None
        assert gov_status.shape[0] == 6, "Should have 6 election cycles"  # 5 historical + 1 target
        assert gov_status.shape[1] == 8, "Should have 8 parties"
        
        # Test specific government periods
        # 2024: AD in government
        ad_col_idx = dataset.political_families.index('AD')
        ps_col_idx = dataset.political_families.index('PS') 
        
        # Find 2024 election row (should be row index for '2024-03-10')
        election_dates = dataset.all_election_dates
        if '2024-03-10' in election_dates:
            row_2024 = election_dates.index('2024-03-10')
            assert gov_status.iloc[row_2024, ad_col_idx] == 1, "AD should be in government after 2024"
            assert gov_status.iloc[row_2024, ps_col_idx] == 0, "PS should not be in government after 2024"
    
    def test_multinomial_conversion_calculations(self):
        """Test that multinomial conversion produces correct counts."""
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        polls = dataset.polls_train
        
        # Test that conversion is mathematically correct
        party_cols = dataset.political_families
        
        # Check first few polls in detail
        for i in range(min(5, len(polls))):
            poll = polls.iloc[i]
            sample_size = poll['sample_size']
            party_votes = poll[party_cols]
            
            # Votes should be integers
            for party in party_cols:
                assert isinstance(party_votes[party], (int, np.integer)), f"Votes should be integers, got {type(party_votes[party])}"
            
            # Votes should sum to sample size
            total_votes = party_votes.sum()
            assert total_votes == sample_size, f"Poll {i}: votes {total_votes} ≠ sample size {sample_size}"
            
            # No party should have more votes than sample size
            for party in party_cols:
                assert party_votes[party] <= sample_size, f"Poll {i}: {party} has {party_votes[party]} > {sample_size}"
                assert party_votes[party] >= 0, f"Poll {i}: {party} has negative votes: {party_votes[party]}"


class TestElectoralSystemCalculations:
    """Test electoral system calculations (D'Hondt, etc.)"""
    
    def test_dhondt_calculation_accuracy(self):
        """Test D'Hondt seat allocation with known examples."""
        
        # Test case 1: Simple case
        votes = {'Party A': 100, 'Party B': 80, 'Party C': 20}
        seats = calculate_dhondt(votes, 5)
        
        # Verify basic properties
        assert sum(seats.values()) == 5, "Should allocate exactly 5 seats"
        assert seats['Party A'] >= seats['Party B'], "Party A should get at least as many seats as Party B"
        assert seats['Party B'] >= seats['Party C'], "Party B should get at least as many seats as Party C"
        
        # Test case 2: Equal votes case
        votes_equal = {'Party A': 100, 'Party B': 100}
        seats_equal = calculate_dhondt(votes_equal, 4)
        
        assert sum(seats_equal.values()) == 4
        assert seats_equal['Party A'] == 2, "Should split equally"
        assert seats_equal['Party B'] == 2, "Should split equally"
        
        # Test case 3: Edge case with zero votes
        votes_with_zero = {'Party A': 100, 'Party B': 0, 'Party C': 50}
        seats_with_zero = calculate_dhondt(votes_with_zero, 3)
        
        assert sum(seats_with_zero.values()) == 3
        assert seats_with_zero['Party B'] == 0, "Party with 0 votes should get 0 seats"
        assert seats_with_zero['Party A'] > 0, "Party A should get some seats"
        assert seats_with_zero['Party C'] > 0, "Party C should get some seats"
    
    def test_dhondt_with_real_data(self):
        """Test D'Hondt with realistic Portuguese election data."""
        # Approximate 2024 election results
        realistic_votes = {
            'AD': 2400000,   # ~29%
            'PS': 2200000,   # ~28% 
            'CH': 1500000,   # ~18%
            'IL': 650000,    # ~8%
            'BE': 400000,    # ~5%
            'CDU': 300000,   # ~4%
            'PAN': 250000,   # ~3%
            'L': 200000      # ~2%
        }
        
        # Allocate seats for a typical district (e.g., 10 seats)
        seats = calculate_dhondt(realistic_votes, 10)
        
        # Basic validation
        assert sum(seats.values()) == 10, "Should allocate exactly 10 seats"
        
        # Top parties should get seats
        assert seats['AD'] > 0, "AD should get seats"
        assert seats['PS'] > 0, "PS should get seats"  
        assert seats['CH'] > 0, "CH should get seats"
        
        # Smaller parties might not get seats in small district
        # (This is mathematically expected with D'Hondt)


class TestDataConsistencyCalculations:
    """Test cross-dataset consistency and relationships."""
    
    def test_poll_election_date_assignments(self):
        """Test that polls are assigned to correct elections."""
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        polls = dataset.polls_train
        
        # Test that recent polls (2024+) are assigned to future election
        recent_polls = polls[polls['date'] >= pd.Timestamp('2024-03-11')]  # After 2024 election
        if len(recent_polls) > 0:
            future_assignments = recent_polls['election_date'].unique()
            assert pd.Timestamp('2026-01-01') in future_assignments, "Recent polls should be assigned to 2026 election"
        
        # Test that older polls assigned to historical elections
        old_polls = polls[polls['date'] < pd.Timestamp('2024-03-10')]
        if len(old_polls) > 0:
            historical_elections = [
                pd.Timestamp('2024-03-10'), pd.Timestamp('2022-01-30'), 
                pd.Timestamp('2019-10-06'), pd.Timestamp('2015-10-04'), pd.Timestamp('2011-06-05')
            ]
            assigned_elections = old_polls['election_date'].unique()
            for assigned in assigned_elections:
                assert assigned in historical_elections, f"Poll assigned to unexpected election: {assigned}"
    
    def test_countdown_calculations(self):
        """Test that countdown days are calculated correctly."""
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        polls = dataset.polls_train
        
        # Test countdown calculation logic
        for i in range(min(10, len(polls))):  # Test first 10 polls
            poll = polls.iloc[i]
            poll_date = poll['date']
            election_date = poll['election_date'] 
            countdown = poll['countdown']
            
            # Calculate expected countdown
            expected_countdown = (election_date - poll_date).days
            
            assert countdown == expected_countdown, f"Poll {i}: countdown {countdown} ≠ expected {expected_countdown}"
            assert countdown >= 0, f"Poll {i}: countdown should be non-negative, got {countdown}"
    
    def test_coordinate_factorization_deterministic(self):
        """Test that coordinate factorization produces consistent results."""
        # Create dataset twice
        dataset1 = ElectionDataset('2026-01-01', [365], [30, 15], None)
        dataset2 = ElectionDataset('2026-01-01', [365], [30, 15], None)
        
        # Test pollster factorization consistency
        pollsters1 = set(dataset1.unique_pollsters)
        pollsters2 = set(dataset2.unique_pollsters)
        assert pollsters1 == pollsters2, "Pollster factorization should be deterministic"
        
        # Test election factorization consistency
        elections1 = dataset1.unique_elections
        elections2 = dataset2.unique_elections
        assert list(elections1) == list(elections2), "Election factorization should be deterministic"


if __name__ == "__main__":
    # Run validation tests standalone
    content_tests = TestDataFrameContent()
    calc_tests = TestElectoralSystemCalculations() 
    consistency_tests = TestDataConsistencyCalculations()
    
    print("=== DataFrame Content Validation ===")
    content_tests.test_poll_data_content_validation()
    content_tests.test_election_results_content_validation()
    content_tests.test_district_results_aggregation()
    content_tests.test_coalition_handling_calculations()
    content_tests.test_multinomial_conversion_calculations()
    
    print("\n=== Electoral System Calculations ===")
    calc_tests.test_dhondt_calculation_accuracy()
    calc_tests.test_dhondt_with_real_data()
    
    print("\n=== Data Consistency Calculations ===")
    consistency_tests.test_poll_election_date_assignments()
    consistency_tests.test_countdown_calculations()
    consistency_tests.test_coordinate_factorization_deterministic()
    
    print("\nAll calculation validation tests completed!")