"""
Data integrity validation tests for edge cases and corruption detection.

These tests catch specific failure modes like parsing errors, calculation mistakes,
and party/coalition inconsistencies that could break the system silently.
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


class TestDataIntegrityValidation:
    """Catch data corruption and parsing errors that could break the system."""
    
    def test_poll_parsing_corruption_detection(self):
        """Test that we detect corrupted or misparsed poll data."""
        polls = load_marktest_polls()
        
        # Test for NaN/infinite values in critical columns
        critical_cols = ['date', 'pollster', 'sample_size'] + ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
        
        for col in critical_cols:
            assert col in polls.columns, f"Critical column {col} missing from poll data"
            
            if col in ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD', 'sample_size']:
                # Numeric columns should not have NaN
                nan_count = polls[col].isna().sum()
                assert nan_count == 0, f"Column {col} has {nan_count} NaN values"
                
                # Should not have infinite values
                inf_count = np.isinf(polls[col]).sum()
                assert inf_count == 0, f"Column {col} has {inf_count} infinite values"
                
                # Should not have negative values
                negative_count = (polls[col] < 0).sum()
                assert negative_count == 0, f"Column {col} has {negative_count} negative values"
        
        # Test date parsing integrity
        assert polls['date'].dtype == 'datetime64[ns]', "Date column should be datetime type"
        invalid_dates = polls['date'].isna().sum()
        assert invalid_dates == 0, f"Found {invalid_dates} invalid dates in poll data"
    
    def test_sample_size_calculation_errors(self):
        """Test detection of sample size calculation errors."""
        polls = load_marktest_polls()
        party_cols = ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
        
        # Calculate vote totals and compare to sample sizes
        for i in range(min(len(polls), 10)):  # Test first 10 polls in detail
            poll = polls.iloc[i]
            sample_size = poll['sample_size']
            
            # Sum party percentages
            party_total = poll[party_cols].sum()
            
            # For percentage data, should sum to approximately 1.0 (allowing for rounding/other)
            assert 0.8 <= party_total <= 1.2, f"Poll {i}: party percentages sum to {party_total:.3f}, should be ~1.0"
            
            # Check for impossible values
            for party in party_cols:
                party_val = poll[party]
                assert 0 <= party_val <= 1, f"Poll {i}: {party} has impossible value {party_val}"
        
        # Test multinomial conversion doesn't create vote counts > sample size
        dataset = ElectionDataset('2026-01-01', [365], [30, 15], None)
        mult_polls = dataset.polls_train
        
        for i in range(min(len(mult_polls), 10)):
            poll = mult_polls.iloc[i]
            sample_size = poll['sample_size']
            party_votes = poll[party_cols]
            
            # No party should have more votes than sample size
            for party in party_cols:
                assert party_votes[party] <= sample_size, f"Poll {i}: {party} has {party_votes[party]} votes > sample size {sample_size}"
            
            # Total votes should equal sample size exactly
            total_votes = party_votes.sum()
            assert total_votes == sample_size, f"Poll {i}: total votes {total_votes} ≠ sample size {sample_size}"
    
    def test_party_coalition_consistency(self):
        """Test consistency between political families, data columns, and coordinates."""
        dataset = ElectionDataset('2026-01-01', [365], [30, 15], None)
        
        # Test that political_families matches data columns
        expected_parties = set(dataset.political_families)
        poll_data_parties = set(['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD'])
        
        assert expected_parties == poll_data_parties, f"Political families {expected_parties} ≠ poll columns {poll_data_parties}"
        
        # Test that all political_families have data columns in polls_train
        for party in dataset.political_families:
            assert party in dataset.polls_train.columns, f"Party {party} missing from polls_train columns"
            assert party in dataset.results_national.columns, f"Party {party} missing from results_national columns"
        
        # Test that data doesn't have unexpected parties (not in political_families)
        poll_cols = set(dataset.polls_train.columns)
        result_cols = set(dataset.results_national.columns)
        
        # These are the only party columns that should exist
        expected_party_cols = set(dataset.political_families)
        
        # Find any unexpected party columns (excluding non-party columns)
        non_party_cols = {'date', 'pollster', 'sample_size', 'election_date', 'countdown', 'gdp'}
        
        # Additional columns from multi-level geographic system
        additional_result_cols = {
            'Circulo', 'geographic_id', 'number_voters', 'subscribed_voters', 
            'null_votes', 'blank_votes', 'district_name', 'full_municipality_id', 
            'full_parish_id', 'territory_code', 'district_code', 'municipality_code', 'parish_code'
        }
        
        unexpected_poll_parties = poll_cols - expected_party_cols - non_party_cols
        unexpected_result_parties = result_cols - expected_party_cols - non_party_cols - additional_result_cols
        
        assert len(unexpected_poll_parties) == 0, f"Unexpected parties in poll data: {unexpected_poll_parties}"
        assert len(unexpected_result_parties) == 0, f"Unexpected parties in result data: {unexpected_result_parties}"
    
    def test_coalition_representation_consistency(self):
        """Test that coalition representation is consistent throughout the system."""
        dataset = ElectionDataset('2026-01-01', [365], [30, 15], None)
        
        # Test AD coalition consistency
        assert 'AD' in dataset.political_families, "AD should be in political families"
        assert 'PSD' not in dataset.political_families, "PSD should not be separate from AD"
        assert 'CDS' not in dataset.political_families, "CDS should not be separate from AD"
        
        # Test that AD has actual vote data (not just zeros)
        ad_poll_data = dataset.polls_train['AD']
        ad_result_data = dataset.results_national['AD']
        
        assert ad_poll_data.sum() > 0, "AD should have vote data in polls"
        assert ad_result_data.sum() > 0, "AD should have vote data in results"
        
        # Test government status consistency
        gov_status = dataset.government_status
        ad_col_idx = dataset.political_families.index('AD')
        
        # AD should be in government for some periods (2024)
        ad_in_gov_periods = (gov_status.iloc[:, ad_col_idx] == 1).sum()
        assert ad_in_gov_periods > 0, "AD should be in government for at least one period"
    
    def test_coordinate_data_consistency(self):
        """Test that coordinates match the actual data structure."""
        dataset = ElectionDataset('2026-01-01', [365], [30, 15], None)
        
        # Test pollster coordinates match actual pollsters in data
        data_pollsters = set(dataset.polls_train['pollster'].unique())
        coord_pollsters = set(dataset.unique_pollsters)
        
        assert data_pollsters == coord_pollsters, f"Data pollsters {data_pollsters} ≠ coordinate pollsters {coord_pollsters}"
        
        # Test election coordinates include all elections in data
        data_elections = set(pd.to_datetime(dataset.polls_train['election_date']).dt.strftime('%Y-%m-%d'))
        coord_elections = set(dataset.all_election_dates)
        
        # Data elections should be subset of coordinate elections (coords may include target)
        assert data_elections.issubset(coord_elections), f"Data elections {data_elections} not in coordinates {coord_elections}"
        
        # Test that historical elections are consistent
        historical_coords = set(dataset.historical_election_dates)
        result_elections = set(pd.to_datetime(dataset.results_national['election_date']).dt.strftime('%Y-%m-%d'))
        
        assert result_elections.issubset(historical_coords), f"Result elections {result_elections} not in historical coords {historical_coords}"


class TestEdgeCaseDetection:
    """Test detection of edge cases that could cause silent failures."""
    
    def test_missing_data_detection(self):
        """Test detection of missing or insufficient data."""
        dataset = ElectionDataset('2026-01-01', [365], [30, 15], None)
        
        # Test minimum data requirements
        assert len(dataset.polls_train) > 50, f"Too few polls for training: {len(dataset.polls_train)}"
        assert len(dataset.results_national) >= 3, f"Too few election results: {len(dataset.results_national)}"
        assert len(dataset.unique_pollsters) >= 3, f"Too few pollsters: {len(dataset.unique_pollsters)}"
        
        # Test data completeness
        critical_attrs = ['political_families', 'historical_election_dates', 'all_election_dates']
        for attr in critical_attrs:
            assert hasattr(dataset, attr), f"Dataset missing critical attribute: {attr}"
            attr_val = getattr(dataset, attr)
            assert len(attr_val) > 0, f"Dataset attribute {attr} is empty"
    
    def test_date_range_validity(self):
        """Test that date ranges are reasonable and consistent."""
        dataset = ElectionDataset('2026-01-01', [365], [30, 15], None)
        
        # Test poll date ranges
        poll_dates = pd.to_datetime(dataset.polls_train['date'])
        min_date = poll_dates.min()
        max_date = poll_dates.max()
        
        # Should span multiple years
        date_span_years = (max_date - min_date).days / 365.25
        assert date_span_years >= 10, f"Poll data spans only {date_span_years:.1f} years, should be >10"
        
        # Should not have future dates beyond reasonable forecasting
        today = pd.Timestamp.now()
        max_future_days = (max_date - today).days
        assert max_future_days <= 1000, f"Polls extend {max_future_days} days into future, seems excessive"
        
        # Test election date consistency
        election_dates = pd.to_datetime(dataset.results_national['election_date'])
        for i in range(len(election_dates) - 1):
            # Elections should be in chronological order (or at least reasonable)
            date_diff = (election_dates.iloc[i+1] - election_dates.iloc[i]).days
            assert abs(date_diff) >= 100, f"Elections too close together: {date_diff} days"
    
    def test_numerical_stability(self):
        """Test for numerical issues that could cause calculation errors."""
        dataset = ElectionDataset('2026-01-01', [365], [30, 15], None)
        
        # Test that vote percentages don't have precision issues
        party_cols = dataset.political_families
        polls = dataset.polls_train
        
        # Check for extremely small or large values that might indicate errors
        for party in party_cols:
            party_values = polls[party]
            
            # No values should be exactly 0.0 (rare in real polling)
            zero_count = (party_values == 0).sum()
            if zero_count > len(party_values) * 0.1:  # >10% zeros might indicate issue
                print(f"Warning: {party} has {zero_count} zero values ({zero_count/len(party_values)*100:.1f}%)")
            
            # Check for suspiciously precise values (might indicate rounding errors)
            # Real poll data should have some natural variation
            unique_values = party_values.nunique()
            if unique_values < 10 and len(party_values) > 100:
                print(f"Warning: {party} has only {unique_values} unique values in {len(party_values)} polls")
    
    def test_data_correlation_sanity(self):
        """Test that data relationships make political sense."""
        dataset = ElectionDataset('2026-01-01', [365], [30, 15], None)
        
        polls = dataset.polls_train
        
        # Test that major parties have reasonable correlation patterns
        # Convert counts back to percentages for political analysis
        if len(polls) > 50:
            ps_pct = polls['PS'] / polls['sample_size']
            ad_pct = polls['AD'] / polls['sample_size'] 
            ps_ad_corr = ps_pct.corr(ad_pct)
            
            # PS and AD should compete somewhat (negative correlation) or be independent
            # But very high positive correlation (>0.8) would be suspicious
            assert ps_ad_corr < 0.8, f"PS-AD percentage correlation {ps_ad_corr:.3f} too high (should compete or be independent)"
        
        # Test temporal consistency - parties shouldn't have impossible jumps
        if len(polls) > 20:
            polls_sorted = polls.sort_values('date')
            
            # Check for impossible party support changes (>30% in single poll)  
            for party in ['PS', 'AD']:  # Major parties
                # Convert to percentages for temporal analysis
                party_pct = polls_sorted[party] / polls_sorted['sample_size']
                pct_diffs = party_pct.diff().abs()
                max_change = pct_diffs.max()
                
                # A party shouldn't change >30 percentage points between consecutive polls (data error indicator)
                assert max_change < 0.3, f"{party} has impossible {max_change:.1%} change between consecutive polls"


if __name__ == "__main__":
    # Run integrity tests standalone
    integrity_tests = TestDataIntegrityValidation()
    edge_case_tests = TestEdgeCaseDetection()
    
    print("=== Data Integrity Validation Tests ===")
    integrity_tests.test_poll_parsing_corruption_detection()
    integrity_tests.test_sample_size_calculation_errors()
    integrity_tests.test_party_coalition_consistency()
    integrity_tests.test_coalition_representation_consistency()
    integrity_tests.test_coordinate_data_consistency()
    
    print("\n=== Edge Case Detection Tests ===")
    edge_case_tests.test_missing_data_detection()
    edge_case_tests.test_date_range_validity()
    edge_case_tests.test_numerical_stability()
    edge_case_tests.test_data_correlation_sanity()
    
    print("\nAll data integrity validation tests completed!")