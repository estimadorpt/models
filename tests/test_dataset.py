import pytest
import pandas as pd
from src.data.dataset import ElectionDataset

def test_find_closest_election_date():
    """Test if polls after the last historical election are assigned the target date."""
    
    # Create a dataset with a future election date
    future_election = "2025-10-05"  # A date not in historical_election_dates
    dataset = ElectionDataset(
        election_date=future_election,
        baseline_timescales=[365],
        election_timescales=[60],
    )
    
    # Create test poll rows with different dates
    last_historical = pd.to_datetime(dataset.historical_election_dates[0])  # 2024-03-10
    
    # Test case 1: Poll before last historical election
    test_row = pd.Series({"date": last_historical - pd.Timedelta(days=30)})
    result = dataset.find_closest_election_date(test_row)
    assert result == last_historical, f"Poll before last election should point to last election {last_historical}"
    
    # Test case 2: Poll after last historical election
    test_row = pd.Series({"date": last_historical + pd.Timedelta(days=1)})
    result = dataset.find_closest_election_date(test_row)
    assert result == pd.to_datetime(future_election), f"Poll after last election should point to future election {future_election}"
    
    # Test case 3: Poll far in the past
    test_row = pd.Series({"date": pd.to_datetime("2010-01-01")})
    result = dataset.find_closest_election_date(test_row)
    assert result == pd.to_datetime("2011-06-05"), "Poll in distant past should point to appropriate historical election"
    
    # Also test using the actual polls loaded by the dataset
    if len(dataset.polls) > 0:
        # Check if any polls exist with dates after the last historical election
        future_polls = dataset.polls[dataset.polls["date"] > last_historical]
        if len(future_polls) > 0:
            # All future polls should be assigned to the target election date
            assert pd.to_datetime(future_election) in future_polls["election_date"].unique(), \
                "Future polls are not correctly assigned to the target election date" 