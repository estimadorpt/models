"""
Performance benchmark tests for data loading pipeline.

These tests capture the current performance characteristics to detect regressions.
"""

import pytest
import time
import psutil
import os
import sys

# Add src to path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from data.dataset import ElectionDataset
from data.loaders import load_marktest_polls, load_election_results


class TestPerformanceBenchmarks:
    """Performance regression tests to ensure changes don't slow down the system."""
    
    def test_poll_loading_performance(self):
        """Test that poll loading completes within reasonable time."""
        start_time = time.time()
        polls = load_marktest_polls()
        load_time = time.time() - start_time
        
        assert len(polls) > 0, "Should load poll data"
        # Allow up to 5 seconds for poll loading (current system takes ~1-2 seconds)
        assert load_time < 5.0, f"Poll loading took {load_time:.2f}s, should be < 5s"
        print(f"Poll loading time: {load_time:.2f}s")
    
    def test_results_loading_performance(self):
        """Test that election results loading completes within reasonable time."""
        historical_dates = [
            '2024-03-10', '2022-01-30', '2019-10-06', '2015-10-04', '2011-06-05'
        ]
        party_columns = ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
        
        start_time = time.time()
        results = load_election_results(historical_dates, party_columns, aggregate_national=True)
        load_time = time.time() - start_time
        
        assert len(results) > 0, "Should load election results"
        # Allow up to 3 seconds for results loading
        assert load_time < 3.0, f"Results loading took {load_time:.2f}s, should be < 3s"
        print(f"Results loading time: {load_time:.2f}s")
    
    def test_dataset_initialization_performance(self):
        """Test that full dataset initialization completes within reasonable time."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        init_time = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Performance assertions
        assert init_time < 10.0, f"Dataset initialization took {init_time:.2f}s, should be < 10s"
        assert memory_increase < 200, f"Memory increase {memory_increase:.1f}MB, should be < 200MB"
        
        # Basic functionality assertions
        assert hasattr(dataset, 'polls_train')
        assert hasattr(dataset, 'results_national')
        assert len(dataset.polls_train) > 0
        assert len(dataset.results_national) > 0
        
        print(f"Dataset initialization time: {init_time:.2f}s")
        print(f"Memory increase: {memory_increase:.1f}MB")
    
    def test_coordinate_generation_performance(self):
        """Test that coordinate generation is reasonably fast."""
        dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        start_time = time.time()
        # Access coordinate-related attributes that trigger generation
        pollsters = dataset.unique_pollsters
        elections = dataset.unique_elections
        parties = dataset.political_families
        coord_time = time.time() - start_time
        
        # Should be very fast since most work done during initialization
        assert coord_time < 0.1, f"Coordinate access took {coord_time:.3f}s, should be < 0.1s"
        
        # Verify coordinates are reasonable
        assert len(pollsters) > 0, "Should have pollsters"
        assert len(parties) == 8, "Should have 8 parties"
        
        print(f"Coordinate generation time: {coord_time:.3f}s")


class TestMemoryUsage:
    """Test memory usage patterns to detect memory leaks or excessive usage."""
    
    def test_repeated_dataset_initialization(self):
        """Test that repeated initialization doesn't cause memory leaks."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy multiple datasets
        for i in range(3):
            dataset = ElectionDataset(
                election_date='2026-01-01',
                baseline_timescales=[365],
                election_timescales=[30, 15],
                test_cutoff=None
            )
            # Access some data to ensure full initialization
            _ = len(dataset.polls_train)
            _ = len(dataset.results_national)
            del dataset
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (allowing for some caching)
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB after 3 iterations, possible leak"
        
        print(f"Memory increase after 3 dataset initializations: {memory_increase:.1f}MB")


if __name__ == "__main__":
    # Run performance tests standalone for profiling
    test_perf = TestPerformanceBenchmarks()
    test_mem = TestMemoryUsage()
    
    print("=== Performance Benchmarks ===")
    test_perf.test_poll_loading_performance()
    test_perf.test_results_loading_performance() 
    test_perf.test_dataset_initialization_performance()
    test_perf.test_coordinate_generation_performance()
    
    print("\n=== Memory Usage Tests ===")
    test_mem.test_repeated_dataset_initialization()
    
    print("\nAll performance benchmarks completed successfully!")