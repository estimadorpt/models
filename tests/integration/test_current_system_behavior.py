"""
Integration test framework for current system behavior validation.

This module implements comprehensive integration tests that validate the entire
pipeline (train -> predict -> viz) against golden master outputs to ensure
no regressions are introduced.

Key Features:
- Full pipeline testing with output comparison
- Golden master validation against test_baselines/
- Performance benchmark tracking  
- Statistical comparison of model outputs
- Coalition handling regression detection
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
import sys
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.data.dataset import ElectionDataset
from src.models.elections_facade import ElectionsFacade
from src.models.dynamic_gp_election_model import DynamicGPElectionModel


@dataclass
class PipelineOutput:
    """Container for pipeline output data and metadata."""
    
    # Training outputs
    model_config: Optional[Dict] = None
    fit_metrics: Optional[Dict] = None
    trace_summary: Optional[Dict] = None
    
    # Prediction outputs
    district_forecast: Optional[Dict] = None
    national_trends: Optional[Dict] = None
    vote_share_summary: Optional[pd.DataFrame] = None
    seat_summary: Optional[pd.DataFrame] = None
    
    # Visualization outputs (file existence checks)
    viz_files: Optional[List[str]] = None
    
    # Performance metrics
    train_time: Optional[float] = None
    predict_time: Optional[float] = None
    viz_time: Optional[float] = None
    
    # Directory info
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate that essential outputs are present."""
        if self.model_config is None:
            warnings.warn("PipelineOutput created without model_config")


@dataclass  
class ComparisonResult:
    """Result of comparing two pipeline outputs."""
    
    passed: bool
    differences: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    
    def add_difference(self, message: str):
        """Add a test failure difference."""
        self.differences.append(message)
        self.passed = False
    
    def add_warning(self, message: str):
        """Add a non-fatal warning."""
        self.warnings.append(message)
    
    def add_metric(self, name: str, value: float):
        """Add a comparison metric."""
        self.metrics[name] = value


class TestCurrentSystemBehavior:
    """
    Integration tests for current system behavior validation.
    
    This class validates that the entire modeling pipeline produces consistent
    outputs compared to golden master baselines, ensuring no regressions.
    """
    
    @classmethod
    def setup_class(cls):
        """Set up test class with golden master data."""
        cls.golden_masters_dir = Path(__file__).parent.parent.parent / "test_baselines"
        cls.temp_output_dir = Path(__file__).parent / "temp_test_outputs"
        
        # Ensure golden masters exist
        if not cls.golden_masters_dir.exists():
            pytest.skip("Golden masters not found. Run scripts/generate_golden_masters.py first.")
        
        # Load golden master metadata
        metadata_file = cls.golden_masters_dir / "golden_masters_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                cls.golden_metadata = json.load(f)
        else:
            cls.golden_metadata = {}
        
        # Clean up temp directory
        if cls.temp_output_dir.exists():
            shutil.rmtree(cls.temp_output_dir)
        cls.temp_output_dir.mkdir(exist_ok=True)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test class resources."""
        if hasattr(cls, 'temp_output_dir') and cls.temp_output_dir.exists():
            shutil.rmtree(cls.temp_output_dir)
    
    def _load_pipeline_output(self, output_dir: Path) -> PipelineOutput:
        """Load pipeline outputs from a directory."""
        output = PipelineOutput(output_dir=str(output_dir))
        
        # Load training outputs
        model_config_file = output_dir / "model_config.json"
        if model_config_file.exists():
            with open(model_config_file) as f:
                output.model_config = json.load(f)
        
        fit_metrics_file = output_dir / "fit_metrics.json"
        if fit_metrics_file.exists():
            with open(fit_metrics_file) as f:
                output.fit_metrics = json.load(f)
        
        # Load prediction outputs
        predictions_dir = output_dir / "predictions"
        if predictions_dir.exists():
            district_file = predictions_dir / "district_forecast.json"
            if district_file.exists():
                with open(district_file) as f:
                    output.district_forecast = json.load(f)
            
            trends_file = predictions_dir / "national_trends.json"
            if trends_file.exists():
                with open(trends_file) as f:
                    output.national_trends = json.load(f)
            
            vote_summary_file = predictions_dir / "vote_share_summary_election_day.csv"
            if vote_summary_file.exists():
                output.vote_share_summary = pd.read_csv(vote_summary_file, index_col=0)
            
            seat_summary_file = predictions_dir / "total_seat_summary_direct_election_day.csv"
            if seat_summary_file.exists():
                output.seat_summary = pd.read_csv(seat_summary_file, index_col=0)
        
        # Check visualization outputs
        viz_dir = output_dir / "visualizations"
        if viz_dir.exists():
            output.viz_files = [f.name for f in viz_dir.iterdir() if f.is_file()]
        
        return output
    
    def _run_pipeline_command(self, cmd: str, description: str) -> Tuple[float, int]:
        """Run a pipeline command and return execution time and return code."""
        print(f"Running: {description}")
        print(f"Command: {cmd}")
        
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.golden_masters_dir.parent)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
        
        return execution_time, result.returncode
    
    def _compare_model_configs(self, golden: Dict, current: Dict) -> ComparisonResult:
        """Compare model configurations for consistency."""
        result = ComparisonResult(passed=True, differences=[], warnings=[], metrics={})
        
        # Critical parameters that must match exactly
        critical_params = ['model_type', 'election_date']
        
        for param in critical_params:
            if param not in golden or param not in current:
                result.add_difference(f"Missing critical parameter: {param}")
                continue
            
            if golden[param] != current[param]:
                result.add_difference(f"Critical parameter mismatch - {param}: {golden[param]} vs {current[param]}")
        
        # Parameters that should be similar (allowing for minor differences)
        numeric_params = ['draws', 'tune', 'chains']
        
        for param in numeric_params:
            if param in golden and param in current:
                golden_val = golden[param]
                current_val = current[param]
                
                if abs(golden_val - current_val) > 0.1 * golden_val:  # 10% tolerance
                    result.add_warning(f"Numeric parameter difference - {param}: {golden_val} vs {current_val}")
                
                result.add_metric(f"{param}_ratio", current_val / golden_val if golden_val != 0 else float('inf'))
        
        return result
    
    def _compare_fit_metrics(self, golden: Dict, current: Dict) -> ComparisonResult:
        """Compare model fit metrics for regression detection."""
        result = ComparisonResult(passed=True, differences=[], warnings=[], metrics={})
        
        # Key fit metrics that should be stable
        key_metrics = ['poll_mae', 'poll_rmse', 'poll_log_score', 'result_district_mae', 'result_district_rmse']
        
        for metric in key_metrics:
            if metric not in golden or metric not in current:
                result.add_warning(f"Missing fit metric: {metric}")
                continue
            
            golden_val = golden[metric]
            current_val = current[metric]
            
            # Calculate relative difference
            if golden_val != 0:
                rel_diff = abs(current_val - golden_val) / abs(golden_val)
                result.add_metric(f"{metric}_rel_diff", rel_diff)
                
                # Allow 20% variation in fit metrics (MCMC sampling variability)
                if rel_diff > 0.2:
                    result.add_difference(f"Fit metric regression - {metric}: {golden_val:.4f} vs {current_val:.4f} (rel_diff: {rel_diff:.2%})")
            else:
                result.add_metric(f"{metric}_abs_diff", abs(current_val - golden_val))
                
                if abs(current_val - golden_val) > 1e-6:
                    result.add_difference(f"Fit metric change from zero - {metric}: {golden_val} vs {current_val}")
        
        return result
    
    def _compare_vote_predictions(self, golden: pd.DataFrame, current: pd.DataFrame) -> ComparisonResult:
        """Compare vote share predictions for consistency."""
        result = ComparisonResult(passed=True, differences=[], warnings=[], metrics={})
        
        # Check that parties match
        golden_parties = set(golden.index) if golden is not None else set()
        current_parties = set(current.index) if current is not None else set()
        
        if golden_parties != current_parties:
            result.add_difference(f"Party mismatch - Golden: {golden_parties}, Current: {current_parties}")
            return result
        
        if golden is None or current is None:
            result.add_difference("Missing vote share data")
            return result
        
        # Compare mean predictions for each party
        for party in golden.index:
            if party not in current.index:
                result.add_difference(f"Missing party in current predictions: {party}")
                continue
            
            golden_mean = float(golden.loc[party, 'mean'].strip('%')) / 100 if isinstance(golden.loc[party, 'mean'], str) else golden.loc[party, 'mean']
            current_mean = float(current.loc[party, 'mean'].strip('%')) / 100 if isinstance(current.loc[party, 'mean'], str) else current.loc[party, 'mean']
            
            abs_diff = abs(golden_mean - current_mean)
            result.add_metric(f"{party}_vote_abs_diff", abs_diff)
            
            # Allow 2 percentage point difference (MCMC variability)
            if abs_diff > 0.02:
                result.add_difference(f"Vote share difference - {party}: {golden_mean:.1%} vs {current_mean:.1%} (diff: {abs_diff:.1%})")
        
        return result
    
    def _compare_seat_predictions(self, golden: pd.DataFrame, current: pd.DataFrame) -> ComparisonResult:
        """Compare seat predictions for consistency."""
        result = ComparisonResult(passed=True, differences=[], warnings=[], metrics={})
        
        if golden is None or current is None:
            result.add_difference("Missing seat prediction data")
            return result
        
        # Compare mean seat predictions
        for party in golden.index:
            if party not in current.index:
                result.add_difference(f"Missing party in current seat predictions: {party}")
                continue
            
            golden_seats = golden.loc[party, 'mean']
            current_seats = current.loc[party, 'mean']
            
            seats_diff = abs(golden_seats - current_seats)
            result.add_metric(f"{party}_seats_abs_diff", seats_diff)
            
            # Allow 2 seat difference (D'Hondt allocation variability)
            if seats_diff > 2.0:
                result.add_difference(f"Seat prediction difference - {party}: {golden_seats:.1f} vs {current_seats:.1f} (diff: {seats_diff:.1f})")
        
        return result
    
    def _compare_visualization_outputs(self, golden: List[str], current: List[str]) -> ComparisonResult:
        """Compare visualization file outputs."""
        result = ComparisonResult(passed=True, differences=[], warnings=[], metrics={})
        
        golden_files = set(golden) if golden else set()
        current_files = set(current) if current else set()
        
        # Check for missing files
        missing_files = golden_files - current_files
        extra_files = current_files - golden_files
        
        if missing_files:
            result.add_difference(f"Missing visualization files: {missing_files}")
        
        if extra_files:
            result.add_warning(f"Extra visualization files: {extra_files}")
        
        result.add_metric('viz_files_generated', len(current_files))
        result.add_metric('viz_files_expected', len(golden_files))
        
        return result
    
    def test_full_pipeline_integration(self):
        """Test the complete train -> predict -> viz pipeline against golden masters."""
        
        # Load golden master outputs
        golden_output = self._load_pipeline_output(self.golden_masters_dir / "train")
        
        # Create temporary output directory for test run
        test_output_dir = self.temp_output_dir / "integration_test"
        
        # Use the same parameters as golden masters
        election_date = golden_output.model_config.get('election_date', '2024-03-10')
        draws = golden_output.model_config.get('draws', 500)
        tune = golden_output.model_config.get('tune', 500)
        seed = 42  # Fixed seed for reproducibility
        
        # Step 1: Train model
        train_cmd = (
            f"pixi run python -m src.main "
            f"--mode train "
            f"--model-type dynamic_gp "
            f"--election-date {election_date} "
            f"--output-dir {test_output_dir} "
            f"--draws {draws} "
            f"--tune {tune} "
            f"--seed {seed}"
        )
        
        train_time, train_returncode = self._run_pipeline_command(train_cmd, "Integration test training")
        assert train_returncode == 0, "Training failed in integration test"
        
        # Find the generated training directory
        train_dirs = list(test_output_dir.glob("dynamic_gp_run_*"))
        assert len(train_dirs) == 1, f"Expected 1 training dir, found {len(train_dirs)}"
        actual_train_dir = train_dirs[0]
        
        # Step 2: Generate predictions
        predict_cmd = (
            f"pixi run python -m src.main "
            f"--mode predict "
            f"--load-dir {actual_train_dir}"
        )
        
        predict_time, predict_returncode = self._run_pipeline_command(predict_cmd, "Integration test prediction")
        assert predict_returncode == 0, "Prediction failed in integration test"
        
        # Step 3: Generate visualizations
        viz_cmd = (
            f"pixi run python -m src.main "
            f"--mode viz "
            f"--load-dir {actual_train_dir}"
        )
        
        viz_time, viz_returncode = self._run_pipeline_command(viz_cmd, "Integration test visualization")
        assert viz_returncode == 0, "Visualization failed in integration test"
        
        # Step 4: Load current outputs and compare
        current_output = self._load_pipeline_output(actual_train_dir)
        current_output.train_time = train_time
        current_output.predict_time = predict_time
        current_output.viz_time = viz_time
        
        # Run all comparisons
        config_result = self._compare_model_configs(golden_output.model_config, current_output.model_config)
        fit_result = self._compare_fit_metrics(golden_output.fit_metrics, current_output.fit_metrics)
        vote_result = self._compare_vote_predictions(golden_output.vote_share_summary, current_output.vote_share_summary)
        seat_result = self._compare_seat_predictions(golden_output.seat_summary, current_output.seat_summary)
        viz_result = self._compare_visualization_outputs(golden_output.viz_files, current_output.viz_files)
        
        # Collect all results
        all_results = [config_result, fit_result, vote_result, seat_result, viz_result]
        
        # Report results
        print(f"\\n=== INTEGRATION TEST RESULTS ===")
        print(f"Training time: {train_time:.1f}s")
        print(f"Prediction time: {predict_time:.1f}s") 
        print(f"Visualization time: {viz_time:.1f}s")
        print(f"Total time: {train_time + predict_time + viz_time:.1f}s")
        
        for i, result in enumerate(['Config', 'Fit Metrics', 'Vote Predictions', 'Seat Predictions', 'Visualizations']):
            test_result = all_results[i]
            status = "✅ PASS" if test_result.passed else "❌ FAIL"
            print(f"\\n{result}: {status}")
            
            if test_result.differences:
                for diff in test_result.differences:
                    print(f"  ERROR: {diff}")
            
            if test_result.warnings:
                for warning in test_result.warnings:
                    print(f"  WARNING: {warning}")
            
            for metric, value in test_result.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Assert no critical failures
        critical_failures = []
        for result in all_results:
            critical_failures.extend(result.differences)
        
        if critical_failures:
            pytest.fail(f"Integration test failed with {len(critical_failures)} critical differences:\\n" + 
                       "\\n".join(f"- {diff}" for diff in critical_failures))
        
        print(f"\\n✅ Integration test PASSED - Current system behavior matches golden masters")
    
    def test_coalition_handling_consistency(self):
        """Test that coalition handling (AD=PSD+CDS) works consistently."""
        
        # Create dataset to test coalition handling
        dataset = ElectionDataset(
            election_date='2024-03-10',  # Use same date as golden masters
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        # Validate coalition structure
        assert 'AD' in dataset.political_families, "AD coalition not found in political families"
        assert 'PSD' not in dataset.political_families, "PSD should be absorbed into AD"
        assert 'CDS' not in dataset.political_families, "CDS should be absorbed into AD"
        
        # Check that AD has actual data
        ad_poll_data = dataset.polls_train['AD']
        ad_result_data = dataset.results_national['AD']
        
        assert ad_poll_data.sum() > 0, "AD should have poll vote data"
        assert ad_result_data.sum() > 0, "AD should have election result data"
        
        # Verify government status consistency  
        gov_status = dataset.government_status
        ad_col_idx = dataset.political_families.index('AD')
        
        # AD should be in government after 2024 election
        last_election_idx = -1  # Most recent election
        assert gov_status.iloc[last_election_idx, ad_col_idx] == 1, "AD should be in government after 2024"
        
        print("✅ Coalition handling consistency validated")
    
    def test_model_training_deterministic(self):
        """Test that model training is deterministic with fixed seed."""
        
        # Train two models with same seed
        test_seeds = [42, 42]  # Same seed should give identical results
        outputs = []
        
        for i, seed in enumerate(test_seeds):
            output_dir = self.temp_output_dir / f"deterministic_test_{i}"
            
            train_cmd = (
                f"pixi run python -m src.main "
                f"--mode train "
                f"--model-type dynamic_gp "
                f"--election-date 2024-03-10 "
                f"--output-dir {output_dir} "
                f"--draws 100 "  # Small number for speed
                f"--tune 100 "
                f"--seed {seed}"
            )
            
            _, returncode = self._run_pipeline_command(train_cmd, f"Deterministic test {i+1}")
            assert returncode == 0, f"Training {i+1} failed"
            
            train_dirs = list(output_dir.glob("dynamic_gp_run_*"))
            assert len(train_dirs) == 1
            
            output = self._load_pipeline_output(train_dirs[0])
            outputs.append(output)
        
        # Compare outputs - they should be very similar
        result = self._compare_fit_metrics(outputs[0].fit_metrics, outputs[1].fit_metrics)
        
        # With same seed, fit metrics should be nearly identical
        for metric_name, rel_diff in result.metrics.items():
            if 'rel_diff' in metric_name:
                assert rel_diff < 0.01, f"Deterministic training failed - {metric_name}: {rel_diff:.4f} > 0.01"
        
        print("✅ Model training determinism validated")
    
    def test_performance_benchmarks(self):
        """Test that performance is within acceptable bounds."""
        
        # Load golden master performance from metadata
        if 'performance' in self.golden_metadata:
            golden_performance = self.golden_metadata['performance']
        else:
            pytest.skip("No performance benchmarks in golden masters")
        
        # Run quick performance test
        output_dir = self.temp_output_dir / "performance_test"
        
        train_cmd = (
            f"pixi run python -m src.main "
            f"--mode train "
            f"--model-type dynamic_gp "
            f"--election-date 2024-03-10 "
            f"--output-dir {output_dir} "
            f"--draws 100 "  # Minimal for performance testing
            f"--tune 100 "
            f"--seed 42"
        )
        
        train_time, returncode = self._run_pipeline_command(train_cmd, "Performance benchmark")
        assert returncode == 0, "Performance test training failed"
        
        # Performance regression check
        if 'train_time' in golden_performance:
            golden_train_time = golden_performance['train_time']
            time_ratio = train_time / golden_train_time
            
            # Allow 50% performance variation (different hardware, etc.)
            assert time_ratio < 1.5, f"Performance regression - training time {time_ratio:.2f}x slower"
            
            print(f"Training performance: {train_time:.1f}s (ratio: {time_ratio:.2f}x)")
        
        print("✅ Performance benchmarks validated")


class TestSystemRegressionDetection:
    """
    Specific regression detection tests for critical system components.
    
    These tests focus on detecting specific failure modes that could
    break the system silently.
    """
    
    def test_data_loading_regression(self):
        """Test that data loading produces consistent results."""
        
        # Load same data twice
        datasets = []
        for i in range(2):
            dataset = ElectionDataset(
                election_date='2024-03-10',
                baseline_timescales=[365],
                election_timescales=[30, 15],
                test_cutoff=None
            )
            datasets.append(dataset)
        
        # Should be identical
        assert datasets[0].polls_train.shape == datasets[1].polls_train.shape, "Poll data shape mismatch"
        assert datasets[0].results_national.shape == datasets[1].results_national.shape, "Results data shape mismatch"
        
        # Party lists should be identical
        assert datasets[0].political_families == datasets[1].political_families, "Political families mismatch"
        
        print("✅ Data loading regression test passed")
    
    def test_coordinate_factorization_stability(self):
        """Test that coordinate factorization is stable across runs."""
        
        datasets = []
        for i in range(3):
            dataset = ElectionDataset(
                election_date='2024-03-10',
                baseline_timescales=[365],
                election_timescales=[30, 15],
                test_cutoff=None
            )
            datasets.append(dataset)
        
        # Coordinate factorizations should be identical
        for i in range(1, len(datasets)):
            assert datasets[0].unique_pollsters == datasets[i].unique_pollsters, f"Pollster coordinates differ (run {i})"
            assert datasets[0].unique_elections == datasets[i].unique_elections, f"Election coordinates differ (run {i})"
            assert datasets[0].political_families == datasets[i].political_families, f"Party coordinates differ (run {i})"
        
        print("✅ Coordinate factorization stability validated")
    
    def test_multinomial_conversion_consistency(self):
        """Test that multinomial conversion is mathematically consistent."""
        
        dataset = ElectionDataset(
            election_date='2024-03-10',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        polls = dataset.polls_train
        party_cols = dataset.political_families
        
        # Check first 20 polls for mathematical consistency
        for i in range(min(20, len(polls))):
            poll = polls.iloc[i]
            sample_size = poll['sample_size']
            party_votes = poll[party_cols]
            
            # Total votes should equal sample size exactly
            total_votes = party_votes.sum()
            assert total_votes == sample_size, f"Poll {i}: votes {total_votes} ≠ sample size {sample_size}"
            
            # All votes should be non-negative integers
            for party in party_cols:
                votes = party_votes[party]
                assert votes >= 0, f"Poll {i}: {party} has negative votes: {votes}"
                assert votes == int(votes), f"Poll {i}: {party} has non-integer votes: {votes}"
                assert votes <= sample_size, f"Poll {i}: {party} has {votes} > sample size {sample_size}"
        
        print("✅ Multinomial conversion consistency validated")


if __name__ == "__main__":
    # Run tests standalone for debugging
    import sys
    
    # Set up test environment
    test_class = TestCurrentSystemBehavior()
    test_class.setup_class()
    
    try:
        print("Running integration test suite...")
        
        # Run key tests
        test_class.test_coalition_handling_consistency()
        test_class.test_model_training_deterministic()
        
        # Skip full pipeline test in standalone mode (too slow)
        print("\\n⚠️  Skipping full pipeline test in standalone mode")
        print("Run with pytest for complete integration testing")
        
        print("\\n✅ All standalone tests passed!")
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        test_class.teardown_class()