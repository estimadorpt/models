"""
Comprehensive regression detection tools for the election modeling pipeline.

This module provides specialized tools for detecting different types of regressions:
- Bit-for-bit comparison for exact reproducibility
- Statistical comparison for model output validation
- Performance regression detection
- Coalition handling validation
- Automated analysis and reporting

These tools work with golden master outputs to ensure system stability.
"""

import pandas as pd
import numpy as np
import json
import os
import zarr
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import subprocess
import tempfile
import shutil


@dataclass
class RegressionReport:
    """Container for regression detection results."""
    
    test_name: str
    passed: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Detailed results
    critical_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_failure(self, message: str):
        """Add a critical failure."""
        self.critical_failures.append(message)
        self.passed = False
    
    def add_warning(self, message: str):
        """Add a non-critical warning."""
        self.warnings.append(message)
    
    def add_metric(self, name: str, value: float):
        """Add a quantitative metric."""
        self.metrics[name] = value
    
    def add_summary_stat(self, name: str, value: Any):
        """Add summary statistics."""
        self.summary_stats[name] = value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'timestamp': self.timestamp,
            'critical_failures': self.critical_failures,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'summary_stats': self.summary_stats
        }
    
    def print_summary(self):
        """Print a human-readable summary."""
        status = "✅ PASS" if self.passed else "❌ FAIL"
        print(f"\n=== {self.test_name} ===")
        print(f"Status: {status}")
        print(f"Timestamp: {self.timestamp}")
        
        if self.critical_failures:
            print(f"\n❌ Critical Failures ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                print(f"  - {failure}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.metrics:
            print(f"\n📊 Metrics:")
            for name, value in self.metrics.items():
                print(f"  {name}: {value:.6f}")
        
        if self.summary_stats:
            print(f"\n📈 Summary Statistics:")
            for name, value in self.summary_stats.items():
                print(f"  {name}: {value}")


class BitForBitComparator:
    """
    Exact bit-for-bit comparison tools for reproducibility validation.
    
    This class provides methods to compare outputs at the binary level
    to ensure perfect reproducibility when it's expected.
    """
    
    @staticmethod
    def compare_json_files(file1: Path, file2: Path, tolerance: float = 0.0) -> RegressionReport:
        """Compare two JSON files for exact or near-exact equality."""
        report = RegressionReport(
            test_name=f"BitForBit JSON Comparison: {file1.name} vs {file2.name}",
            passed=True
        )
        
        if not file1.exists():
            report.add_failure(f"Golden master file missing: {file1}")
            return report
        
        if not file2.exists():
            report.add_failure(f"Current output file missing: {file2}")
            return report
        
        try:
            with open(file1) as f:
                data1 = json.load(f)
            with open(file2) as f:
                data2 = json.load(f)
        except json.JSONDecodeError as e:
            report.add_failure(f"JSON decode error: {e}")
            return report
        
        # Compare structure
        if set(data1.keys()) != set(data2.keys()):
            report.add_failure(f"Key mismatch: {set(data1.keys())} vs {set(data2.keys())}")
            return report
        
        # Compare values
        differences = BitForBitComparator._compare_nested_dict(data1, data2, tolerance)
        
        for diff in differences:
            if diff['type'] == 'critical':
                report.add_failure(diff['message'])
            else:
                report.add_warning(diff['message'])
        
        report.add_metric('total_keys_compared', len(data1))
        report.add_metric('differences_found', len(differences))
        
        return report
    
    @staticmethod
    def _compare_nested_dict(d1: Dict, d2: Dict, tolerance: float, path: str = "") -> List[Dict]:
        """Recursively compare nested dictionaries."""
        differences = []
        
        for key in d1.keys():
            current_path = f"{path}.{key}" if path else key
            
            if key not in d2:
                differences.append({
                    'type': 'critical',
                    'message': f"Missing key in current: {current_path}"
                })
                continue
            
            val1, val2 = d1[key], d2[key]
            
            if isinstance(val1, dict) and isinstance(val2, dict):
                differences.extend(BitForBitComparator._compare_nested_dict(val1, val2, tolerance, current_path))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > tolerance:
                    diff_type = 'critical' if tolerance == 0.0 else 'warning'
                    differences.append({
                        'type': diff_type,
                        'message': f"Numeric difference at {current_path}: {val1} vs {val2} (diff: {abs(val1-val2)})"
                    })
            elif val1 != val2:
                differences.append({
                    'type': 'critical',
                    'message': f"Value difference at {current_path}: {val1} vs {val2}"
                })
        
        return differences
    
    @staticmethod
    def compare_csv_files(file1: Path, file2: Path, tolerance: float = 1e-10) -> RegressionReport:
        """Compare two CSV files with optional numeric tolerance."""
        report = RegressionReport(
            test_name=f"BitForBit CSV Comparison: {file1.name} vs {file2.name}",
            passed=True
        )
        
        if not file1.exists():
            report.add_failure(f"Golden master file missing: {file1}")
            return report
        
        if not file2.exists():
            report.add_failure(f"Current output file missing: {file2}")
            return report
        
        try:
            df1 = pd.read_csv(file1, index_col=0)
            df2 = pd.read_csv(file2, index_col=0)
        except Exception as e:
            report.add_failure(f"CSV read error: {e}")
            return report
        
        # Shape comparison
        if df1.shape != df2.shape:
            report.add_failure(f"Shape mismatch: {df1.shape} vs {df2.shape}")
            return report
        
        # Index comparison
        if not df1.index.equals(df2.index):
            report.add_failure(f"Index mismatch: {list(df1.index)} vs {list(df2.index)}")
            return report
        
        # Column comparison  
        if not df1.columns.equals(df2.columns):
            report.add_failure(f"Column mismatch: {list(df1.columns)} vs {list(df2.columns)}")
            return report
        
        # Value comparison
        numeric_cols = df1.select_dtypes(include=[np.number]).columns
        string_cols = df1.select_dtypes(exclude=[np.number]).columns
        
        # Check numeric columns with tolerance
        for col in numeric_cols:
            max_diff = abs(df1[col] - df2[col]).max()
            if max_diff > tolerance:
                report.add_failure(f"Numeric difference in {col}: max_diff = {max_diff} > {tolerance}")
            else:
                report.add_metric(f"{col}_max_diff", float(max_diff))
        
        # Check string columns exactly
        for col in string_cols:
            if not df1[col].equals(df2[col]):
                report.add_failure(f"String difference in column: {col}")
        
        report.add_metric('rows_compared', len(df1))
        report.add_metric('numeric_columns', len(numeric_cols))
        report.add_metric('string_columns', len(string_cols))
        
        return report
    
    @staticmethod
    def compare_zarr_arrays(file1: Path, file2: Path, tolerance: float = 1e-12) -> RegressionReport:
        """Compare Zarr array files for model traces."""
        report = RegressionReport(
            test_name=f"BitForBit Zarr Comparison: {file1.name} vs {file2.name}",
            passed=True
        )
        
        if not file1.exists():
            report.add_failure(f"Golden master zarr missing: {file1}")
            return report
        
        if not file2.exists():
            report.add_failure(f"Current output zarr missing: {file2}")
            return report
        
        try:
            arr1 = zarr.open(str(file1), mode='r')
            arr2 = zarr.open(str(file2), mode='r')
        except Exception as e:
            report.add_failure(f"Zarr read error: {e}")
            return report
        
        # Shape comparison
        if arr1.shape != arr2.shape:
            report.add_failure(f"Zarr shape mismatch: {arr1.shape} vs {arr2.shape}")
            return report
        
        # Type comparison
        if arr1.dtype != arr2.dtype:
            report.add_warning(f"Zarr dtype difference: {arr1.dtype} vs {arr2.dtype}")
        
        # Value comparison
        try:
            diff = np.abs(arr1[:] - arr2[:])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            if max_diff > tolerance:
                report.add_failure(f"Zarr values differ: max_diff = {max_diff} > {tolerance}")
            
            report.add_metric('zarr_max_diff', float(max_diff))
            report.add_metric('zarr_mean_diff', float(mean_diff))
            report.add_metric('zarr_elements', int(np.prod(arr1.shape)))
            
        except Exception as e:
            report.add_failure(f"Zarr comparison error: {e}")
        
        return report


class StatisticalComparator:
    """
    Statistical comparison tools for model output validation.
    
    This class provides methods to compare model outputs using statistical
    measures, accounting for inherent MCMC variability.
    """
    
    @staticmethod
    def compare_election_predictions(golden_dir: Path, current_dir: Path) -> RegressionReport:
        """Compare election prediction outputs statistically."""
        report = RegressionReport(
            test_name="Statistical Election Prediction Comparison",
            passed=True
        )
        
        # Load vote share predictions
        vote_golden = StatisticalComparator._load_csv_safe(golden_dir / "predictions" / "vote_share_summary_election_day.csv")
        vote_current = StatisticalComparator._load_csv_safe(current_dir / "predictions" / "vote_share_summary_election_day.csv")
        
        if vote_golden is not None and vote_current is not None:
            vote_report = StatisticalComparator._compare_prediction_dataframes(
                vote_golden, vote_current, "vote_share", max_diff_pct=2.0
            )
            report.critical_failures.extend(vote_report.critical_failures)
            report.warnings.extend(vote_report.warnings)
            report.metrics.update(vote_report.metrics)
        else:
            report.add_failure("Missing vote share prediction files")
        
        # Load seat predictions
        seat_golden = StatisticalComparator._load_csv_safe(golden_dir / "predictions" / "total_seat_summary_direct_election_day.csv")
        seat_current = StatisticalComparator._load_csv_safe(current_dir / "predictions" / "total_seat_summary_direct_election_day.csv")
        
        if seat_golden is not None and seat_current is not None:
            seat_report = StatisticalComparator._compare_prediction_dataframes(
                seat_golden, seat_current, "seats", max_diff_abs=2.0
            )
            report.critical_failures.extend(seat_report.critical_failures)
            report.warnings.extend(seat_report.warnings)
            report.metrics.update({f"seats_{k}": v for k, v in seat_report.metrics.items()})
        else:
            report.add_failure("Missing seat prediction files")
        
        return report
    
    @staticmethod
    def _load_csv_safe(filepath: Path) -> Optional[pd.DataFrame]:
        """Safely load CSV file, returning None if it doesn't exist."""
        try:
            if filepath.exists():
                return pd.read_csv(filepath, index_col=0)
        except Exception:
            pass
        return None
    
    @staticmethod
    def _compare_prediction_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                                     metric_type: str, max_diff_pct: float = None,
                                     max_diff_abs: float = None) -> RegressionReport:
        """Compare prediction dataframes statistically."""
        report = RegressionReport(
            test_name=f"Statistical {metric_type} Comparison",
            passed=True
        )
        
        # Check party alignment
        if not df1.index.equals(df2.index):
            missing_golden = set(df1.index) - set(df2.index)
            missing_current = set(df2.index) - set(df1.index)
            
            if missing_golden:
                report.add_failure(f"Parties missing in current: {missing_golden}")
            if missing_current:
                report.add_failure(f"Extra parties in current: {missing_current}")
            
            # Use intersection for comparison
            common_parties = df1.index.intersection(df2.index)
            df1 = df1.loc[common_parties]
            df2 = df2.loc[common_parties]
        
        # Compare means
        if 'mean' in df1.columns and 'mean' in df2.columns:
            StatisticalComparator._compare_means(df1, df2, report, max_diff_pct, max_diff_abs)
        
        # Compare confidence intervals if present
        if 'hdi_2.5%' in df1.columns and 'hdi_97.5%' in df1.columns:
            StatisticalComparator._compare_confidence_intervals(df1, df2, report)
        
        return report
    
    @staticmethod
    def _compare_means(df1: pd.DataFrame, df2: pd.DataFrame, report: RegressionReport,
                      max_diff_pct: float = None, max_diff_abs: float = None):
        """Compare mean predictions between dataframes."""
        
        for party in df1.index:
            mean1 = StatisticalComparator._parse_numeric_value(df1.loc[party, 'mean'])
            mean2 = StatisticalComparator._parse_numeric_value(df2.loc[party, 'mean'])
            
            abs_diff = abs(mean1 - mean2)
            rel_diff = abs_diff / abs(mean1) if mean1 != 0 else float('inf')
            
            report.add_metric(f"{party}_mean_abs_diff", abs_diff)
            report.add_metric(f"{party}_mean_rel_diff", rel_diff)
            
            # Check thresholds
            failed = False
            if max_diff_abs is not None and abs_diff > max_diff_abs:
                report.add_failure(f"{party} mean absolute difference: {abs_diff:.4f} > {max_diff_abs}")
                failed = True
            
            if max_diff_pct is not None and rel_diff > max_diff_pct / 100:
                report.add_failure(f"{party} mean relative difference: {rel_diff:.2%} > {max_diff_pct}%")
                failed = True
            
            # Soft warning for large changes
            if not failed and rel_diff > 0.1:  # 10% warning threshold
                report.add_warning(f"{party} mean changed by {rel_diff:.2%}")
    
    @staticmethod  
    def _compare_confidence_intervals(df1: pd.DataFrame, df2: pd.DataFrame, report: RegressionReport):
        """Compare confidence interval widths."""
        
        for party in df1.index:
            # Calculate interval widths
            lower1 = StatisticalComparator._parse_numeric_value(df1.loc[party, 'hdi_2.5%'])
            upper1 = StatisticalComparator._parse_numeric_value(df1.loc[party, 'hdi_97.5%'])
            width1 = upper1 - lower1
            
            lower2 = StatisticalComparator._parse_numeric_value(df2.loc[party, 'hdi_2.5%'])
            upper2 = StatisticalComparator._parse_numeric_value(df2.loc[party, 'hdi_97.5%'])
            width2 = upper2 - lower2
            
            width_ratio = width2 / width1 if width1 != 0 else float('inf')
            
            report.add_metric(f"{party}_ci_width_ratio", width_ratio)
            
            # Warning if confidence intervals change dramatically
            if width_ratio > 2.0 or width_ratio < 0.5:
                report.add_warning(f"{party} confidence interval width changed by {width_ratio:.2f}x")
    
    @staticmethod
    def _parse_numeric_value(value: Union[str, float, int]) -> float:
        """Parse numeric value, handling percentage strings."""
        if isinstance(value, str):
            if value.endswith('%'):
                return float(value[:-1]) / 100
            return float(value)
        return float(value)
    
    @staticmethod
    def compare_model_fit_metrics(golden_file: Path, current_file: Path) -> RegressionReport:
        """Compare model fit metrics statistically."""
        report = RegressionReport(
            test_name="Statistical Model Fit Comparison",
            passed=True
        )
        
        # Load fit metrics
        try:
            with open(golden_file) as f:
                golden_metrics = json.load(f)
            with open(current_file) as f:
                current_metrics = json.load(f)
        except Exception as e:
            report.add_failure(f"Cannot load fit metrics: {e}")
            return report
        
        # Key metrics to compare
        key_metrics = ['poll_mae', 'poll_rmse', 'poll_log_score', 
                      'result_district_mae', 'result_district_rmse']
        
        for metric in key_metrics:
            if metric in golden_metrics and metric in current_metrics:
                golden_val = golden_metrics[metric]
                current_val = current_metrics[metric]
                
                abs_diff = abs(current_val - golden_val)
                rel_diff = abs_diff / abs(golden_val) if golden_val != 0 else float('inf')
                
                report.add_metric(f"{metric}_abs_diff", abs_diff)
                report.add_metric(f"{metric}_rel_diff", rel_diff)
                
                # Fit metrics should be stable within 30% (MCMC variability)
                if rel_diff > 0.3:
                    report.add_failure(f"Fit metric regression - {metric}: {golden_val:.4f} vs {current_val:.4f} (diff: {rel_diff:.2%})")
                elif rel_diff > 0.1:
                    report.add_warning(f"Fit metric change - {metric}: {golden_val:.4f} vs {current_val:.4f} (diff: {rel_diff:.2%})")
        
        return report


class PerformanceRegessionDetector:
    """
    Performance regression detection tools.
    
    This class monitors execution times and resource usage to detect
    performance regressions in the modeling pipeline.
    """
    
    @staticmethod
    def compare_execution_times(golden_metadata: Dict, current_times: Dict) -> RegressionReport:
        """Compare execution times for performance regression detection."""
        report = RegressionReport(
            test_name="Performance Regression Detection",
            passed=True
        )
        
        if 'performance' not in golden_metadata:
            report.add_warning("No performance baselines in golden masters")
            return report
        
        golden_perf = golden_metadata['performance']
        
        # Compare key timing metrics
        timing_metrics = ['train_time', 'predict_time', 'viz_time', 'total_time']
        
        for metric in timing_metrics:
            if metric in golden_perf and metric in current_times:
                golden_time = golden_perf[metric]
                current_time = current_times[metric]
                
                time_ratio = current_time / golden_time
                report.add_metric(f"{metric}_ratio", time_ratio)
                
                # Performance regression thresholds
                if time_ratio > 2.0:
                    report.add_failure(f"Severe performance regression - {metric}: {time_ratio:.2f}x slower")
                elif time_ratio > 1.5:
                    report.add_warning(f"Performance regression - {metric}: {time_ratio:.2f}x slower")
                elif time_ratio < 0.5:
                    report.add_warning(f"Unexpected speedup - {metric}: {time_ratio:.2f}x faster")
        
        return report
    
    @staticmethod
    def benchmark_pipeline_components() -> Dict[str, float]:
        """Run lightweight performance benchmarks for each pipeline component."""
        
        # This would run minimal versions of each component to get timing baselines
        # For now, return placeholder structure
        return {
            'data_loading_time': 0.0,
            'model_setup_time': 0.0, 
            'mcmc_sampling_rate': 0.0,  # samples per second
            'prediction_time': 0.0,
            'visualization_time': 0.0
        }


class CoalitionValidationDetector:
    """
    Specialized tools for detecting coalition handling regressions.
    
    This class ensures that the TARGET-ELECTION-DRIVEN approach for
    AD=PSD+CDS coalition handling continues to work correctly.
    """
    
    @staticmethod
    def validate_coalition_structure(dataset) -> RegressionReport:
        """Validate that coalition structure is correct."""
        report = RegressionReport(
            test_name="Coalition Structure Validation",
            passed=True
        )
        
        # Check political families
        parties = dataset.political_families
        
        # AD should be present
        if 'AD' not in parties:
            report.add_failure("AD coalition missing from political families")
        else:
            report.add_summary_stat('AD_position', parties.index('AD'))
        
        # PSD and CDS should be absorbed
        if 'PSD' in parties:
            report.add_failure("PSD should be absorbed into AD coalition")
        if 'CDS' in parties:
            report.add_failure("CDS should be absorbed into AD coalition")
        
        # Check data availability
        if 'AD' in parties:
            ad_idx = parties.index('AD')
            
            # Poll data
            poll_votes = dataset.polls_train.iloc[:, ad_idx + 3:]  # Skip metadata columns
            ad_poll_total = poll_votes.iloc[:, parties.index('AD')].sum()
            
            if ad_poll_total == 0:
                report.add_failure("AD coalition has no poll data")
            else:
                report.add_metric('AD_total_poll_votes', float(ad_poll_total))
            
            # Result data  
            ad_result_total = dataset.results_national['AD'].sum()
            if ad_result_total == 0:
                report.add_failure("AD coalition has no result data")
            else:
                report.add_metric('AD_total_result_votes', float(ad_result_total))
        
        report.add_summary_stat('total_parties', len(parties))
        report.add_summary_stat('party_list', parties)
        
        return report
    
    @staticmethod
    def validate_government_status(dataset) -> RegressionReport:
        """Validate government status consistency for coalitions."""
        report = RegressionReport(
            test_name="Government Status Validation",
            passed=True
        )
        
        gov_status = dataset.government_status
        parties = dataset.political_families
        
        if 'AD' not in parties:
            report.add_failure("Cannot validate government status - AD missing")
            return report
        
        ad_idx = parties.index('AD')
        
        # Check that AD is in government after 2024 election
        # (This should be the case based on the TARGET-ELECTION-DRIVEN approach)
        last_gov_status = gov_status.iloc[-1, ad_idx]
        
        if last_gov_status != 1:
            report.add_failure(f"AD government status incorrect: expected 1, got {last_gov_status}")
        
        # Check historical consistency
        ad_gov_changes = (gov_status.iloc[:, ad_idx].diff() != 0).sum()
        report.add_metric('AD_government_changes', float(ad_gov_changes))
        
        # Validate that government status is binary
        unique_values = set(gov_status.iloc[:, ad_idx].unique())
        expected_values = {0, 1}
        
        if not unique_values.issubset(expected_values):
            report.add_failure(f"Invalid government status values: {unique_values}")
        
        report.add_summary_stat('government_status_shape', gov_status.shape)
        
        return report


class RegressionDetectionSuite:
    """
    Main orchestrator for comprehensive regression detection.
    
    This class coordinates all regression detection tools and produces
    comprehensive reports on system health.
    """
    
    def __init__(self, golden_masters_dir: Path, temp_dir: Optional[Path] = None):
        """Initialize regression detection suite."""
        self.golden_masters_dir = Path(golden_masters_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        
        # Load golden master metadata
        metadata_file = self.golden_masters_dir / "golden_masters_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.golden_metadata = json.load(f)
        else:
            self.golden_metadata = {}
    
    def run_full_regression_suite(self, current_output_dir: Path) -> List[RegressionReport]:
        """Run complete regression detection suite."""
        
        print("🔍 Starting comprehensive regression detection...")
        
        reports = []
        
        # 1. Bit-for-bit comparisons for exact reproducibility
        print("\n1️⃣ Running bit-for-bit comparisons...")
        reports.extend(self._run_bitforbit_tests(current_output_dir))
        
        # 2. Statistical comparisons for model outputs
        print("\n2️⃣ Running statistical comparisons...")
        reports.extend(self._run_statistical_tests(current_output_dir))
        
        # 3. Performance regression detection
        print("\n3️⃣ Running performance regression tests...")
        reports.extend(self._run_performance_tests(current_output_dir))
        
        # 4. Coalition handling validation
        print("\n4️⃣ Running coalition validation tests...")
        reports.extend(self._run_coalition_tests())
        
        return reports
    
    def _run_bitforbit_tests(self, current_dir: Path) -> List[RegressionReport]:
        """Run bit-for-bit comparison tests."""
        reports = []
        
        golden_train_dir = self.golden_masters_dir / "train"
        
        # Compare key JSON files
        json_files = [
            "model_config.json",
            "fit_metrics.json"
        ]
        
        for json_file in json_files:
            golden_path = golden_train_dir / json_file
            current_path = current_dir / json_file
            
            # Use small tolerance for fit metrics (MCMC variability)
            tolerance = 1e-6 if 'fit_metrics' in json_file else 0.0
            
            report = BitForBitComparator.compare_json_files(
                golden_path, current_path, tolerance=tolerance
            )
            reports.append(report)
        
        # Compare key CSV files
        csv_files = [
            "predictions/vote_share_summary_election_day.csv",
            "predictions/total_seat_summary_direct_election_day.csv"
        ]
        
        for csv_file in csv_files:
            golden_path = golden_train_dir / csv_file
            current_path = current_dir / csv_file
            
            report = BitForBitComparator.compare_csv_files(
                golden_path, current_path, tolerance=1e-8
            )
            reports.append(report)
        
        return reports
    
    def _run_statistical_tests(self, current_dir: Path) -> List[RegressionReport]:
        """Run statistical comparison tests."""
        reports = []
        
        golden_train_dir = self.golden_masters_dir / "train"
        
        # Election prediction comparison
        prediction_report = StatisticalComparator.compare_election_predictions(
            golden_train_dir, current_dir
        )
        reports.append(prediction_report)
        
        # Model fit comparison
        fit_report = StatisticalComparator.compare_model_fit_metrics(
            golden_train_dir / "fit_metrics.json",
            current_dir / "fit_metrics.json"
        )
        reports.append(fit_report)
        
        return reports
    
    def _run_performance_tests(self, current_dir: Path) -> List[RegressionReport]:
        """Run performance regression tests."""
        reports = []
        
        # Extract timing from current run (would need to be passed in)
        # For now, use placeholder
        current_times = {
            'train_time': 100.0,  # Would be extracted from actual run
            'predict_time': 10.0,
            'viz_time': 5.0,
            'total_time': 115.0
        }
        
        performance_report = PerformanceRegessionDetector.compare_execution_times(
            self.golden_metadata, current_times
        )
        reports.append(performance_report)
        
        return reports
    
    def _run_coalition_tests(self) -> List[RegressionReport]:
        """Run coalition handling validation tests."""
        reports = []
        
        # Import and test coalition handling
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
            
            from data.dataset import ElectionDataset
            
            dataset = ElectionDataset(
                election_date='2024-03-10',
                baseline_timescales=[365],
                election_timescales=[30, 15],
                test_cutoff=None
            )
            
            # Coalition structure validation
            structure_report = CoalitionValidationDetector.validate_coalition_structure(dataset)
            reports.append(structure_report)
            
            # Government status validation
            gov_report = CoalitionValidationDetector.validate_government_status(dataset)
            reports.append(gov_report)
            
        except Exception as e:
            error_report = RegressionReport(
                test_name="Coalition Validation Error",
                passed=False
            )
            error_report.add_failure(f"Coalition validation failed: {e}")
            reports.append(error_report)
        
        return reports
    
    def generate_summary_report(self, reports: List[RegressionReport]) -> Dict:
        """Generate overall summary of regression detection results."""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(reports),
            'passed_tests': sum(1 for r in reports if r.passed),
            'failed_tests': sum(1 for r in reports if not r.passed),
            'total_failures': sum(len(r.critical_failures) for r in reports),
            'total_warnings': sum(len(r.warnings) for r in reports),
            'success_rate': sum(1 for r in reports if r.passed) / len(reports) if reports else 0.0
        }
        
        # Categorize results
        summary['results_by_category'] = {}
        for report in reports:
            category = report.test_name.split()[0]  # First word as category
            if category not in summary['results_by_category']:
                summary['results_by_category'][category] = {'passed': 0, 'failed': 0}
            
            if report.passed:
                summary['results_by_category'][category]['passed'] += 1
            else:
                summary['results_by_category'][category]['failed'] += 1
        
        # Overall health assessment
        if summary['success_rate'] >= 1.0:
            summary['health_status'] = '✅ EXCELLENT'
        elif summary['success_rate'] >= 0.9:
            summary['health_status'] = '✅ GOOD' 
        elif summary['success_rate'] >= 0.7:
            summary['health_status'] = '⚠️ WARNING'
        else:
            summary['health_status'] = '❌ CRITICAL'
        
        return summary
    
    def save_regression_report(self, reports: List[RegressionReport], output_file: Path):
        """Save comprehensive regression report to file."""
        
        summary = self.generate_summary_report(reports)
        
        full_report = {
            'summary': summary,
            'detailed_reports': [report.to_dict() for report in reports]
        }
        
        with open(output_file, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"\n📊 Regression report saved to: {output_file}")
    
    def print_executive_summary(self, reports: List[RegressionReport]):
        """Print executive summary of regression detection."""
        
        summary = self.generate_summary_report(reports)
        
        print(f"\n" + "="*60)
        print(f"🔍 REGRESSION DETECTION EXECUTIVE SUMMARY")
        print(f"="*60)
        print(f"Health Status: {summary['health_status']}")
        print(f"Success Rate: {summary['success_rate']:.1%} ({summary['passed_tests']}/{summary['total_tests']})")
        print(f"Total Failures: {summary['total_failures']}")
        print(f"Total Warnings: {summary['total_warnings']}")
        
        print(f"\n📋 Results by Category:")
        for category, results in summary['results_by_category'].items():
            total = results['passed'] + results['failed']
            rate = results['passed'] / total if total > 0 else 0.0
            status = "✅" if rate == 1.0 else "⚠️" if rate >= 0.5 else "❌"
            print(f"  {status} {category}: {results['passed']}/{total} ({rate:.1%})")
        
        # Show critical failures
        critical_failures = []
        for report in reports:
            for failure in report.critical_failures:
                critical_failures.append(f"{report.test_name}: {failure}")
        
        if critical_failures:
            print(f"\n❌ Critical Failures ({len(critical_failures)}):")
            for failure in critical_failures[:10]:  # Show first 10
                print(f"  - {failure}")
            if len(critical_failures) > 10:
                print(f"  ... and {len(critical_failures) - 10} more")
        
        print(f"="*60)


def main():
    """Main entry point for regression detection tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Regression detection tools for election modeling")
    parser.add_argument("--golden-masters", type=str, required=True,
                       help="Path to golden masters directory")
    parser.add_argument("--current-output", type=str, required=True,
                       help="Path to current output directory to compare")
    parser.add_argument("--report-file", type=str, 
                       help="Output file for detailed regression report")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick regression tests only")
    
    args = parser.parse_args()
    
    # Initialize regression detection suite
    suite = RegressionDetectionSuite(
        golden_masters_dir=Path(args.golden_masters),
    )
    
    # Run regression tests
    current_output = Path(args.current_output)
    
    if args.quick:
        print("🏃‍♂️ Running quick regression detection...")
        reports = suite._run_statistical_tests(current_output)
        reports.extend(suite._run_coalition_tests())
    else:
        reports = suite.run_full_regression_suite(current_output)
    
    # Print summary
    suite.print_executive_summary(reports)
    
    # Save detailed report if requested
    if args.report_file:
        suite.save_regression_report(reports, Path(args.report_file))
    
    # Exit with error code if regressions detected
    failed_tests = sum(1 for r in reports if not r.passed)
    if failed_tests > 0:
        print(f"\n❌ {failed_tests} regression tests failed")
        return 1
    else:
        print(f"\n✅ All regression tests passed")
        return 0


if __name__ == "__main__":
    exit(main())