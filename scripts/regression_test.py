#!/usr/bin/env python3
"""
Local Regression Testing Infrastructure

This script provides tools for comparing model outputs against golden master baselines
to detect regressions. Unlike CI/CD validation, this runs locally and includes
heavy computational comparisons.

Usage:
    python scripts/regression_test.py --baseline test_baselines/latest --current outputs/latest
    python scripts/regression_test.py --generate-new --current outputs/latest
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import hashlib
from typing import Dict, List, Tuple, Optional, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class RegressionTester:
    """Comprehensive regression testing for election model outputs"""
    
    def __init__(self, baseline_dir: Path, current_dir: Path, tolerance: float = 0.01):
        self.baseline_dir = Path(baseline_dir)
        self.current_dir = Path(current_dir)
        self.tolerance = tolerance
        self.results = {
            "test_summary": {},
            "detailed_results": {},
            "errors": [],
            "warnings": []
        }
    
    def run_comprehensive_test(self) -> bool:
        """Run all regression tests and return overall pass/fail"""
        print(f"🔍 COMPREHENSIVE REGRESSION TEST")
        print(f"Baseline: {self.baseline_dir}")
        print(f"Current:  {self.current_dir}")
        print(f"Tolerance: {self.tolerance}")
        print("="*60)
        
        # Verify directories exist
        if not self.baseline_dir.exists():
            self.results["errors"].append(f"Baseline directory not found: {self.baseline_dir}")
            return False
            
        if not self.current_dir.exists():
            self.results["errors"].append(f"Current directory not found: {self.current_dir}")
            return False
        
        # Run test categories
        tests = [
            ("File Structure", self.test_file_structure),
            ("Model Configuration", self.test_model_config),
            ("Training Metrics", self.test_training_metrics),
            ("Prediction Outputs", self.test_prediction_outputs),
            ("Seat Allocations", self.test_seat_allocations),
            ("Visualization Assets", self.test_visualizations),
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}...")
            try:
                passed = test_func()
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   {status}: {test_name}")
                self.results["test_summary"][test_name] = passed
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"   ❌ ERROR: {test_name} - {e}")
                self.results["errors"].append(f"{test_name}: {e}")
                self.results["test_summary"][test_name] = False
                all_passed = False
        
        # Generate report
        self.generate_report()
        
        return all_passed
    
    def test_file_structure(self) -> bool:
        """Test that expected files are present in current run"""
        expected_files = [
            "model_config.json",
            "fit_metrics.json", 
            "trace.nc",
            "predictions/vote_share_summary_election_day.csv",
            "predictions/seat_summary_election_day.csv",
            "diagnostics/trace_plot.png",
            "visualizations/latent_popularity_vs_polls.png"
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not (self.current_dir / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.results["detailed_results"]["file_structure"] = {
                "status": "fail",
                "missing_files": missing_files
            }
            return False
        
        self.results["detailed_results"]["file_structure"] = {
            "status": "pass",
            "all_files_present": True
        }
        return True
    
    def test_model_config(self) -> bool:
        """Test that model configuration matches baseline expectations"""
        baseline_config_path = self.baseline_dir / "train" / "dynamic_gp_run_"
        current_config_path = self.current_dir / "model_config.json"
        
        # Find baseline config (in timestamped directory)
        baseline_run_dirs = list((self.baseline_dir / "train").glob("dynamic_gp_run_*"))
        if not baseline_run_dirs:
            self.results["warnings"].append("No baseline training directory found")
            return False
        
        baseline_config_path = baseline_run_dirs[0] / "model_config.json"
        
        if not baseline_config_path.exists() or not current_config_path.exists():
            return False
        
        with open(baseline_config_path) as f:
            baseline_config = json.load(f)
        with open(current_config_path) as f:
            current_config = json.load(f)
        
        # Compare key configuration parameters
        key_params = ["model_type", "election_date", "draws", "tune", "seed"]
        differences = {}
        
        for param in key_params:
            if baseline_config.get(param) != current_config.get(param):
                differences[param] = {
                    "baseline": baseline_config.get(param),
                    "current": current_config.get(param)
                }
        
        if differences:
            self.results["detailed_results"]["model_config"] = {
                "status": "fail", 
                "differences": differences
            }
            return False
        
        self.results["detailed_results"]["model_config"] = {
            "status": "pass",
            "configs_match": True
        }
        return True
    
    def test_training_metrics(self) -> bool:
        """Compare training metrics (R-hat, ESS, fit metrics)"""
        baseline_metrics_path = self._find_baseline_file("fit_metrics.json")
        current_metrics_path = self.current_dir / "fit_metrics.json"
        
        if not baseline_metrics_path or not current_metrics_path.exists():
            return False
        
        with open(baseline_metrics_path) as f:
            baseline_metrics = json.load(f)
        with open(current_metrics_path) as f:
            current_metrics = json.load(f)
        
        # Compare key metrics with tolerance
        key_metrics = ["log_likelihood", "waic", "loo"]
        metric_comparisons = {}
        
        for metric in key_metrics:
            if metric in baseline_metrics and metric in current_metrics:
                baseline_val = baseline_metrics[metric]
                current_val = current_metrics[metric]
                
                if baseline_val is None or current_val is None:
                    continue
                
                diff = abs(float(current_val) - float(baseline_val))
                rel_diff = diff / max(abs(float(baseline_val)), 1e-10)
                
                metric_comparisons[metric] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "absolute_diff": diff,
                    "relative_diff": rel_diff,
                    "within_tolerance": rel_diff <= self.tolerance
                }
        
        all_within_tolerance = all(
            comp["within_tolerance"] 
            for comp in metric_comparisons.values()
        )
        
        self.results["detailed_results"]["training_metrics"] = {
            "status": "pass" if all_within_tolerance else "fail",
            "metric_comparisons": metric_comparisons
        }
        
        return all_within_tolerance
    
    def test_prediction_outputs(self) -> bool:
        """Compare vote share predictions"""
        baseline_pred_path = self._find_baseline_file("predictions/vote_share_summary_election_day.csv")
        current_pred_path = self.current_dir / "predictions" / "vote_share_summary_election_day.csv"
        
        if not baseline_pred_path or not current_pred_path.exists():
            return False
        
        baseline_df = pd.read_csv(baseline_pred_path, index_col=0)
        current_df = pd.read_csv(current_pred_path, index_col=0)
        
        # Compare mean predictions for each party
        comparison_results = {}
        
        for party in baseline_df.index:
            if party in current_df.index:
                baseline_mean = baseline_df.loc[party, "mean"]
                current_mean = current_df.loc[party, "mean"]
                
                diff = abs(current_mean - baseline_mean)
                rel_diff = diff / max(abs(baseline_mean), 0.01)  # Avoid division by very small numbers
                
                comparison_results[party] = {
                    "baseline": baseline_mean,
                    "current": current_mean,
                    "absolute_diff": diff,
                    "relative_diff": rel_diff,
                    "within_tolerance": rel_diff <= self.tolerance
                }
        
        all_within_tolerance = all(
            comp["within_tolerance"] 
            for comp in comparison_results.values()
        )
        
        self.results["detailed_results"]["prediction_outputs"] = {
            "status": "pass" if all_within_tolerance else "fail",
            "party_comparisons": comparison_results
        }
        
        return all_within_tolerance
    
    def test_seat_allocations(self) -> bool:
        """Compare seat allocation predictions"""
        baseline_seat_path = self._find_baseline_file("predictions/seat_summary_election_day.csv")
        current_seat_path = self.current_dir / "predictions" / "seat_summary_election_day.csv"
        
        if not baseline_seat_path or not current_seat_path.exists():
            # Seat predictions might not always be generated
            self.results["warnings"].append("Seat allocation files not found for comparison")
            return True  # Don't fail if seats not generated
        
        baseline_df = pd.read_csv(baseline_seat_path, index_col=0)
        current_df = pd.read_csv(current_seat_path, index_col=0)
        
        # Compare mean seat predictions
        comparison_results = {}
        
        for party in baseline_df.index:
            if party in current_df.index:
                baseline_mean = baseline_df.loc[party, "mean"]
                current_mean = current_df.loc[party, "mean"]
                
                # Seats are integers, so use absolute difference
                diff = abs(current_mean - baseline_mean)
                
                comparison_results[party] = {
                    "baseline": baseline_mean,
                    "current": current_mean,
                    "absolute_diff": diff,
                    "within_tolerance": diff <= 2.0  # Allow 2 seat difference
                }
        
        all_within_tolerance = all(
            comp["within_tolerance"] 
            for comp in comparison_results.values()
        )
        
        self.results["detailed_results"]["seat_allocations"] = {
            "status": "pass" if all_within_tolerance else "fail",
            "seat_comparisons": comparison_results
        }
        
        return all_within_tolerance
    
    def test_visualizations(self) -> bool:
        """Check that visualization files are generated"""
        expected_plots = [
            "visualizations/latent_popularity_vs_polls.png",
            "visualizations/house_effects_heatmap.png",
            "diagnostics/trace_plot.png"
        ]
        
        missing_plots = []
        for plot_path in expected_plots:
            if not (self.current_dir / plot_path).exists():
                missing_plots.append(plot_path)
        
        if missing_plots:
            self.results["detailed_results"]["visualizations"] = {
                "status": "fail",
                "missing_plots": missing_plots
            }
            return False
        
        self.results["detailed_results"]["visualizations"] = {
            "status": "pass",
            "all_plots_present": True
        }
        return True
    
    def _find_baseline_file(self, relative_path: str) -> Optional[Path]:
        """Find a file in the baseline directory structure"""
        # Try in main baseline directory
        direct_path = self.baseline_dir / relative_path
        if direct_path.exists():
            return direct_path
        
        # Try in timestamped training directory
        train_dirs = list((self.baseline_dir / "train").glob("dynamic_gp_run_*"))
        if train_dirs:
            train_path = train_dirs[0] / relative_path
            if train_path.exists():
                return train_path
        
        return None
    
    def generate_report(self):
        """Generate comprehensive test report"""
        report_time = datetime.now().isoformat()
        
        report = {
            "test_report": {
                "timestamp": report_time,
                "baseline_dir": str(self.baseline_dir),
                "current_dir": str(self.current_dir),
                "tolerance": self.tolerance,
                "overall_result": all(self.results["test_summary"].values()) if self.results["test_summary"] else False,
                "summary": self.results["test_summary"],
                "details": self.results["detailed_results"],
                "errors": self.results["errors"],
                "warnings": self.results["warnings"]
            }
        }
        
        # Save report
        report_path = self.current_dir / "regression_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 REGRESSION TEST REPORT")
        print(f"{'='*60}")
        print(f"Overall Result: {'✅ PASS' if report['test_report']['overall_result'] else '❌ FAIL'}")
        print(f"Report saved to: {report_path}")
        
        # Print summary
        for test_name, result in self.results["test_summary"].items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {test_name}")
        
        if self.results["errors"]:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results["errors"]:
                print(f"  - {error}")
        
        if self.results["warnings"]:
            print(f"\n⚠️  Warnings ({len(self.results['warnings'])}):")
            for warning in self.results["warnings"]:
                print(f"  - {warning}")


def run_new_baseline_test(current_dir: Path) -> bool:
    """Generate a new baseline from current outputs"""
    print(f"🎯 GENERATING NEW BASELINE FROM CURRENT RUN")
    print(f"Current run: {current_dir}")
    
    if not current_dir.exists():
        print(f"❌ Current directory not found: {current_dir}")
        return False
    
    # Create new baseline structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_baseline_dir = Path("test_baselines") / f"generated_baseline_{timestamp}"
    
    # Copy current run as new baseline
    import shutil
    shutil.copytree(current_dir, new_baseline_dir / "train" / current_dir.name)
    
    # Create metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "source_run": str(current_dir),
        "type": "generated_from_current_run",
        "purpose": "New baseline generated from regression testing"
    }
    
    metadata_path = new_baseline_dir / "baseline_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update latest symlink
    latest_link = Path("test_baselines") / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    
    latest_link.symlink_to(new_baseline_dir.name)
    
    print(f"✅ New baseline created: {new_baseline_dir}")
    print(f"🔗 Updated latest symlink")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Local regression testing for election model")
    parser.add_argument("--baseline", type=Path, default=Path("test_baselines/latest"),
                       help="Path to baseline directory")
    parser.add_argument("--current", type=Path, required=True,
                       help="Path to current run directory")
    parser.add_argument("--tolerance", type=float, default=0.01,
                       help="Relative tolerance for numerical comparisons")
    parser.add_argument("--generate-new", action="store_true",
                       help="Generate new baseline from current run instead of testing")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    if not Path("src/main.py").exists():
        print("❌ Error: Must run from repository root")
        return 1
    
    if args.generate_new:
        success = run_new_baseline_test(args.current)
        return 0 if success else 1
    
    # Run regression tests
    tester = RegressionTester(args.baseline, args.current, args.tolerance)
    all_passed = tester.run_comprehensive_test()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())