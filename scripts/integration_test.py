#!/usr/bin/env python3
"""
Integration Test Suite for Election Model Pipeline

This script runs comprehensive integration tests that validate the full 
train -> predict -> viz pipeline works correctly. It can run against 
golden master baselines or generate new test runs.

Usage:
    python scripts/integration_test.py --mode full --baseline test_baselines/latest
    python scripts/integration_test.py --mode quick --no-baseline
"""

import os
import sys
import argparse
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class IntegrationTestSuite:
    """Comprehensive integration testing for the election model pipeline"""
    
    def __init__(self, baseline_dir: Optional[Path] = None, debug: bool = False):
        self.baseline_dir = baseline_dir
        self.debug = debug
        self.test_dir = None
        self.results = {
            "test_metadata": {},
            "pipeline_tests": {},
            "regression_tests": {},
            "errors": [],
            "warnings": []
        }
    
    def run_full_integration_test(self) -> bool:
        """Run complete integration test suite"""
        print("🧪 ELECTION MODEL INTEGRATION TEST SUITE")
        print("="*60)
        
        # Create temporary test directory
        self.test_dir = Path(tempfile.mkdtemp(prefix="integration_test_"))
        print(f"Test directory: {self.test_dir}")
        
        try:
            # Store test metadata
            self.results["test_metadata"] = {
                "start_time": datetime.now().isoformat(),
                "test_directory": str(self.test_dir),
                "baseline_directory": str(self.baseline_dir) if self.baseline_dir else None,
                "debug_mode": self.debug
            }
            
            # Run pipeline tests
            pipeline_success = self._run_pipeline_tests()
            
            # Run regression tests if baseline available
            regression_success = True
            if self.baseline_dir and self.baseline_dir.exists():
                regression_success = self._run_regression_tests()
            else:
                self.results["warnings"].append("No baseline directory provided, skipping regression tests")
            
            overall_success = pipeline_success and regression_success
            
            # Generate final report
            self._generate_test_report(overall_success)
            
            return overall_success
            
        finally:
            # Cleanup unless debug mode
            if not self.debug and self.test_dir and self.test_dir.exists():
                shutil.rmtree(self.test_dir)
                print(f"🧹 Cleaned up test directory: {self.test_dir}")
    
    def _run_pipeline_tests(self) -> bool:
        """Test the complete train -> predict -> viz pipeline"""
        print("\n📋 RUNNING PIPELINE INTEGRATION TESTS")
        
        # Create temporary test directory if not already created
        if self.test_dir is None:
            self.test_dir = Path(tempfile.mkdtemp(prefix="integration_test_"))
            print(f"Test directory: {self.test_dir}")
        
        # Use smaller parameters for faster testing
        test_params = {
            "election_date": "2024-03-10",
            "model_type": "dynamic_gp", 
            "draws": 200,  # Smaller for speed
            "tune": 200,
            "chains": 2,
            "seed": 12345
        }
        
        pipeline_results = {}
        
        # Test 1: Model Training
        print("\n🔧 Test 1: Model Training Pipeline")
        train_success, train_dir = self._test_model_training(test_params)
        pipeline_results["training"] = {
            "success": train_success,
            "output_directory": str(train_dir) if train_dir else None
        }
        
        if not train_success:
            self.results["pipeline_tests"] = pipeline_results
            return False
        
        # Test 2: Prediction Generation
        print("\n📊 Test 2: Prediction Generation Pipeline") 
        predict_success = self._test_prediction_generation(train_dir)
        pipeline_results["prediction"] = {"success": predict_success}
        
        # Test 3: Visualization Generation
        print("\n📈 Test 3: Visualization Pipeline")
        viz_success = self._test_visualization_generation(train_dir)
        pipeline_results["visualization"] = {"success": viz_success}
        
        # Test 4: Output Validation
        print("\n✅ Test 4: Output File Validation")
        validation_success = self._test_output_validation(train_dir)
        pipeline_results["output_validation"] = {"success": validation_success}
        
        self.results["pipeline_tests"] = pipeline_results
        
        overall_pipeline_success = all([
            train_success, predict_success, viz_success, validation_success
        ])
        
        print(f"\n📋 Pipeline Tests: {'✅ PASS' if overall_pipeline_success else '❌ FAIL'}")
        return overall_pipeline_success
    
    def _test_model_training(self, params: Dict) -> Tuple[bool, Optional[Path]]:
        """Test model training pipeline"""
        train_output_dir = self.test_dir / "train_test"
        
        cmd = [
            "pixi", "run", "python", "-m", "src.main",
            "--mode", "train",
            "--model-type", params["model_type"],
            "--election-date", params["election_date"],
            "--output-dir", str(train_output_dir),
            "--draws", str(params["draws"]),
            "--tune", str(params["tune"]),
            "--chains", str(params["chains"]),
            "--seed", str(params["seed"])
        ]
        
        if self.debug:
            cmd.append("--debug")
        
        success = self._run_command(cmd, "Model Training")
        
        if success:
            # Check if the output was saved directly to train_output_dir (for tests)
            if (train_output_dir / "model_config.json").exists():
                actual_train_dir = train_output_dir
                print(f"   Training output: {actual_train_dir}")
                return True, actual_train_dir
            else:
                # Look for timestamped directory (for normal runs)
                actual_dirs = list(train_output_dir.glob(f"{params['model_type']}_run_*"))
                if actual_dirs:
                    actual_train_dir = actual_dirs[0]
                    print(f"   Training output: {actual_train_dir}")
                    return True, actual_train_dir
                else:
                    self.results["errors"].append("Training succeeded but no output directory found")
                    return False, None
        
        return False, None
    
    def _test_prediction_generation(self, train_dir: Path) -> bool:
        """Test prediction generation pipeline"""
        cmd = [
            "pixi", "run", "python", "-m", "src.main",
            "--mode", "predict", 
            "--load-dir", str(train_dir),
            "--prediction-date-mode", "election_day",
            "--seat-prediction-samples", "500",  # Smaller for speed
            "--skip-comparison-in-predict"
        ]
        
        if self.debug:
            cmd.append("--debug")
        
        return self._run_command(cmd, "Prediction Generation")
    
    def _test_visualization_generation(self, train_dir: Path) -> bool:
        """Test visualization generation pipeline"""
        cmd = [
            "pixi", "run", "python", "-m", "src.main",
            "--mode", "viz",
            "--load-dir", str(train_dir)
        ]
        
        if self.debug:
            cmd.append("--debug")
        
        return self._run_command(cmd, "Visualization Generation")
    
    def _test_output_validation(self, train_dir: Path) -> bool:
        """Validate that expected output files were generated"""
        expected_files = [
            "model_config.json",
            "fit_metrics.json",
            "trace.zarr",  # System saves as zarr, not nc
            "predictions/vote_share_summary_election_day.csv",
            "diagnostics/diagnostic_trace_plot_baseline_gp.png",  # Actual diagnostic file name
            "visualizations/latent_popularity_vs_polls.png"
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not (train_dir / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.results["errors"].append(f"Missing output files: {missing_files}")
            print(f"   ❌ Missing files: {missing_files}")
            return False
        
        print(f"   ✅ All expected files present")
        return True
    
    def _run_regression_tests(self) -> bool:
        """Run regression tests against baseline"""
        print("\n🔍 RUNNING REGRESSION TESTS")
        
        # Find the test output directory
        test_runs = list(self.test_dir.glob("train_test/*/"))
        if not test_runs:
            self.results["errors"].append("No test run found for regression testing")
            return False
        
        test_run_dir = test_runs[0]
        
        # Run regression test script
        regression_cmd = [
            "pixi", "run", "python", "scripts/regression_test.py",
            "--baseline", str(self.baseline_dir),
            "--current", str(test_run_dir),
            "--tolerance", "0.05"  # 5% tolerance for test runs
        ]
        
        regression_success = self._run_command(regression_cmd, "Regression Testing")
        
        # Load regression test report if available
        report_path = test_run_dir / "regression_test_report.json"
        if report_path.exists():
            try:
                with open(report_path) as f:
                    regression_report = json.load(f)
                self.results["regression_tests"] = regression_report["test_report"]
            except Exception as e:
                self.results["warnings"].append(f"Could not load regression report: {e}")
        
        return regression_success
    
    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status"""
        print(f"   Running: {description}")
        if self.debug:
            print(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=not self.debug,
                text=True,
                cwd=project_root,
                timeout=1800  # 30 minute timeout
            )
            
            success = result.returncode == 0
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {status}: {description}")
            
            if not success:
                error_msg = f"{description} failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr[:500]}..."
                self.results["errors"].append(error_msg)
            
            return success
            
        except subprocess.TimeoutExpired:
            self.results["errors"].append(f"{description} timed out after 30 minutes")
            return False
        except Exception as e:
            self.results["errors"].append(f"{description} failed with exception: {e}")
            return False
    
    def _generate_test_report(self, overall_success: bool):
        """Generate comprehensive test report"""
        self.results["test_metadata"]["end_time"] = datetime.now().isoformat()
        self.results["test_metadata"]["overall_success"] = overall_success
        
        report_path = self.test_dir / "integration_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📊 INTEGRATION TEST REPORT")
        print("="*60)
        print(f"Overall Result: {'✅ PASS' if overall_success else '❌ FAIL'}")
        print(f"Report: {report_path}")
        
        # Print summary
        if self.results["pipeline_tests"]:
            print("\n📋 Pipeline Tests:")
            for test_name, test_result in self.results["pipeline_tests"].items():
                status = "✅ PASS" if test_result["success"] else "❌ FAIL" 
                print(f"  {status} {test_name}")
        
        if self.results["regression_tests"]:
            print("\n🔍 Regression Tests:")
            if "summary" in self.results["regression_tests"]:
                for test_name, result in self.results["regression_tests"]["summary"].items():
                    status = "✅ PASS" if result else "❌ FAIL"
                    print(f"  {status} {test_name}")
        
        if self.results["errors"]:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        if self.results["warnings"]:
            print(f"\n⚠️  Warnings ({len(self.results['warnings'])}):")
            for warning in self.results["warnings"][:3]:  # Show first 3 warnings
                print(f"  - {warning}")


def main():
    parser = argparse.ArgumentParser(description="Integration test suite for election model")
    parser.add_argument("--mode", choices=["full", "quick"], default="full",
                       help="Test mode: full (with regression) or quick (pipeline only)")
    parser.add_argument("--baseline", type=Path, 
                       help="Path to baseline directory for regression testing")
    parser.add_argument("--no-baseline", action="store_true",
                       help="Skip regression tests even if baseline exists")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output and preserve test directory")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    if not Path("src/main.py").exists():
        print("❌ Error: Must run from repository root")
        return 1
    
    # Set up baseline directory
    baseline_dir = None
    if not args.no_baseline:
        if args.baseline:
            baseline_dir = args.baseline
        elif Path("test_baselines/latest").exists():
            baseline_dir = Path("test_baselines/latest")
        
        if baseline_dir and not baseline_dir.exists():
            print(f"⚠️  Baseline directory not found: {baseline_dir}")
            baseline_dir = None
    
    # Run integration tests
    tester = IntegrationTestSuite(baseline_dir, args.debug)
    
    try:
        if args.mode == "quick":
            # Quick mode: just pipeline tests
            success = tester._run_pipeline_tests()
            # Generate a quick report for pipeline tests
            if tester.test_dir:
                tester._generate_test_report(success)
        else:
            # Full mode: pipeline + regression tests
            success = tester.run_full_integration_test()
        
        return 0 if success else 1
        
    finally:
        # Cleanup for quick mode if needed
        if args.mode == "quick" and not args.debug and tester.test_dir and tester.test_dir.exists():
            shutil.rmtree(tester.test_dir)
            print(f"🧹 Cleaned up test directory: {tester.test_dir}")


if __name__ == "__main__":
    sys.exit(main())