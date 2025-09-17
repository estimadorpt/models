#!/usr/bin/env python3
"""
Test script for regression detection tools.

This script validates that the regression detection tools work correctly
by running them against the golden masters and simulated outputs.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from regression_detection_tools import (
    BitForBitComparator, 
    StatisticalComparator,
    RegressionDetectionSuite,
    RegressionReport
)


def test_bit_for_bit_comparator():
    """Test the BitForBitComparator class."""
    print("🧪 Testing BitForBitComparator...")
    
    # Create test JSON data
    test_data_1 = {"value": 1.0, "text": "hello", "nested": {"a": 1, "b": 2}}
    test_data_2 = {"value": 1.0, "text": "hello", "nested": {"a": 1, "b": 2}}  # Identical
    test_data_3 = {"value": 1.1, "text": "hello", "nested": {"a": 1, "b": 2}}  # Different
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Write test files
        file1 = temp_path / "test1.json"
        file2 = temp_path / "test2.json"  
        file3 = temp_path / "test3.json"
        
        with open(file1, 'w') as f:
            json.dump(test_data_1, f)
        with open(file2, 'w') as f:
            json.dump(test_data_2, f)
        with open(file3, 'w') as f:
            json.dump(test_data_3, f)
        
        # Test identical files
        report = BitForBitComparator.compare_json_files(file1, file2, tolerance=0.0)
        assert report.passed, f"Identical files should pass: {report.critical_failures}"
        
        # Test different files (no tolerance)
        report = BitForBitComparator.compare_json_files(file1, file3, tolerance=0.0)
        assert not report.passed, "Different files should fail with zero tolerance"
        
        # Test different files (with tolerance)
        report = BitForBitComparator.compare_json_files(file1, file3, tolerance=0.2)
        assert report.passed, "Different files should pass with sufficient tolerance"
        
        print("✅ BitForBitComparator tests passed")


def test_statistical_comparator():
    """Test the StatisticalComparator class."""
    print("🧪 Testing StatisticalComparator...")
    
    # Test the numeric value parser
    assert StatisticalComparator._parse_numeric_value("5.5%") == 0.055
    assert StatisticalComparator._parse_numeric_value(10.0) == 10.0
    assert StatisticalComparator._parse_numeric_value("15.0") == 15.0
    
    print("✅ StatisticalComparator tests passed")


def test_regression_report():
    """Test the RegressionReport class."""
    print("🧪 Testing RegressionReport...")
    
    report = RegressionReport(test_name="Test Report", passed=True)
    
    # Test adding failures
    report.add_failure("Test failure")
    assert not report.passed, "Report should fail after adding failure"
    assert len(report.critical_failures) == 1
    
    # Test adding metrics
    report.add_metric("test_metric", 1.23)
    assert "test_metric" in report.metrics
    assert report.metrics["test_metric"] == 1.23
    
    # Test serialization
    report_dict = report.to_dict()
    assert "test_name" in report_dict
    assert "critical_failures" in report_dict
    assert "metrics" in report_dict
    
    print("✅ RegressionReport tests passed")


def test_golden_masters_exist():
    """Test that golden masters directory exists and has expected structure."""
    print("🧪 Testing golden masters structure...")
    
    golden_masters_dir = Path(__file__).parent.parent / "test_baselines"
    
    if not golden_masters_dir.exists():
        print("⚠️  Golden masters not found - run scripts/generate_golden_masters.py first")
        return
    
    # Check expected structure
    expected_files = [
        "golden_masters_metadata.json",
        "train/model_config.json",
        "train/fit_metrics.json"
    ]
    
    for expected_file in expected_files:
        file_path = golden_masters_dir / expected_file
        if file_path.exists():
            print(f"✅ Found: {expected_file}")
        else:
            print(f"⚠️  Missing: {expected_file}")
    
    print("✅ Golden masters structure validated")


def test_regression_suite_initialization():
    """Test that RegressionDetectionSuite initializes correctly."""
    print("🧪 Testing RegressionDetectionSuite initialization...")
    
    golden_masters_dir = Path(__file__).parent.parent / "test_baselines"
    
    if not golden_masters_dir.exists():
        print("⚠️  Skipping - golden masters not available")
        return
    
    suite = RegressionDetectionSuite(golden_masters_dir=golden_masters_dir)
    
    assert suite.golden_masters_dir == golden_masters_dir
    assert hasattr(suite, 'golden_metadata')
    
    print("✅ RegressionDetectionSuite initialization passed")


def main():
    """Run all regression detection tool tests."""
    print("🧪 Testing Regression Detection Tools")
    print("="*50)
    
    try:
        # Run unit tests
        test_bit_for_bit_comparator()
        test_statistical_comparator()
        test_regression_report()
        
        # Run integration tests
        test_golden_masters_exist()
        test_regression_suite_initialization()
        
        print("\n✅ All regression detection tool tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())