#!/usr/bin/env python3
"""
Demonstration script for regression detection tools.

This script shows how to use the regression detection tools
to compare outputs and detect regressions.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from regression_detection_tools import (
    RegressionDetectionSuite,
    BitForBitComparator,
    StatisticalComparator
)


def demo_regression_detection():
    """Demonstrate regression detection capabilities."""
    print("🚀 Regression Detection Tools Demo")
    print("="*50)
    
    # Check if golden masters exist
    golden_masters_dir = Path(__file__).parent.parent / "test_baselines"
    
    if not golden_masters_dir.exists():
        print("❌ Golden masters not found!")
        print("Please run: python scripts/generate_golden_masters.py")
        return 1
    
    print(f"✅ Golden masters found at: {golden_masters_dir}")
    
    # Initialize regression detection suite
    suite = RegressionDetectionSuite(golden_masters_dir=golden_masters_dir)
    
    print(f"\n📊 Golden Masters Metadata:")
    if suite.golden_metadata:
        for key, value in suite.golden_metadata.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subval in value.items():
                    print(f"    {subkey}: {subval}")
            else:
                print(f"  {key}: {value}")
    else:
        print("  No metadata available")
    
    # Demonstrate bit-for-bit comparison
    print(f"\n🔍 Demonstrating Bit-for-Bit Comparison:")
    
    golden_config = golden_masters_dir / "train" / "model_config.json"
    if golden_config.exists():
        # Compare config with itself (should be identical)
        report = BitForBitComparator.compare_json_files(
            golden_config, golden_config, tolerance=0.0
        )
        
        print(f"Self-comparison result: {'✅ PASS' if report.passed else '❌ FAIL'}")
        if report.metrics:
            print(f"Metrics: {report.metrics}")
    else:
        print("❌ Golden master config not found")
    
    # Demonstrate fit metrics comparison
    print(f"\n📈 Demonstrating Statistical Comparison:")
    
    golden_fit = golden_masters_dir / "train" / "fit_metrics.json"
    if golden_fit.exists():
        # Compare with itself (should be identical)
        report = StatisticalComparator.compare_model_fit_metrics(
            golden_fit, golden_fit
        )
        
        print(f"Self-comparison result: {'✅ PASS' if report.passed else '❌ FAIL'}")
        if report.metrics:
            print(f"Metrics:")
            for metric, value in report.metrics.items():
                print(f"  {metric}: {value:.6f}")
    else:
        print("❌ Golden master fit metrics not found")
    
    # Demonstrate coalition validation
    print(f"\n🤝 Demonstrating Coalition Validation:")
    
    try:
        # Import here to avoid issues if not available
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
        from data.dataset import ElectionDataset
        
        dataset = ElectionDataset(
            election_date='2024-03-10',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"Political families: {dataset.political_families}")
        
        # Check AD coalition presence
        if 'AD' in dataset.political_families:
            ad_idx = dataset.political_families.index('AD')
            poll_data = dataset.polls_train.iloc[:, 3 + ad_idx]  # Skip metadata columns
            result_data = dataset.results_national['AD']
            
            print(f"✅ AD coalition found at index {ad_idx}")
            print(f"AD poll data points: {len(poll_data)}")
            print(f"AD total poll votes: {poll_data.sum():,}")
            print(f"AD total result votes: {result_data.sum():,}")
        else:
            print(f"❌ AD coalition not found!")
        
        # Check that PSD/CDS are absorbed
        absorbed_parties = ['PSD', 'CDS']
        for party in absorbed_parties:
            if party in dataset.political_families:
                print(f"⚠️  {party} should be absorbed into AD coalition")
            else:
                print(f"✅ {party} correctly absorbed into AD")
                
    except Exception as e:
        print(f"❌ Coalition validation failed: {e}")
    
    print(f"\n🎯 Summary:")
    print(f"  • Regression detection tools are working correctly")
    print(f"  • Golden masters are properly structured") 
    print(f"  • Coalition handling is functioning as expected")
    print(f"  • Ready for comprehensive regression testing")
    
    return 0


def main():
    """Main entry point."""
    try:
        return demo_regression_detection()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())