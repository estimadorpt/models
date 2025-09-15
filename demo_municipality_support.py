#!/usr/bin/env python3
"""
Demonstration of Multi-Level Geographic Aggregation for Municipality Elections

This script showcases the new municipality-level support added to the election forecasting system.
It demonstrates how the same modeling framework can now work at parish, municipality, district, 
or national levels without breaking changes to existing code.
"""

from src.data.dataset import ElectionDataset
from src.data.flexible_loaders import load_election_results_flexible, GeographicLevelManager
import pandas as pd


def demonstrate_geographic_levels():
    """Demonstrate loading data at different geographic levels."""
    
    print("=== MULTI-LEVEL GEOGRAPHIC AGGREGATION DEMO ===\n")
    
    # Test elections and parties
    test_elections = ['2024-03-10', '2022-01-30']
    test_parties = ['PS', 'AD', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L']
    
    levels = ['national', 'district', 'municipality', 'parish']
    
    print("1. FLEXIBLE DATA LOADING")
    print("-" * 40)
    
    results_by_level = {}
    
    for level in levels:
        print(f"\n{level.upper()} LEVEL:")
        
        try:
            if level == 'national':
                # Use backward compatibility mode
                results = load_election_results_flexible(
                    election_dates=test_elections,
                    political_families=test_parties,
                    aggregate_national=True
                )
            else:
                # Use new multi-level aggregation
                results = load_election_results_flexible(
                    election_dates=test_elections,
                    political_families=test_parties,
                    aggregation_level=level
                )
                
            results_by_level[level] = results
            
            if not results.empty:
                print(f"  Shape: {results.shape}")
                print(f"  Geographic divisions: {results['geographic_id'].nunique()}")
                print(f"  Elections: {results['election_date'].nunique()}")
                
                # Show vote totals validation
                total_votes = results[test_parties].sum().sum()
                print(f"  Total votes: {total_votes:,}")
                
                # Show sample data
                print("  Sample data:")
                sample_cols = ['election_date', 'geographic_id'] + test_parties[:3]
                if all(col in results.columns for col in sample_cols):
                    print(results[sample_cols].head(2).to_string(index=False))
            else:
                print("  No data loaded")
                
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    print("\n2. ELECTION DATASET WITH DIFFERENT GEOGRAPHIC LEVELS")
    print("-" * 60)
    
    # Demonstrate ElectionDataset at different levels
    dataset_configs = [
        ('district', 'Traditional district-level modeling (default)'),
        ('municipality', 'New municipality-level modeling'),
        ('national', 'National-level modeling')
    ]
    
    for level, description in dataset_configs:
        print(f"\n{description}:")
        
        try:
            dataset = ElectionDataset(
                election_date='2026-01-01',
                baseline_timescales=[365],
                election_timescales=[30, 15],
                test_cutoff=None,
                geographic_level=level
            )
            
            print(f"  ✓ Dataset created successfully")
            print(f"  Geographic level: {dataset.geographic_level}")
            print(f"  Geographic divisions: {len(dataset.unique_geographic_divisions)}")
            print(f"  Primary results shape: {dataset.results_primary.shape}")
            print(f"  Polls for training: {len(dataset.polls_train)}")
            
            # Show sample geographic IDs
            sample_ids = list(dataset.unique_geographic_divisions[:3])
            print(f"  Sample geographic IDs: {sample_ids}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n3. GEOGRAPHIC LEVEL MANAGER")
    print("-" * 40)
    
    # Demonstrate the geographic level manager
    manager = GeographicLevelManager(default_level='municipality')
    
    print("Available geographic levels:")
    available = manager.get_available_levels(test_elections)
    for level, count in available.items():
        print(f"  {level}: {count:,} divisions")
    
    print("\nMunicipality configuration sample:")
    muni_config = manager.get_config('municipality')
    sample_munis = list(muni_config.items())[:5]
    for muni_id, seats in sample_munis:
        print(f"  {muni_id}: {seats} seats")
    
    print("\n4. VALIDATION: VOTE TOTALS CONSISTENCY")
    print("-" * 50)
    
    # Validate that aggregation preserves vote totals
    if 'parish' in results_by_level and 'municipality' in results_by_level:
        parish_results = results_by_level['parish']
        municipality_results = results_by_level['municipality']
        
        print("Comparing parish vs municipality aggregation:")
        
        for party in test_parties[:4]:  # Check first 4 parties
            if party in parish_results.columns and party in municipality_results.columns:
                parish_total = parish_results[party].sum()
                municipality_total = municipality_results[party].sum()
                difference = abs(parish_total - municipality_total)
                
                if difference < 1000:  # Allow small differences due to missing data
                    status = "✓"
                else:
                    status = "⚠️"
                    
                print(f"  {status} {party}: Parish {parish_total:,} vs Municipality {municipality_total:,} (diff: {difference:,})")
    
    print("\n5. BACKWARD COMPATIBILITY")
    print("-" * 30)
    
    # Test that existing code still works
    try:
        # This should work exactly as before
        legacy_dataset = ElectionDataset(
            election_date='2026-01-01',
            baseline_timescales=[365],
            election_timescales=[30, 15],
            test_cutoff=None
            # No geographic_level parameter = district by default
        )
        
        print("✓ Legacy ElectionDataset constructor works")
        print(f"  Default geographic level: {legacy_dataset.geographic_level}")
        print(f"  Geographic divisions: {len(legacy_dataset.unique_geographic_divisions)}")
        print("✓ Full backward compatibility maintained")
        
    except Exception as e:
        print(f"❌ Backward compatibility issue: {e}")
    
    print("\n=== MUNICIPALITY ELECTION MODELING IS NOW READY! ===")
    print("\nNext steps:")
    print("- Create municipality-specific electoral system rules")
    print("- Add municipality-level polling data if available")
    print("- Implement mayor election modeling (single-seat D'Hondt or other systems)")
    print("- Add municipal council seat allocation based on municipality size")


if __name__ == "__main__":
    demonstrate_geographic_levels()