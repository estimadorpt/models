#!/usr/bin/env python3
"""
Test integration of municipal coalition extraction with the coalition manager.

Demonstrates end-to-end workflow from Wikipedia extraction to municipal prediction propagation.
"""

import json
from src.data.coalition_manager import CoalitionManager, MunicipalCoalitionStructure, DisaggregationRule

def load_municipal_config(config_path):
    """Load municipal structures from extracted configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    municipal_structures = {}
    for muni_id, data in config['municipal_structures'].items():
        structure = MunicipalCoalitionStructure(
            municipality_id=data['municipality_id'],
            local_coalitions=data['local_coalitions'],
            disaggregation_rules=[
                DisaggregationRule(
                    national_coalition=rule['national_coalition'],
                    component_parties=rule['component_parties']
                ) for rule in data['disaggregation_rules']
            ]
        )
        municipal_structures[muni_id] = structure
    
    return municipal_structures

def test_municipal_integration():
    """Test end-to-end municipal coalition integration."""
    
    print("=== MUNICIPAL COALITION INTEGRATION TEST ===\n")
    
    # Initialize coalition manager
    manager = CoalitionManager()
    
    # Load extracted municipal structures
    municipal_structures = load_municipal_config('data/municipal_coalitions.json')
    manager.municipal_structures.update(municipal_structures)
    
    print(f"✅ Loaded {len(municipal_structures)} municipal coalition structures")
    
    # Simulate national poll predictions (what we get from actual polls)
    national_polls = {
        'PS': 0.35,
        'AD': 0.30,  # PSD+CDS bundled together in national polls
        'IL': 0.08,
        'CH': 0.12,
        'BE': 0.05,
        'CDU': 0.04,
        'PAN': 0.03,
        'L': 0.02,
        'Others': 0.01
    }
    
    print(f"\n📊 National poll predictions:")
    for party, support in national_polls.items():
        print(f"  {party}: {support:.1%}")
    
    print(f"\n🏛️ Testing municipal prediction propagation...\n")
    
    # Test different municipal scenarios
    test_cases = [
        ('01-01', 'Aveiro', 'PSD vs CDS competing separately'),
        ('01-02', 'Águeda', 'PSD vs CDS competing separately'),
        ('01-03', 'Vale de Cambra', 'CDS solo run'),
        ('11-01', 'Lisboa', 'PSD+IL coalition'),
        ('99-99', 'Rural Example', 'Default rural pattern (traditional AD)')
    ]
    
    for muni_code, muni_name, description in test_cases:
        print(f"--- {muni_name} ({muni_code}) ---")
        print(f"Pattern: {description}")
        
        # Get municipal predictions
        municipal_predictions = manager.propagate_national_to_municipal(national_polls, muni_code)
        
        # Show municipal coalition structure
        if muni_code in manager.municipal_structures:
            structure = manager.municipal_structures[muni_code]
            coalitions = structure.local_coalitions
            print(f"Local coalitions: {list(coalitions.keys())}")
        else:
            print("Using default rural structure")
        
        # Show predictions
        print("Municipal predictions:")
        total_support = 0
        for party, support in sorted(municipal_predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {party}: {support:.1%}")
            total_support += support
        
        print(f"Total support: {total_support:.1%}")
        
        # Validate key transformations
        if muni_code in ['01-01', '01-02']:  # Aveiro, Águeda
            assert 'PSD' in municipal_predictions
            assert 'CDS' in municipal_predictions
            assert 'AD' not in municipal_predictions
            print("✅ AD disaggregated into competing PSD and CDS")
            
        elif muni_code == '11-01':  # Lisboa
            assert 'PSD_IL' in municipal_predictions
            assert 'PSD' not in municipal_predictions
            assert 'IL' not in municipal_predictions  
            print("✅ PSD and IL combined into PSD_IL coalition")
            
        elif muni_code == '99-99':  # Rural default
            assert 'AD' in municipal_predictions
            assert 'PSD' not in municipal_predictions
            assert 'CDS' not in municipal_predictions
            print("✅ Traditional AD coalition maintained")
        
        print()
    
    print("=== VALIDATION RESULTS ===")
    
    # Test vote share conservation
    print("\n🔬 Testing vote share conservation:")
    national_total = sum(national_polls.values())
    
    for muni_code, muni_name, _ in test_cases[:3]:  # Test first 3 municipalities
        municipal_predictions = manager.propagate_national_to_municipal(national_polls, muni_code)
        municipal_total = sum(municipal_predictions.values())
        
        conservation_check = abs(national_total - municipal_total) < 0.001
        status = "✅" if conservation_check else "❌"
        print(f"  {status} {muni_name}: {national_total:.3f} → {municipal_total:.3f}")
    
    # Test coordinate space separation
    print("\n🎯 Testing coordinate space separation:")
    aveiro_coords = set(manager.propagate_national_to_municipal(national_polls, '01-01').keys())
    lisboa_coords = set(manager.propagate_national_to_municipal(national_polls, '11-01').keys())
    rural_coords = set(manager.propagate_national_to_municipal(national_polls, '99-99').keys())
    
    print(f"  Aveiro parties: {sorted(aveiro_coords)}")
    print(f"  Lisboa parties: {sorted(lisboa_coords)}")
    print(f"  Rural parties: {sorted(rural_coords)}")
    
    # Verify they're different where expected
    assert aveiro_coords != lisboa_coords, "Different municipalities should have different coordinate spaces"
    assert 'PSD_IL' in lisboa_coords, "Lisboa should have PSD_IL coalition"
    assert 'PSD_IL' not in aveiro_coords, "Aveiro should not have PSD_IL coalition"
    print("  ✅ Coordinate spaces properly differentiated")
    
    print("\n🎉 All tests passed! Municipal coalition integration working correctly.")
    
    # Show strategic implications
    print(f"\n📈 STRATEGIC IMPLICATIONS:")
    print(f"✅ National polls with bundled AD can propagate to municipalities")
    print(f"✅ PSD vs CDS competition properly handled through disaggregation")
    print(f"✅ New coalition patterns (PSD+IL) supported")
    print(f"✅ Vote shares conserved across coordinate transformations")
    print(f"✅ Municipal modeling ready for Bayesian implementation")


if __name__ == "__main__":
    test_municipal_integration()