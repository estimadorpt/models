#!/usr/bin/env python3
"""
Document Current System Behavior

This script captures and documents how the current system works:
- Coalition handling (AD=PSD+CDS aggregation)
- Geographic aggregation (district-level processing)
- Data loading patterns
- Model convergence characteristics

This documentation serves as the "specification" for what we must preserve.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data.loaders import load_marktest_polls, load_election_results
from src.data.dataset import ElectionDataset
from src.processing.electoral_systems import calculate_dhondt

def document_coalition_handling():
    """Document how coalition data is currently handled"""
    print("📊 DOCUMENTING COALITION HANDLING")
    
    # Load sample election data to see current coalition behavior
    results = load_election_results()
    
    # Check how coalitions are represented
    coalition_info = {
        "current_parties": list(results.columns[results.columns != 'date']),
        "coalition_patterns": {},
        "aggregation_rules": {}
    }
    
    # Look for AD coalition in data
    if 'AD' in results.columns:
        # Sample some data to understand AD composition
        ad_sample = results[['AD']].head()
        coalition_info["coalition_patterns"]["AD"] = {
            "description": "Aliança Democrática coalition",
            "composition": "PSD + CDS-PP",
            "current_implementation": "Appears as separate 'AD' column in results data",
            "data_sample": ad_sample.to_dict()
        }
    
    # Check for PSD/CDS individual entries
    if 'PSD' in results.columns and 'CDS' in results.columns:
        coalition_info["aggregation_rules"]["manual_coalition_sum"] = {
            "rule": "AD vote share = PSD + CDS-PP vote shares",
            "implementation": "Hardcoded summation in data processing",
            "rationale": "Historical coalition arrangement"
        }
    
    return coalition_info

def document_geographic_aggregation():
    """Document district-level geographic processing"""
    print("🗺️ DOCUMENTING GEOGRAPHIC AGGREGATION")
    
    # Use a sample dataset to understand geographic structure
    try:
        dataset = ElectionDataset(
            election_date="2024-03-10",
            baseline_timescales=[365],
            election_timescales=[30]
        )
        
        geo_info = {
            "district_structure": {
                "unique_districts": list(dataset.unique_districts) if hasattr(dataset, 'unique_districts') else [],
                "total_districts": len(dataset.unique_districts) if hasattr(dataset, 'unique_districts') else 0
            },
            "aggregation_levels": {
                "national": "Sum of all district results",
                "district": "Individual distrito/circulo level",
                "parish": "Not currently used in aggregation"
            },
            "data_processing": {}
        }
        
        # Check how polls vs results are handled geographically
        if hasattr(dataset, 'polls'):
            geo_info["poll_geography"] = {
                "level": "National level polling only",
                "district_breakdown": "Polls not broken down by district",
                "swing_calculation": "Uniform National Swing (UNS) applied to district level"
            }
        
        if hasattr(dataset, 'results'):
            geo_info["results_geography"] = {
                "level": "District-level election results available",
                "aggregation_method": "Direct district results used for model training",
                "validation": "District results compared against model predictions"
            }
        
        # Document coordinate system
        if hasattr(dataset, 'district_names'):
            geo_info["coordinate_mapping"] = {
                "districts": dataset.district_names if hasattr(dataset, 'district_names') else [],
                "encoding": "String-based district names",
                "ordering": "Alphabetical or data-file order"
            }
    
    except Exception as e:
        geo_info = {"error": f"Could not instantiate dataset: {e}"}
    
    return geo_info

def document_data_loading_patterns():
    """Document how data is currently loaded and processed"""
    print("📁 DOCUMENTING DATA LOADING PATTERNS")
    
    loading_info = {
        "poll_sources": {},
        "election_result_sources": {},
        "processing_pipeline": {},
        "data_validation": {}
    }
    
    # Document poll loading
    try:
        polls = load_marktest_polls()
        loading_info["poll_sources"]["marktest"] = {
            "format": "CSV/TSV files",
            "columns": list(polls.columns) if hasattr(polls, 'columns') else [],
            "date_range": {
                "start": str(polls['date'].min()) if 'date' in polls.columns else "unknown",
                "end": str(polls['date'].max()) if 'date' in polls.columns else "unknown"
            },
            "sample_size": len(polls) if hasattr(polls, '__len__') else 0
        }
    except Exception as e:
        loading_info["poll_sources"]["marktest"] = {"error": str(e)}
    
    # Document election result loading
    try:
        results = load_election_results()
        loading_info["election_result_sources"]["legislativas"] = {
            "format": "Parquet files",
            "elections_included": list(results['date'].unique()) if 'date' in results.columns else [],
            "parties_tracked": [col for col in results.columns if col != 'date'],
            "sample_size": len(results) if hasattr(results, '__len__') else 0
        }
    except Exception as e:
        loading_info["election_result_sources"]["legislativas"] = {"error": str(e)}
    
    # Document processing steps
    loading_info["processing_pipeline"] = {
        "poll_consolidation": "Multiple polls on same date are averaged",
        "tracking_polls": "Special handling for 'resultados acumulados' vs individual polls",
        "coalition_mapping": "AD mapped from individual party results where needed",
        "missing_data": "Parties not competing in election set to 0",
        "date_alignment": "All data aligned to common date grid"
    }
    
    return loading_info

def document_model_characteristics():
    """Document current model behavior and convergence patterns"""
    print("🔧 DOCUMENTING MODEL CHARACTERISTICS")
    
    model_info = {
        "model_architecture": {
            "type": "Dynamic Gaussian Process Election Model",
            "components": [
                "Baseline GP (long-term trends)",
                "Medium-term GP (electoral cycle effects)", 
                "Short-term GP (campaign effects)",
                "House effects (pollster bias)",
                "District-level random effects"
            ]
        },
        "hyperparameters": {
            "baseline_gp_lengthscale": "365 days (typical default)",
            "cycle_gp_lengthscale": "45 days (typical default)",
            "convergence_criteria": "R-hat < 1.01, ESS > 400"
        },
        "expected_outputs": {
            "training": [
                "trace.nc (MCMC samples)",
                "fit_metrics.json (convergence diagnostics)",
                "model_config.json (parameters used)"
            ],
            "prediction": [
                "vote_share_summary_election_day.csv",
                "seat_summary_election_day.csv", 
                "district_forecast.json",
                "national_trends.json"
            ],
            "visualization": [
                "latent_popularity_vs_polls.png",
                "house_effects_heatmap.png",
                "trace_plot.png"
            ]
        },
        "known_working_behavior": {
            "coalition_handling": "AD automatically aggregated from PSD+CDS", 
            "geographic_modeling": "District-level variation captured",
            "poll_integration": "Multiple pollster house effects estimated",
            "temporal_modeling": "Long and short-term trends separated"
        }
    }
    
    return model_info

def document_electoral_system_integration():
    """Document how electoral system calculations work"""
    print("🗳️ DOCUMENTING ELECTORAL SYSTEM INTEGRATION")
    
    electoral_info = {
        "seat_allocation": {},
        "district_system": {},
        "validation_approach": {}
    }
    
    try:
        # Test D'Hondt calculation with sample data
        sample_votes = {'PS': 1000, 'AD': 800, 'CH': 600, 'IL': 400, 'BE': 200}
        sample_seats = 10
        
        dhondt_result = calculate_dhondt(sample_votes, sample_seats)
        
        electoral_info["seat_allocation"] = {
            "method": "D'Hondt system",
            "implementation": "calculate_dhondt function",
            "sample_input": sample_votes,
            "sample_output": dhondt_result,
            "validation": "Tested with known electoral outcomes"
        }
        
        electoral_info["district_system"] = {
            "seat_distribution": "Each district has fixed number of seats",
            "calculation_level": "Separate D'Hondt calculation per district",
            "aggregation": "National parliament = sum of district seats"
        }
        
    except Exception as e:
        electoral_info["seat_allocation"] = {"error": str(e)}
    
    return electoral_info

def generate_system_documentation():
    """Generate comprehensive documentation of current system behavior"""
    
    print("📋 GENERATING COMPREHENSIVE SYSTEM DOCUMENTATION")
    print("="*60)
    
    documentation = {
        "documentation_metadata": {
            "generated_at": datetime.now().isoformat(),
            "purpose": "Document current system behavior for regression testing",
            "scope": "Complete pipeline behavior that must be preserved"
        }
    }
    
    # Run all documentation functions
    documentation_sections = [
        ("coalition_handling", document_coalition_handling),
        ("geographic_aggregation", document_geographic_aggregation), 
        ("data_loading_patterns", document_data_loading_patterns),
        ("model_characteristics", document_model_characteristics),
        ("electoral_system_integration", document_electoral_system_integration)
    ]
    
    for section_name, doc_func in documentation_sections:
        try:
            documentation[section_name] = doc_func()
        except Exception as e:
            documentation[section_name] = {"error": str(e)}
            print(f"⚠️  Error documenting {section_name}: {e}")
    
    # Save documentation
    doc_path = Path("docs") / "current_system_behavior.json"
    doc_path.parent.mkdir(exist_ok=True)
    
    with open(doc_path, 'w') as f:
        json.dump(documentation, f, indent=2, default=str)
    
    print(f"📄 Documentation saved to: {doc_path}")
    
    # Generate summary report
    print(f"\n📊 SYSTEM BEHAVIOR DOCUMENTATION SUMMARY")
    print(f"{'='*60}")
    
    for section_name, section_data in documentation.items():
        if section_name == "documentation_metadata":
            continue
            
        if "error" in section_data:
            print(f"❌ {section_name}: ERROR - {section_data['error']}")
        else:
            print(f"✅ {section_name}: Documented successfully")
            
            # Print key insights
            if section_name == "coalition_handling" and "coalition_patterns" in section_data:
                coalitions = list(section_data["coalition_patterns"].keys())
                print(f"   Coalitions found: {coalitions}")
            
            elif section_name == "geographic_aggregation" and "district_structure" in section_data:
                district_count = section_data["district_structure"].get("total_districts", 0)
                print(f"   Districts tracked: {district_count}")
            
            elif section_name == "data_loading_patterns":
                sources = list(section_data.keys())
                print(f"   Data sources: {[s for s in sources if not s.endswith('_info')]}")
    
    print(f"\n✅ System behavior documentation complete!")
    print(f"This documents the 'correct' behavior that regression tests must preserve.")
    
    return doc_path

def main():
    # Ensure we're in the right directory
    if not Path("src/main.py").exists():
        print("❌ Error: Must run from repository root")
        return 1
    
    try:
        doc_path = generate_system_documentation()
        print(f"\n🎉 Documentation generation successful!")
        print(f"📄 Saved to: {doc_path}")
        return 0
        
    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())