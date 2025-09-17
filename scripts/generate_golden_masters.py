#!/usr/bin/env python3
"""
Generate golden master baselines for regression testing.

This script runs the full pipeline (train -> predict -> viz) and saves outputs
as reference baselines for future regression testing.
"""

import os
import sys
import subprocess
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ {description} completed successfully")


def generate_golden_masters():
    """Generate golden master baselines."""
    
    # Configuration
    base_dir = Path(__file__).parent.parent
    test_baselines_dir = base_dir / "test_baselines"
    
    # Use a historical election date for consistency (2024-03-10 - Portuguese legislative election)
    historical_election = "2024-03-10"
    
    # Use smaller sample sizes for faster baseline generation
    draws = 500
    tune = 500
    
    print(f"Generating golden masters with election date: {historical_election}")
    print(f"Output directory: {test_baselines_dir}")
    print(f"MCMC settings: {draws} draws, {tune} tune")
    
    # Clean up existing baselines
    if test_baselines_dir.exists():
        print(f"Removing existing baselines at {test_baselines_dir}")
        shutil.rmtree(test_baselines_dir)
    
    # Create fresh directories
    test_baselines_dir.mkdir(exist_ok=True)
    (test_baselines_dir / "train").mkdir(exist_ok=True)
    (test_baselines_dir / "predict").mkdir(exist_ok=True)
    (test_baselines_dir / "viz").mkdir(exist_ok=True)
    
    # Change to model directory
    original_dir = os.getcwd()
    os.chdir(base_dir)
    
    try:
        # Step 1: Train the model
        train_cmd = (
            f"pixi run python -m src.main "
            f"--mode train "
            f"--model-type dynamic_gp "
            f"--election-date {historical_election} "
            f"--output-dir test_baselines/train "
            f"--draws {draws} "
            f"--tune {tune} "
            f"--seed 42"  # Fixed seed for reproducibility
        )
        
        run_command(train_cmd, "Training model for golden masters")
        
        # Find the generated training directory (it will have a timestamp)
        train_dirs = list((test_baselines_dir / "train").glob("dynamic_gp_run_*"))
        if not train_dirs:
            print("ERROR: No training output directory found")
            sys.exit(1)
        
        latest_train_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
        print(f"Training completed in: {latest_train_dir}")
        
        # Step 2: Generate predictions
        predict_cmd = (
            f"pixi run python -m src.main "
            f"--mode predict "
            f"--load-dir {latest_train_dir} "
            f"--output-dir test_baselines/predict"
        )
        
        run_command(predict_cmd, "Generating predictions for golden masters")
        
        # Step 3: Generate visualizations
        viz_cmd = (
            f"pixi run python -m src.main "
            f"--mode viz "
            f"--load-dir {latest_train_dir}"
        )
        
        run_command(viz_cmd, "Generating visualizations for golden masters")
        
        # Step 4: Save metadata about the golden masters
        metadata = {
            "created_at": datetime.now().isoformat(),
            "election_date": historical_election,
            "model_type": "dynamic_gp",
            "draws": draws,
            "tune": tune,
            "seed": 42,
            "train_dir": str(latest_train_dir.relative_to(base_dir)),
            "description": "Golden master baselines for comprehensive regression testing",
            "git_branch": subprocess.run(["git", "branch", "--show-current"], 
                                       capture_output=True, text=True).stdout.strip(),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], 
                                       capture_output=True, text=True).stdout.strip()
        }
        
        metadata_file = test_baselines_dir / "golden_masters_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print("GOLDEN MASTERS GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Metadata saved to: {metadata_file}")
        print(f"Training output: {latest_train_dir}")
        print(f"Predictions output: test_baselines/predict/")
        print(f"Visualizations: {latest_train_dir}/visualizations/")
        print("\nThese baselines can now be used for regression testing.")
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    generate_golden_masters()