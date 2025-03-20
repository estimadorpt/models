import os
import pytest
import tempfile
import shutil
from datetime import datetime
import argparse

from src.main import fit_model, load_model, cross_validate
from src.config import DEFAULT_BASELINE_TIMESCALE, DEFAULT_ELECTION_TIMESCALES

@pytest.fixture
def test_args():
    """Create a temporary directory and basic args for testing"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Basic args that all tests will use
    args = argparse.Namespace(
        election_date='2024-03-10',  # Most recent election
        baseline_timescales=[DEFAULT_BASELINE_TIMESCALE],
        election_timescales=DEFAULT_ELECTION_TIMESCALES,
        draws=2,  # Minimal draws for quick testing
        tune=2,   # Minimal tuning for quick testing
        debug=True,
        fast=True,  # Skip plots and diagnostics
        notify=False,
        cutoff_date=None,
        output_dir=temp_dir
    )
    
    yield args
    
    # Cleanup after tests
    shutil.rmtree(temp_dir)

@pytest.mark.optional
def test_fit_mode(test_args):
    """Test that fit mode runs without errors"""
    test_args.mode = 'fit'
    
    # Run fit_model
    model = fit_model(test_args)
    
    # Basic assertions
    assert model is not None
    assert model.trace is not None
    assert os.path.exists(os.path.join(test_args.output_dir, 'trace.zarr'))

@pytest.mark.optional
def test_load_mode(test_args):
    """Test that load mode runs without errors"""
    # First fit a model to load
    test_args.mode = 'fit'
    fitted_model = fit_model(test_args)
    assert fitted_model is not None
    
    # Now test loading
    test_args.mode = 'load'
    test_args.load_dir = test_args.output_dir
    
    loaded_model = load_model(test_args)
    
    # Basic assertions
    assert loaded_model is not None
    assert loaded_model.trace is not None
    assert hasattr(loaded_model.model, 'model')
    assert loaded_model.model.model is not None

@pytest.mark.optional
def test_cross_validate_mode(test_args):
    """Test that cross-validation mode runs without errors"""
    test_args.mode = 'cross-validate'
    
    # Run cross-validation
    cv_results = cross_validate(test_args)
    
    # Basic assertions
    assert cv_results is not None
    assert len(cv_results) > 0
    assert os.path.exists(os.path.join(test_args.output_dir, 'cross_validation'))
    assert os.path.exists(os.path.join(test_args.output_dir, 'cross_validation', 'cross_validation_results.csv'))

def test_all_modes_sequential(test_args):
    """Test all modes in sequence to ensure they work together"""
    # 1. First fit a model
    test_args.mode = 'fit'
    fitted_model = fit_model(test_args)
    assert fitted_model is not None
    
    # 2. Then load it
    test_args.mode = 'load'
    test_args.load_dir = test_args.output_dir
    loaded_model = load_model(test_args)
    assert loaded_model is not None
    
    # 3. Finally run cross-validation
    test_args.mode = 'cross-validate'
    cv_results = cross_validate(test_args)
    assert cv_results is not None 