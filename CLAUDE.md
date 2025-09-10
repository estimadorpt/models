# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This is a Python-based Bayesian election forecasting project with multiple environment setup options.

### Recommended Setup: Pixi (Modern, Fast)
```bash
# Install Pixi if not already installed
# macOS/Linux: curl -fsSL https://pixi.sh/install.sh | bash
# Or visit: https://pixi.sh/latest/

# Clone and enter project directory
cd models

# Activate Pixi environment (automatically installs dependencies)
pixi shell

# Run commands using pixi
pixi run test
pixi run train
pixi run viz
```

### Alternative: Conda (Traditional)
```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate models

# Install additional development dependencies if needed
pip install -e .
```

## Development Workflow

### Branching Strategy

**ALWAYS create a feature branch before making changes:**

```bash
# Start any new work
git checkout main
git pull origin main
git checkout -b feature/descriptive-name

# Work on your changes
# ... make changes ...

# When ready to merge
git push origin feature/descriptive-name
# Create PR through GitHub
```

**Branch Naming Convention:**
- `feature/` - New features or enhancements
- `fix/` - Bug fixes  
- `refactor/` - Code refactoring without functionality changes
- `test/` - Adding or improving tests
- `docs/` - Documentation updates

**Critical Rule**: Never commit directly to `main`. Always use feature branches and pull requests.

## Common Commands

### With Pixi (Recommended)
```bash
# Testing
pixi run test              # Run all tests
pixi run test-cov          # Run tests with coverage
pixi run format            # Format code with black
pixi run lint              # Check code formatting
pixi run notebook          # Start Jupyter Lab

# Model operations
pixi run train             # Train default model
pixi run viz               # Visualize latest model
pixi run diagnose          # Diagnose latest model
pixi run predict           # Generate predictions
```

### With Conda/Python
```bash
# Testing
pytest                     # Run all tests
pytest --cov=src           # Run tests with coverage
pytest -m "not optional"   # Run only non-optional tests
pytest tests/test_dataset.py  # Run specific test file
```

### Running the Election Model

All model commands should be run from the project root directory using the module syntax:

```bash
# Train a new model
python -m src.main --mode train --election-date 2026-01-01 --output-dir outputs --draws 1000 --tune 1000

# Generate visualizations from a saved model
python -m src.main --mode viz --load-dir outputs/latest

# Generate diagnostic plots
python -m src.main --mode diagnose --load-dir outputs/latest

# Generate election predictions
python -m src.main --mode predict --load-dir outputs/latest

# Visualize raw data
python -m src.main --mode visualize-data --election-date 2026-01-01 --output-dir outputs/data_viz

# Run cross-validation
python -m src.main --mode cross-validate --output-dir outputs/cv_results --draws 500 --tune 500
```

## Architecture Overview

### Core Components

1. **ElectionsFacade** (`src/models/elections_facade.py`): Main interface that simplifies interaction with the election model system. Provides methods for data loading, model training, inference, and visualization.

2. **Model Classes**:
   - `BaseElectionModel` (`src/models/base_model.py`): Abstract base class for all election models
   - `StaticBaselineElectionModel` (`src/models/static_baseline_election_model.py`): Model with static baseline support
   - `DynamicGPElectionModel` (`src/models/dynamic_gp_election_model.py`): Model with Gaussian Process dynamics

3. **ElectionDataset** (`src/data/dataset.py`): Manages loading and processing of polling data, election results, and economic indicators

4. **Data Processing**:
   - `src/data/loaders.py`: Functions for loading various data sources
   - `src/processing/`: Forecasting, seat prediction, and electoral system calculations

### Model Architecture

The system implements a Bayesian hierarchical model with:
- Baseline party support using long-term averages
- Time-varying components via Gaussian Process priors
- House effects for polling organization biases
- Election-specific dynamics

Key model parameters:
- `baseline_timescales`: GP timescales for baseline trends (default: [365] days)
- `election_timescales`: GP timescales for election-specific trends (default: [14, 60] days)

### Data Sources

The model integrates multiple data types:
- Poll data: marktest_polls.csv, polls_renascenca.tsv, popstar_sondagens_data.csv
- Election results: legislativas_*.parquet files
- Economic indicators: gdp.csv

### Model Modes

- `train`: Fit a new model and save results
- `viz`: Generate visualization plots from saved models
- `diagnose`: Generate MCMC diagnostic plots
- `predict`: Generate election outcome forecasts
- `visualize-data`: Plot raw input data
- `cross-validate`: Perform cross-validation on historical elections

### Key Parameters for Training

- `--draws`: Posterior samples per chain (default: 1000)
- `--tune`: Tuning/warmup steps (default: 1000)
- `--target-accept`: NUTS target acceptance rate (default: 0.95)
- `--cutoff-date`: Exclude data after this date for retrodictive testing
- `--notify`: Send completion notifications via ntfy.sh

## Output Structure

Models are saved to timestamped directories under `outputs/` with a symbolic link at `outputs/latest` pointing to the most recent run.