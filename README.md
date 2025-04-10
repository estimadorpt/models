# Election Model

A Bayesian hierarchical model for election forecasting based on polling data.

## Project Structure

The project is organized in a modular way to facilitate maintainability and reusability:

```
project/
├── data/               # Data files
├── notebooks/          # Jupyter notebooks for analysis
├── src/                # Source code
│   ├── data/           # Data loading and processing
│   │   ├── loaders.py  # Functions for loading data
│   │   └── dataset.py  # ElectionDataset class
│   ├── models/         # Model implementation
│   │   ├── election_model.py     # Core Bayesian model
│   │   └── elections_facade.py   # Facade for simplifying API
│   ├── visualization/  # Visualization utilities
│   │   └── plots.py    # Plotting functions
│   ├── utils/          # General utilities
│   ├── config.py       # Configuration settings
│   └── main.py         # Main entry point
├── .github/            # GitHub configuration
├── .vscode/            # VSCode configuration
└── environment.yml     # Conda environment specification
```

## Data Structure

The model uses several data sources:

1. Poll data (marktest_polls.csv, polls_renascenca.tsv, popstar_sondagens_data.csv)
2. Election results (legislativas_*.parquet)
3. Economic indicators (gdp.csv)

## How to Use

### Basic Usage

```python
from src.models.elections_facade import ElectionsFacade

# Create a model for a specific election
model = ElectionsFacade(
    election_date='2024-03-10',
    baseline_timescales=[365],
    election_timescales=[60]
)

# Run inference
prior, trace, posterior = model.run_inference(draws=200, tune=100)

# Generate forecast
prediction = model.generate_forecast()

# Create plots
retro_fig = model.plot_retrodictive_check()
forecast_fig = model.plot_forecast(hdi=True)
```

### Command-line Interface

You can run the model from the command line in different modes. **Make sure to run these commands from the project root directory (the one containing the `src` folder).**

```bash
# Train (fit) a new model for a specific election
python -m src.main --mode train --election-date 2026-01-01 --output-dir outputs --draws 1000 --tune 1000 --notify

# Load a saved model and generate diagnostic plots
# Ensure --load-dir points to the model directory (e.g., outputs/2024-01-01_120000 or outputs/latest)
python -m src.main --mode diagnose --load-dir outputs/latest

# Load a saved model and generate visualization plots (historical fit, etc.)
python -m src.main --mode viz --load-dir outputs/latest

# Load a saved model and generate the election outcome forecast distribution
python -m src.main --mode predict --load-dir outputs/latest

# Visualize the raw input data for a given election context
python -m src.main --mode visualize-data --election-date 2026-01-01 --output-dir outputs/data_viz

# Run cross-validation across historical elections
# Note: --election-date is ignored here; it runs on all past elections found in the data
python -m src.main --mode cross-validate --output-dir outputs/cv_results --draws 500 --tune 500

# Train a model excluding data after a certain date (for retrodictive testing)
python -m src.main --mode train --election-date 2024-03-10 --output-dir outputs --cutoff-date 2024-01-01
```

Available modes:
- `train`: Train a new model and save the results.
- `viz`: Load a saved model and generate visualization plots (historical fit, components, etc.).
- `visualize-data`: Visualize the raw input poll and election data.
- `diagnose`: Load a saved model and generate MCMC diagnostic plots.
- `cross-validate`: Perform cross-validation by fitting the model to past elections.
- `predict`: Load a saved model and generate the election outcome forecast distribution.

Key parameters:
- `--mode`: The operation mode (required). Choices: `train`, `viz`, `visualize-data`, `diagnose`, `cross-validate`, `predict`.
- `--election-date`: Target election date (YYYY-MM-DD). Required for `train`, used for context in `visualize-data`.
- `--output-dir`: Directory to save outputs (models, plots, etc.). Defaults to `outputs/latest`.
- `--load-dir`: Directory to load a saved model from. Required for `viz`, `diagnose`, `predict`.
- `--cutoff-date`: Exclude polling data after this date (YYYY-MM-DD) during training. Useful for retrodictive testing.
- `--baseline-timescale`: Timescale(s) in days for the baseline GP kernel (e.g., `365` or `180 365`). Default: `365.0`.
- `--election-timescale`: Timescale(s) in days for the election-specific GP kernel (e.g., `60` or `30 90`). Default: `[14.0, 60.0]`.
- `--draws`: Number of posterior samples per chain. Default: 1000.
- `--tune`: Number of tuning (warmup) steps per chain. Default: 1000.
- `--target-accept`: Target acceptance rate for NUTS sampler. Default: 0.95.
- `--chains`: Number of MCMC chains (currently unused in `main.py`'s `run_inference` call). Default: 4.
- `--seed`: Random seed for reproducibility. Default: 8675309.
- `--notify`: Send a notification via ntfy.sh upon completion or error.
- `--debug`: Enable detailed diagnostic output.

## Model Description

The model is a Bayesian hierarchical model that incorporates:

1. **Baseline party support**: Long-term average support for each party
2. **Time-varying components**: Using Gaussian Process priors to capture trends over time
3. **House effects**: Systematic biases by polling organizations
4. **Election-specific effects**: Unique dynamics of each election cycle

The model uses PyMC for Bayesian inference, performing MCMC sampling to estimate the posterior distribution of model parameters.

## Visualizations

The model provides various visualizations:

1. Retrodictive check: How well the model fits historical polling data
2. Forecast plot: Predictions for the upcoming election
3. House effects: Systematic biases by polling organizations
4. Party correlations: Relationships between party vote shares
5. Component plots: Visualizations of various model components

## References

- [PyMC Documentation](https://docs.pymc.io/)
- [ArviZ Documentation](https://arviz-devs.github.io/arviz/)