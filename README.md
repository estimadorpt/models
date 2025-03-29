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
# Fit a new model for a future election (using all available data)
python -m src.main --mode fit --election-date 2024-03-10 --draws 1000 --tune 1000

# Load a previously saved model and generate forecasts
# Note: Ensure --load-dir points to the correct path relative to the project root
python -m src.main --mode load --election-date 2024-03-10 --load-dir outputs/your-model-directory

# Run retrodictive testing with data cutoff
python -m src.main --mode fit --election-date 2024-03-10 --cutoff-date 2024-01-01

# Run cross-validation across past elections
python -m src.main --mode cross-validate --election-date 2024-03-10

# Run nowcasting using a pre-trained model
# Note: Ensure --load-dir points to the correct path (e.g., outputs/latest)
python -m src.main --mode nowcast --load-dir outputs/latest

# Generate visualizations for a saved model
# Note: Ensure --load-dir points to the correct path
python -m src.main --mode viz --load-dir outputs/latest
```

Available modes:
- `fit`: Train a new model
- `predict`: Generate predictions using a saved model (Note: Currently implies loading, might need refinement based on `main.py` logic)
- `predict-history`: Generate historical predictions (Note: Might need refinement based on `main.py` logic)
- `retrodictive`: Run retrodictive evaluation (Note: Might need refinement based on `main.py` logic)
- `nowcast`: Estimate current support using the latest polls and a saved model
- `cross-validate`: Perform cross-validation on past elections
- `viz`: Generate visualizations for a saved model

Key parameters:
- `--election-date`: Target election date (YYYY-MM-DD)
- `--cutoff-date`: Exclude data after this date for retrodictive testing
- `--baseline-timescales`: Timescales for baseline GP in days (default: 365)
- `--election-timescales`: Timescales for election-specific GP in days (default: 60)
- `--draws`: Number of posterior samples (default: 1000)
- `--tune`: Number of tuning steps for NUTS sampler (default: 1000)
- `--load-dir`: Directory to load saved model from
- `--output-dir`: Directory to save outputs
- `--fast`: Skip plots and non-essential operations for faster execution
- `--notify`: Send notifications via ntfy.sh
- `--debug`: Enable detailed diagnostic output

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