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

You can also run the model from the command line:

```bash
# Run with default settings
python src/main.py

# Run with custom settings
python src/main.py --election-date 2024-03-10 --draws 500 --tune 200 --notify

# Load previously saved results
python src/main.py --load-results --notify
```

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