# Bayesian Election Forecasting Skill

This skill provides specialized knowledge for Bayesian election forecasting models using PyMC. It covers both general Bayesian modeling principles and specific patterns used in this election forecasting codebase.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Architecture Overview](#architecture-overview)
3. [PyMC Modeling Patterns](#pymc-modeling-patterns)
4. [Model Components](#model-components)
5. [Diagnostics and Validation](#diagnostics-and-validation)
6. [Common Issues and Solutions](#common-issues-and-solutions)
7. [Best Practices](#best-practices)

---

## Core Concepts

### Why Bayesian Methods for Election Forecasting?

Bayesian hierarchical models excel at election forecasting because they:
- **Pool information** across pollsters, time periods, and geographic regions
- **Quantify uncertainty** properly through full posterior distributions
- **Handle sparse data** via hierarchical shrinkage
- **Incorporate prior knowledge** about polling bias and temporal dynamics
- **Enable probabilistic predictions** for election outcomes

### Key Components of Election Models

**Latent Support**: The true, unobserved level of support for each party at time t
- Polls are noisy measurements of this latent support
- Evolves smoothly over time (no sudden jumps)
- Model using Gaussian Process priors

**House Effects**: Systematic polling bias by organization
- Some pollsters consistently over/under-estimate certain parties
- Modeled as pollster-specific offsets
- Hierarchical structure pools information across pollsters

**Temporal Dynamics**: How support changes over time
- **Baseline trends**: Long-term (multi-year) structural factors
- **Election-specific trends**: Campaign dynamics (weeks to months)
- Modeled using Gaussian Processes with different timescales

**Geographic Effects**: Regional variations in party support
- District, municipality, or parish-level deviations from national trends
- Hierarchical structure with party-district interactions

---

## Architecture Overview

### Class Hierarchy

```
BaseElectionModel (abstract)
├── StaticBaselineElectionModel
├── DynamicGPElectionModel
├── PresidentialElectionModel
└── MunicipalCouplingModel
```

### Key Classes

**ElectionsFacade** (`src/models/elections_facade.py`)
- Main interface for users
- Simplifies data loading, model building, inference, and visualization
- Handles saving/loading of inference results
- Provides diagnostic plots

**BaseElectionModel** (`src/models/base_model.py`)
- Abstract base class defining common interface
- Manages dataset, coordinates, and data containers
- Provides `sample_all()` method for MCMC
- Subclasses implement `build_model()`, `_build_coords()`, `_build_data_containers()`

**ElectionDataset** (`src/data/dataset.py`)
- Manages all data loading and preprocessing
- Supports multiple geographic levels: parish, municipality, district, national
- Handles train/test splits
- Creates non-competing party masks
- Defines political families (parties) and election dates

### Data Flow

```
1. ElectionDataset loads raw data
   ↓
2. ElectionsFacade initializes with dataset and model class
   ↓
3. Model.build_model() creates PyMC structure
   ↓
4. Facade.run_inference() samples posterior
   ↓
5. Facade.save_inference_results() persists to Zarr
   ↓
6. Facade.generate_diagnostic_plots() creates visualizations
```

---

## PyMC Modeling Patterns

### Hierarchical Structure

Election models use hierarchical priors to pool information:

```python
# Party-level parameters (top level)
party_baseline ~ Normal(mu=prior_means, sigma=prior_sds)

# Pollster-party effects (mid level)
house_effects_sd ~ HalfNormal(sigma=0.05)  # Global SD
house_effects_raw ~ Normal(0, 1)  # Non-centered
house_effects = house_effects_raw * house_effects_sd  # Transform

# Poll-level observations (bottom level)
poll_share ~ DirichletMultinomial(concentration=alpha, observed=poll_counts)
```

### Gaussian Process Time Dynamics

GPs model smooth temporal evolution with configurable timescales:

```python
# Define timescale (e.g., 365 days for baseline, 14 days for campaign)
ls = pm.Constant("baseline_ls", 365.0)

# Covariance function
cov_func = pm.gp.cov.ExpQuad(input_dim=1, ls=ls)

# GP prior over time
gp = pm.gp.Latent(cov_func=cov_func)
time_effect = gp.prior("baseline_gp", X=time_points)
```

**Timescale Interpretation**:
- `ls=365`: Changes correlated over ~1 year (baseline trends)
- `ls=60`: Changes correlated over ~2 months (pre-campaign)
- `ls=14`: Changes correlated over ~2 weeks (official campaign)

### Non-Centered Parameterization

Always use non-centered parameterization for better sampling:

```python
# BAD (centered):
house_effects ~ Normal(0, house_effects_sd)

# GOOD (non-centered):
house_effects_raw ~ Normal(0, 1)
house_effects = house_effects_raw * house_effects_sd
```

### Softmax for Probabilities

Convert latent scores to probabilities that sum to 1:

```python
# latent_mu: [time, parties] - raw latent scores
# Add non-competing mask before softmax
latent_mu_masked = latent_mu + non_competing_mask  # -100 for non-competing

# Softmax to get probabilities
latent_popularity = pm.Deterministic(
    "latent_popularity",
    pm.math.softmax(latent_mu_masked, axis=-1)
)
```

### Dirichlet-Multinomial Likelihood

Models polling data with overdispersion:

```python
# concentration parameter controls variance
# Higher concentration = less variance (more precise polls)
concentration = pm.Deterministic(
    "concentration_polls",
    pm.math.exp(log_concentration_polls)
)

# Alpha for Dirichlet-Multinomial
alpha = popularity * concentration  # [polls, parties]

# Likelihood
poll_counts = pm.DirichletMultinomial(
    "poll_counts",
    n=sample_sizes,  # poll sample sizes
    a=alpha,         # concentration parameters
    observed=observed_counts
)
```

### Proper Indexing with pm.Data

Use `pm.Data` for dynamic indexing that supports prediction:

```python
# Create index arrays
pollster_idx = pm.Data("pollster_idx", pollster_indices, mutable=True)
time_idx = pm.Data("time_idx", time_indices, mutable=True)

# Use advanced indexing
house_effect_for_poll = house_effects[pollster_idx, :]  # [polls, parties]
latent_support_at_time = latent_trajectory[time_idx, :]  # [polls, parties]
```

---

## Model Components

### 1. Baseline Party Support

Long-term structural support based on historical results:

```python
# Prior from historical election results
historical_means = results.mean(axis=0)  # Mean across elections
historical_sds = results.std(axis=0)     # SD across elections

party_baseline = pm.Normal(
    "party_baseline",
    mu=historical_means,
    sigma=historical_sds * 2,  # Wider for uncertainty
    dims="parties_complete"
)
```

### 2. Gaussian Process Dynamics

**Baseline GP** (long timescale):
```python
baseline_gp = pm.gp.Latent(
    cov_func=pm.gp.cov.ExpQuad(1, ls=baseline_timescale)
)
baseline_effect = baseline_gp.prior(
    "baseline_gp_coef",
    X=calendar_dates[:, None]
)
```

**Election-specific GP** (short timescale):
```python
# Multiple timescales for campaign dynamics
for i, ls in enumerate(election_timescales):
    gp = pm.gp.Latent(cov_func=pm.gp.cov.ExpQuad(1, ls=ls))
    effect = gp.prior(f"election_gp_{i}", X=days_to_election[:, None])
    short_term_effect += effect
```

### 3. House Effects

Pollster-specific biases:

```python
# Hierarchical house effects
house_effects_sd = pm.HalfNormal("house_effects_sd", sigma=0.05)
house_effects_raw = pm.Normal(
    "house_effects_raw",
    mu=0,
    sigma=1,
    dims=["pollsters", "parties_complete"]
)
house_effects = pm.Deterministic(
    "house_effects",
    house_effects_raw * house_effects_sd
)
```

### 4. District/Geographic Effects

Regional deviations from national trends:

```python
district_effects_sd = pm.HalfNormal("district_effects_sd", sigma=0.10)
district_effects_raw = pm.Normal(
    "district_effects_raw",
    mu=0,
    sigma=1,
    dims=["districts", "parties_complete"]
)
district_effects = pm.Deterministic(
    "district_effects",
    district_effects_raw * district_effects_sd
)
```

### 5. Non-Competing Parties

Handle parties that didn't compete in certain elections:

```python
# Mask: 0 for competing, -100 for non-competing
# Applied before softmax to zero out non-competing parties
is_here_polls = polls[parties] > 0
non_competing_mask = np.where(is_here_polls, 0, -100)

# Apply in model
latent_mu_masked = latent_mu + non_competing_mask
popularity = pm.math.softmax(latent_mu_masked, axis=-1)
```

### 6. Concentration Parameters

Control poll variance (higher = less noise):

```python
# Log scale for positivity constraint
log_concentration_polls = pm.Normal(
    "log_concentration_polls",
    mu=np.log(500),  # Prior: moderate precision
    sigma=1.0
)
concentration_polls = pm.Deterministic(
    "concentration_polls",
    pm.math.exp(log_concentration_polls)
)
```

---

## Diagnostics and Validation

### MCMC Diagnostics

**R-hat (Gelman-Rubin statistic)**:
- Measures convergence across chains
- **Target: < 1.01** (this codebase standard)
- Values > 1.01 indicate poor convergence
- Check: `arviz.summary(trace)['r_hat']`

**Effective Sample Size (ESS)**:
- Number of independent samples
- **Target: > 400** (recommended minimum)
- ESS_bulk: posterior bulk, ESS_tail: tails
- Low ESS → increase draws or fix parameterization

**Divergences**:
- Indicate NUTS sampler struggles (usually posterior geometry issues)
- **Target: 0 divergences**
- Solutions:
  1. Increase `target_accept` (e.g., 0.95 → 0.99)
  2. Use non-centered parameterization
  3. Reparameterize model
  4. Check for label switching or multimodality

**Energy Plot**:
- Compares energy distribution between transitions
- Should overlap well between energy and marginal energy
- Mismatch suggests biased sampling

**Tree Depth**:
- Samples hitting max tree depth indicate complex geometry
- Increase `max_treedepth` if > 10% hit limit

### Model Validation

**Prior Predictive Checks**:
```python
with model:
    prior = pm.sample_prior_predictive()

# Check if priors produce reasonable values
# E.g., poll shares between 0-1, sum to 1
```

**Posterior Predictive Checks**:
```python
with model:
    posterior_pred = pm.sample_posterior_predictive(trace)

# Compare predicted vs observed polls
# Check calibration: are observations within credible intervals?
```

**Retrodictive Validation**:
- Train on data up to cutoff date
- Predict held-out polls
- Check if predictions are well-calibrated

**Cross-Validation**:
- Train on elections excluding target election
- Predict target election
- Assess out-of-sample accuracy

### Quality Checks in This Codebase

The `_analyze_trace_quality()` method checks:
1. Divergences count and percentage
2. Tree depth saturation
3. Key parameter R-hat and ESS
4. Concentration parameter values

The `generate_diagnostic_plots()` creates:
- Trace plots (per variable category)
- Energy plots
- Pair plots (for scalar params)
- Forest plots (house effects, district effects)
- Summary text with convergence issues

---

## Common Issues and Solutions

### Issue: Divergences

**Symptoms**:
- Warning: "X divergent transitions detected"
- Usually indicates problematic posterior geometry

**Solutions**:
1. **Increase target_accept**:
   ```python
   facade.run_inference(target_accept=0.99)  # from default 0.9
   ```

2. **Non-centered parameterization**:
   ```python
   # Instead of: x ~ Normal(mu, sigma)
   x_raw ~ Normal(0, 1)
   x = mu + x_raw * sigma
   ```

3. **Tighter priors**:
   ```python
   # If baseline has divergences, tighten prior
   party_baseline ~ Normal(mu, sigma=historical_sd)  # instead of 2*historical_sd
   ```

4. **Check for label switching**:
   - If parties are unlabeled, sampler can swap them
   - Add identifiability constraints

### Issue: Low ESS

**Symptoms**:
- ESS < 400 for key parameters
- High autocorrelation in chains

**Solutions**:
1. **Increase draws/tune**:
   ```python
   facade.run_inference(draws=3000, tune=3000)  # from default 1000/1000
   ```

2. **Check parameterization**:
   - Switch to non-centered if centered
   - May need reparameterization

3. **Multi-scale GPs**:
   - If GP has low ESS, may need different timescale
   - Or use Hilbert space approximation

### Issue: Coordinate Mismatches

**Symptoms**:
- KeyError when selecting coordinates
- Shape mismatch errors during indexing

**Solutions**:
1. **Check coordinate names**:
   ```python
   print(trace.posterior.coords)  # See what coords exist
   ```

2. **Ensure consistent naming**:
   - Use `dims` parameter in pm.Deterministic
   - Match coordinate names exactly

3. **Date normalization**:
   ```python
   # Always normalize dates when comparing
   calendar_coords = pd.to_datetime(coords).normalize()
   target_date = pd.Timestamp(date).normalize()
   ```

### Issue: Non-Competing Parties

**Symptoms**:
- New parties appear mid-dataset
- Softmax probabilities don't sum to 1 for all parties

**Solutions**:
1. **Use masks consistently**:
   ```python
   # Polls mask
   is_here = polls[parties] > 0
   mask = np.where(is_here, 0, -100)

   # Results mask
   is_competing = results > 0
   mask_results = np.where(is_competing, 0, -100)
   ```

2. **Check first appearance date**:
   ```python
   first_appearance = results[parties].gt(0).idxmax()
   is_present = calendar_date >= first_appearance
   ```

3. **Apply before softmax**:
   - Mask must be added before softmax, not after

### Issue: Memory/Performance

**Symptoms**:
- Model building is very slow
- Sampling takes excessive memory
- Diagnostic plots fail to generate

**Solutions**:
1. **Limit deterministics**:
   - Only save essential variables
   - Remove large intermediate calculations

2. **Use coords efficiently**:
   - Don't expand dimensions unnecessarily
   - Use indexing instead of broadcasting

3. **Diagnostic plot limits**:
   - Skip plots for variables with > 50 elements
   - Use `compact=True` in plot_trace

---

## Best Practices

### Model Development

1. **Start simple, add complexity**:
   - Begin with static baseline model
   - Add GP dynamics incrementally
   - Add house effects, then geographic effects

2. **Check priors first**:
   ```python
   prior = pm.sample_prior_predictive()
   # Verify priors produce reasonable ranges
   ```

3. **Use informative priors**:
   - Base party support on historical results
   - House effects centered at 0 with small SD
   - Concentration from expected poll precision

4. **Validate incrementally**:
   - After each addition, check diagnostics
   - Ensure R-hat < 1.01 before proceeding

### Parameterization

1. **Always non-centered for hierarchical models**:
   ```python
   # Template
   param_raw ~ Normal(0, 1)
   param = mean + param_raw * sd
   ```

2. **Log scale for positive parameters**:
   ```python
   log_param ~ Normal(...)
   param = pm.math.exp(log_param)
   ```

3. **Proper GP timescales**:
   - Baseline: 365 days (annual cycle)
   - Pre-campaign: 30-60 days
   - Campaign: 7-14 days

### MCMC Sampling

1. **Sufficient warmup**:
   - `tune >= 1000` for complex models
   - `tune >= 2000` if convergence issues

2. **Multiple chains**:
   - `chains=4` minimum for convergence checks
   - Use `cores=4` for parallelization

3. **Target acceptance**:
   - Default: `target_accept=0.9`
   - If divergences: increase to 0.95 or 0.99

4. **Use NumPyro sampler**:
   ```python
   pm.sample(nuts_sampler='numpyro')  # Faster than PyMC
   ```

### Saving and Loading

1. **Use Zarr format**:
   ```python
   facade.save_inference_results(directory=output_dir)
   # Saves: prior_checks.zarr, trace.zarr, posterior_checks.zarr
   ```

2. **Output naming convention**:
   ```
   ACCEPTED_{model}_{election}_{version}_{description}/

   Example: ACCEPTED_presidential_2026_v1_zerosumnormal/
   ```

3. **Mark validated runs**:
   - Prefix with `ACCEPTED_` only after:
     - R-hat < 1.01 for all parameters
     - ESS > 400 for key parameters
     - 0 divergences
     - Results validated against data

4. **Version control**:
   - Increment version (v1, v2, v3) for significant changes
   - Add description of key feature (gp, zerosumnormal, etc.)

### Code Organization

1. **Separate concerns**:
   - Data loading: `ElectionDataset`
   - Model structure: `BaseElectionModel` subclasses
   - Inference & viz: `ElectionsFacade`

2. **Use coordinates properly**:
   ```python
   coords = {
       "parties_complete": political_families,
       "calendar_time": calendar_dates,
       "pollsters": pollster_names,
       "districts": district_names
   }
   ```

3. **Consistent variable naming**:
   - `_raw`: Non-centered raw variables
   - `_sd`: Standard deviation hyperparameters
   - `_gp`: Gaussian process components
   - `latent_`: Before softmax transformation
   - `noisy_`: After adding measurement noise

### Debugging

1. **Enable debug mode**:
   ```python
   facade = ElectionsFacade(..., debug=True)
   # Prints detailed diagnostics during building
   ```

2. **Check shapes frequently**:
   ```python
   print(f"Shape: {var.eval().shape}")
   print(f"Dims: {model.coords}")
   ```

3. **Validate data alignment**:
   ```python
   # Ensure dates match
   assert len(poll_dates) == len(poll_data)

   # Check for NaNs
   assert not poll_data.isna().any()
   ```

4. **Use small test runs**:
   ```python
   # Quick test with minimal sampling
   facade.run_inference(draws=50, tune=50, chains=2)
   ```

### Documentation

1. **Comment model assumptions**:
   ```python
   # Assumption: House effects are constant over time
   # Future: Consider time-varying house effects
   ```

2. **Document coordinate dimensions**:
   ```python
   latent_mu = ...  # [calendar_time, parties_complete]
   house_effects = ...  # [pollsters, parties_complete]
   ```

3. **Explain non-obvious choices**:
   ```python
   # Use log(concentration) because:
   # 1. Ensures positivity
   # 2. Better MCMC geometry
   # 3. Symmetric prior around typical values
   ```

---

## Advanced Topics

### Multi-Level Geographic Modeling

This codebase supports parish → municipality → district → national hierarchy:

```python
dataset = ElectionDataset(
    election_date="2026-01-26",
    geographic_level="municipality",  # or "parish", "district", "national"
    ...
)
```

**Coupling geographic levels**:
- National trends inform district/municipality predictions
- District effects modeled as deviations from national
- Uses `GeographicLevelManager` for flexible aggregation

### Zero-Sum Constraints

For presidential models with zero-sum normal dynamics:

```python
# Sum-to-zero constraint for identifiability
party_time_effect_raw ~ Normal(0, 1)  # [time, parties-1]
# Last party effect = -sum(others)
party_time_effect[:, -1] = -pm.math.sum(party_time_effect_raw, axis=-1)
```

### Election-Specific Baselines

Different baseline for each election cycle:

```python
election_party_baseline = pm.Normal(
    "election_party_baseline",
    mu=party_baseline,  # Centered on overall baseline
    sigma=baseline_variation,
    dims=["elections", "parties_complete"]
)
```

### Predictors (Economic Indicators)

Incorporate economic predictors:

```python
# Load GDP, unemployment, etc.
predictors = dataset.campaign_preds  # [polls, n_predictors]

# Add to model
beta = pm.Normal("beta", mu=0, sigma=1, dims="predictors")
econ_effect = pm.math.dot(predictors, beta)  # [polls]
latent_mu += econ_effect[:, None]  # Broadcast to parties
```

---

## Quick Reference

### Key Files

- `src/models/elections_facade.py`: Main interface
- `src/models/base_model.py`: Abstract base class
- `src/data/dataset.py`: Data loading and preprocessing
- `src/processing/forecasting.py`: Seat projections, CI calculations

### Common Commands

```bash
# Train model
python -m src.main --mode train --election-date 2026-01-26 --draws 1000 --tune 1000

# Generate visualizations
python -m src.main --mode viz --load-dir outputs/latest

# Diagnostics
python -m src.main --mode diagnose --load-dir outputs/latest

# Cross-validation
python -m src.main --mode cross-validate --output-dir outputs/cv_results
```

### Typical Workflow

```python
from src.models.elections_facade import ElectionsFacade
from src.models.presidential_election_model import PresidentialElectionModel

# 1. Initialize
facade = ElectionsFacade(
    election_date="2026-01-26",
    model_class=PresidentialElectionModel,
    baseline_timescales=[365],
    election_timescales=[30, 14]
)

# 2. Build model
facade.build_model()

# 3. Run inference
facade.run_inference(draws=1000, tune=1000, target_accept=0.95)

# 4. Save results
facade.save_inference_results(directory="outputs/model_run_2025_12_09")

# 5. Generate diagnostics
facade.generate_diagnostic_plots(directory="outputs/model_run_2025_12_09/diagnostics")

# 6. Get predictions
latent_pop = facade.get_latent_popularity(date_mode='election_day')
```

### Key Diagnostic Criteria

- **R-hat**: < 1.01 for all parameters
- **ESS_bulk**: > 400 for key parameters
- **Divergences**: 0 (or < 0.1% if unavoidable)
- **Max tree depth**: < 10% of samples

### Useful ArviZ Functions

```python
import arviz as az

# Summary statistics
summary = az.summary(trace, var_names=['party_baseline', 'house_effects_sd'])

# Convergence diagnostics
az.plot_trace(trace, var_names=['party_baseline'])
az.plot_energy(trace)
az.plot_rank(trace, var_names=['party_baseline'])

# Posterior analysis
az.plot_forest(trace, var_names=['house_effects'])
az.plot_posterior(trace, var_names=['concentration_polls'])
```

---

## References

### Bayesian Election Forecasting

- **Linzer (2013)**: Dynamic Bayesian forecasting of presidential elections
- **Economist Model** (2020): Time-varying fundamentals + polling
- **538 Model**: Hierarchical polling aggregation with adjustments
- **PyMC Examples**: election-forecasting notebooks

### PyMC Documentation

- **Gaussian Processes**: https://www.pymc.io/projects/examples/en/latest/gaussian_processes/
- **Hierarchical Models**: https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
- **NUTS Diagnostics**: https://www.pymc.io/projects/docs/en/stable/api/diagnostics.html

### This Codebase

- **CLAUDE.md**: Project-specific instructions and conventions
- **README.md**: Setup and usage instructions
- **tests/**: Example usage and validation tests
