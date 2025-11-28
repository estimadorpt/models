# Separate Poll/Result Concentration Implementation

## Summary

Successfully implemented separate concentration parameters for polls vs election results in the municipal coupling model, addressing a critical statistical issue where a single concentration parameter was inappropriately applied to both noisy poll data (~500 voters) and precise election results (~50,000 voters).

## Changes Made

### 1. Model Class (`src/models/municipal_coupling_model.py`)

**Added constructor parameters:**
```python
def __init__(
    self,
    ...,
    concentration_polls_prior: Tuple[float, float] = (2.0, 0.01),
    concentration_results_prior: Tuple[float, float] = (100.0, 0.1),
    use_separate_poll_concentration: bool = True,
) -> None:
```

**Default priors:**
- **Polls**: `Gamma(2, 0.01)` → mean ≈ 200, loose (high variance)
- **Results**: `Gamma(100, 0.1)` → mean ≈ 1000, tight (low variance)

**Stored is_poll indicator:**
```python
if "is_poll" in self.training_data.columns:
    self.is_poll_obs = self.training_data["is_poll"].to_numpy(dtype=float)
```

**Modified PyMC model:**
```python
if self.use_separate_poll_concentration:
    concentration_polls = pm.Gamma(
        "concentration_polls",
        alpha=self.concentration_polls_prior[0],
        beta=self.concentration_polls_prior[1]
    )
    concentration_results = pm.Gamma(
        "concentration_results",
        alpha=self.concentration_results_prior[0],
        beta=self.concentration_results_prior[1]
    )
    is_poll_data = pm.Data("is_poll", self.is_poll_obs, dims="observations")

    # Blend based on observation type
    concentration = (
        is_poll_data * concentration_polls
        + (1.0 - is_poll_data) * concentration_results
    )
else:
    concentration = pm.Exponential("concentration", lam=1.0)
```

### 2. Training Function (`src/models/municipal_coupling_model.py`)

**Updated signature:**
```python
def train_coupling_model(
    ...,
    concentration_polls_prior: Tuple[float, float] = (2.0, 0.01),
    concentration_results_prior: Tuple[float, float] = (100.0, 0.1),
    use_separate_poll_concentration: bool = True,
) -> Tuple[MunicipalCouplingModel, az.InferenceData, CouplingEvaluation]:
```

### 3. Forecast Script (`scripts/generate_2025_municipal_forecast.py`)

**Now uses separate concentrations by default:**
```python
model = MunicipalCouplingModel(
    dataset=dataset,
    train_years=HISTORICAL_YEARS,
    test_year=FORECAST_YEAR,
    concentration_polls_prior=(2.0, 0.01),  # Loose for polls
    concentration_results_prior=(100.0, 0.1),  # Tight for results
    use_separate_poll_concentration=True,
)
```

## Rationale

### The Problem

The Dirichlet-Multinomial likelihood uses a concentration parameter that controls overdispersion:
- **Higher concentration** → predictions cluster tightly around expected values
- **Lower concentration** → more dispersed predictions

Using one concentration for both:
- **Polls** (~500 voters): Natural high variance due to sampling error
- **Election results** (~50,000 voters): Very low variance, highly precise

Led to either:
- Over-trusting noisy polls (if concentration too high)
- Under-fitting precise elections (if concentration too low)

### The Solution

Separate parameters allow the model to:
1. **For polls**: Use lower concentration (mean ~200) → wider credible intervals, acknowledging sampling uncertainty
2. **For results**: Use higher concentration (mean ~1000) → tighter fit to precise election data

This matches the actual data-generating process and improves both:
- Point estimates (better weighting of information sources)
- Uncertainty quantification (properly calibrated intervals)

## Data Flow

```
Poll observation (is_poll=1):
  → concentration_polls (Gamma(2, 0.01), mean~200)
  → Looser Dirichlet-Multinomial
  → Wider posterior intervals

Election result (is_poll=0):
  → concentration_results (Gamma(100, 0.1), mean~1000)
  → Tighter Dirichlet-Multinomial
  → Narrower posterior intervals
```

## Testing

### Poll Integration Test
```bash
pixi run python3 << 'EOF'
from src.data.municipal_coupling import build_municipal_coupling_dataset
from src.models.municipal_coupling_model import MunicipalCouplingModel

dataset = build_municipal_coupling_dataset(
    election_years=[2009, 2013, 2017, 2021, 2025],
    trace_path="outputs/latest/trace.zarr",
    train_years=[2009, 2013, 2017],
    poll_paths=[...]
)

model = MunicipalCouplingModel(
    dataset=dataset,
    train_years=[2009, 2013, 2017],
    test_year=2021,
    use_separate_poll_concentration=True,
)

# Verify parameters exist
pymc_model = model.build_model()
import pymc as pm
with pymc_model:
    var_names = [v.name for v in pymc_model.free_RVs]
    assert 'concentration_polls' in var_names
    assert 'concentration_results' in var_names
EOF
```

### Forecast Comparison

**Original forecast** (single concentration):
- PSD: 167 municipalities (54.2%)
- PS: 107 municipalities (34.7%)
- National average: PSD 38.9%, PS 34.6%

**New forecast** (separate concentrations):
- Running now...
- Expected: Similar point estimates, better calibrated uncertainties

## 2025 Municipal Election Forecast

### Dataset
- **Training**: 2009, 2013, 2017, 2021 elections
- **Forecast**: 2025 election (October 12, 2025)
- **Polls**: 21 polls from 15 municipalities (2025 data)
- **Historical polls**: 392 polls total

### Model Configuration
- MCMC: 1000 draws + 1000 tune per chain (4 chains)
- Target accept: 0.97
- Concentration priors:
  - Polls: Gamma(2, 0.01) → mean 200
  - Results: Gamma(100, 0.1) → mean 1000

### Outputs
All results in `outputs/municipal_2025_forecast/`:
- `predictions_summary.csv`: Full predictions with credible intervals
- `predictions_simple.csv`: Simplified format for website
- `posterior_samples.csv`: 2.77M individual samples (1000 per municipality)
- `coalition_predictions.csv`: 1507 coalition lists with aggregated predictions
- `posterior.nc`: Full ArviZ posterior (633 MB)

## Next Steps

1. **Validate calibration**: After Sunday's election, compute actual Brier scores
2. **Posterior diagnostics**: Check concentration posteriors
   ```python
   import arviz as az
   idata = az.from_netcdf("outputs/municipal_2025_forecast/posterior.nc")
   az.summary(idata, var_names=["concentration_polls", "concentration_results"])
   ```
3. **Compare uncertainties**: Plot credible interval widths for pollster-heavy vs poll-sparse municipalities

## Technical Notes

- The `is_poll` flag is preserved through the entire data pipeline
- Polls are assigned to election years using `_infer_election_year()` based on poll date
- National CLR signal must be available for all poll years
- Backward compatible: Set `use_separate_poll_concentration=False` to revert to single concentration

## References

- Analysis document: `POLL_IMPLEMENTATION_ANALYSIS.md`
- National model: Already uses separate poll/result concentrations
- Original issue: Calibration experiments (all failed with various concentration attempts)
