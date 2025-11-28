# Municipal Poll Implementation Analysis

## Overview

I've reviewed the implementation of poll integration into the municipal coupling model. Here's a comprehensive analysis of how it works and potential issues.

## Implementation Details

### 1. Poll Loading (`src/data/municipal_polls.py`)

**How it works:**
- Loads ERC poll data from parquet files filtered by `geographic_scope == "municipal"`
- Maps poll responses to TARGET_PARTIES using a priority system:
  1. **Coalition aliases** (most reliable): Municipality-specific mappings from `municipal_coalitions_{year}.parquet`
  2. **Coalition members field**: Explicitly listed coalition components
  3. **Label parsing**: Fallback tokenization of party names
- Uses `COALITION_SPLIT_WEIGHTS` for proper disaggregation of multi-party lists
- Converts percentages to vote counts based on effective sample size
- Sets `is_poll=1` flag to distinguish from election results

**Coalition mapping example from test:**
- Aveiro poll: PS=26%, PSD=19%, CDS=3%, CH=14%, OTHER=38%
- 15 municipalities with 21 total poll observations for 2025
- Response rate: 80% (482 effective sample from 603 total)

✅ **Working well**: Coalition allocation is sophisticated and properly handles:
- Manual splits (e.g., PSD/CDS-PP → 80%/20%)
- Municipality-specific aliases
- Effective sample size calculation accounting for non-response

### 2. Dataset Integration (`src/data/municipal_coupling.py`)

**How it works:**
- Combines election results (`is_poll=0`) and polls (`is_poll=1`) into single DataFrame
- Election results: Full vote counts, incumbency flags, new party indicators
- Poll observations: Vote counts, availability flags, but zero for:
  - `new_LOCAL_INC`, `new_OTHER` (structural features of actual elections)
  - Incumbent flags (can't infer from poll responses)
- Baseline CLR computed from election results only (excluding polls) ✅
- Availability columns use `max()` across polls and results for same municipality/year

✅ **Working correctly**: Polls are properly integrated without contaminating baseline estimates

### 3. PyMC Model (`src/models/municipal_coupling_model.py`)

**How it handles polls:**

```python
# __init__:
poll_years = dataset.results.loc[dataset.results.get("is_poll", 0) == 1, "election_year"]
self.observation_years = sorted(set(self.train_years) | set(poll_years))
self.election_to_idx = {year: idx for idx, year in enumerate(self.observation_years)}

# _prepare_training_frame:
base_mask = dataset.results["election_year"].isin(self.train_years)
poll_mask = dataset.results.get("is_poll", 0) == 1
train_results = dataset.results[base_mask | poll_mask]
```

**National signal matching:**
- `observation_years` includes both election years AND poll years (e.g., [2009, 2013, 2017, 2021, 2025])
- `national_matrix` is indexed by `observation_years`
- `load_national_signal_clr()` correctly returns values for all years including 2025

✅ **Working correctly**: Polls get the right national signal for their year

### 4. Likelihood in PyMC Model

```python
concentration = pm.Exponential("concentration", lam=1.0)

effective_concentration = concentration / (
    1.0 + new_local_concentration * new_local_data
    + new_other_concentration * new_other_data
)

vote_concentration = pm.Deterministic(
    "dirichlet_alpha",
    pm.math.clip(vote_shares * effective_concentration[:, None], 1e-6, np.inf)
)

# Dirichlet-Multinomial likelihood
alpha_sum = pm.math.sum(vote_concentration, axis=1)
logp_terms = (
    pt.gammaln(total_votes_data + 1.0)
    + pt.gammaln(alpha_sum)
    - pt.gammaln(total_votes_data + alpha_sum)
    + pm.math.sum(pt.gammaln(counts_data + vote_concentration), axis=1)
    - pm.math.sum(pt.gammaln(counts_data + 1.0), axis=1)
    - pm.math.sum(pt.gammaln(vote_concentration), axis=1)
)
```

## Critical Issue Identified

### ⚠️ **Single Concentration Parameter for Both Polls and Elections**

**Problem:**
- The model uses ONE `concentration` parameter for ALL observations (polls + elections)
- This is inappropriate because:
  - **Election results** (~10,000-100,000 voters): Very precise, low overdispersion
  - **Poll samples** (~400-800 respondents): Much noisier, high overdispersion

**Expected behavior:**
- Elections should have HIGH concentration (tight around expected shares)
- Polls should have LOWER concentration (more dispersed due to sampling error)

**Current behavior:**
- Model tries to find a single compromise concentration
- This either:
  - Over-fits polls (concentration too high → polls treated as too informative)
  - Under-fits elections (concentration too low → ignores precise election data)

**Evidence:**
- In the national model, there ARE separate concentration parameters:
  - `concentration_polls`: For poll observations
  - `concentration_results`: For election results
- This distinction was attempted in `run_calibration_experiments.py` but reverted due to implementation bugs

### Recommendation

**Option 1: Add poll indicator and separate concentration (recommended)**

```python
# In build_model():
is_poll_data = pm.Data(
    "is_poll",
    self.training_data["is_poll"].to_numpy(dtype=float),
    dims="observations"
)

concentration_polls = pm.Exponential("concentration_polls", lam=0.01)  # Lower mean → looser
concentration_results = pm.Exponential("concentration_results", lam=1.0)  # Higher mean → tighter

concentration_obs = (
    is_poll_data * concentration_polls
    + (1.0 - is_poll_data) * concentration_results
)

effective_concentration = concentration_obs / (
    1.0 + new_local_concentration * new_local_data
    + new_other_concentration * new_other_data
)
```

**Option 2: Use sample size to adjust concentration automatically**

```python
# Compute concentration proportional to sqrt(sample size)
# Larger samples → tighter concentration
base_concentration = pm.Exponential("base_concentration", lam=1.0)
concentration_obs = base_concentration * pm.math.sqrt(total_votes_data / 1000.0)
```

**Option 3: Simply increase the Exponential rate for polls in preprocessing**

During data loading, artificially inflate poll sample sizes by a factor (e.g., 0.1x) to naturally give them lower effective concentration in the likelihood. This is a hack but avoids model changes.

## Other Observations

✅ **Working well:**
1. Coalition mapping sophisticated and municipality-aware
2. Polls don't contaminate historical baselines
3. National signal correctly indexed for poll years
4. Availability flags properly set
5. Structural features (incumbency, new parties) correctly zeroed for polls

⚠️ **Needs attention:**
1. **Single concentration parameter** - Major issue affecting model fit
2. No diagnostics comparing poll fit vs election fit separately

## Testing Recommendations

1. **Run the forecast script** and examine:
   - Are poll observations unduly influential on predictions?
   - Check `dirichlet_alpha` values for poll vs election observations

2. **Separate Brier scores** for:
   - In-sample election fit
   - Out-of-sample poll predictions (if historical polls available)

3. **Posterior predictive checks**:
   - Generate synthetic data from posterior
   - Compare distribution of poll observations vs election observations
   - Should show polls have wider variance

## Conclusion

The poll integration infrastructure is **well-designed** with proper coalition handling and data separation. However, the **single concentration parameter** is a significant statistical issue that likely causes:
- Over-reliance on noisy poll data, OR
- Under-fitting of precise election results

For the 2025 forecast, I recommend either:
1. Implementing separate poll/result concentration (proper fix)
2. Running WITHOUT polls initially to establish a baseline
3. Using polls only for municipalities without recent election history

The current implementation will work but may produce miscalibrated uncertainties.
