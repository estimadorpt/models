# Website Deployment Log

This file tracks which model runs are currently deployed to [estimador.pt](https://estimador.pt).

## Current Production Version

### Presidential Election 2026

**Model Run:** `ACCEPTED_presidential_dec_2025_polls`  
**Deployed:** 2024-12-20  
**Last Poll Date:** 2025-12-16 (Intercampus)  

Files deployed to `estimador-web/public/data/`:
- `presidential_forecast.json`
- `presidential_win_probabilities.json`
- `presidential_trends.json`
- `presidential_snapshot_probabilities.json`
- `presidential_trajectories.json`
- `presidential_house_effects.json`
- `presidential_polls.json`
- `presidential_head_to_head.json`
- `presidential_snapshot_runoff_pairs.json` ← Computed at last poll date (Dec 16)

**Key Results (as of Dec 16, 2025):**
- André Ventura: 46.5% leading probability
- Marques Mendes: 40.4% leading probability  
- Gouveia e Melo: 11.4% leading probability
- Top runoff: André Ventura vs Marques Mendes (58.2%)

---

## Previous Versions

### ACCEPTED_presidential_2026_v1_zerosumnormal (Dec 8, 2024)

**Status:** Superseded by `ACCEPTED_presidential_dec_2025_polls`  
**Last Poll Date:** 2025-12-03  
**Notes:** Did not include Dec 12-16 polls. Showed Gouveia e Melo higher.

---

## Deployment Process

1. Train model with latest polls
2. Mark output folder with `ACCEPTED_` prefix
3. Copy JSON files to `estimador-web/public/data/`
4. Generate `presidential_snapshot_runoff_pairs.json` from trace at last poll date
5. Update this log
6. Push both repositories

### Generating Snapshot Runoff Pairs

```python
from src.processing.presidential_forecasting import build_runoff_pairs_json, save_json
import arviz as az
import pandas as pd
import numpy as np

# Load trace
idata = az.from_zarr('outputs/ACCEPTED_xxx/trace.zarr')
candidates = list(idata.posterior.coords['candidates'].values)
calendar_time = pd.to_datetime(idata.posterior.coords['calendar_time'].values)
probs = idata.posterior['national_probs_calendar']

# Find last poll date index
last_poll_date = '2025-12-16'  # Update this
cutoff_idx = next(i for i, t in enumerate(calendar_time) if t.strftime('%Y-%m-%d') >= last_poll_date)

# Compute and save
flat = probs.isel(calendar_time=cutoff_idx).values.reshape(-1, len(candidates))
snapshot_runoff_data = build_runoff_pairs_json(flat, candidates, '2026-01-18', snapshot_date=last_poll_date)
save_json(snapshot_runoff_data, 'outputs/ACCEPTED_xxx', 'snapshot_runoff_pairs.json', 'presidential_')
```

