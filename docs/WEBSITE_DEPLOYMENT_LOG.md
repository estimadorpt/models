# Website Deployment Log

This file tracks which model runs are currently deployed to [estimador.pt](https://estimador.pt).

## Current Production Version

### Presidential Election 2026

**Model Run:** `presidential_dec_2025_v2`  
**Deployed:** 2025-12-22  
**Last Poll Date:** 2025-12-19 (Pitagórica/TVI-CNN)  
**Total Polls:** 12

Files deployed to `estimador-web/public/data/`:
- `presidential_forecast.json`
- `presidential_win_probabilities.json`
- `presidential_trends.json`
- `presidential_snapshot_probabilities.json`
- `presidential_trajectories.json`
- `presidential_house_effects.json`
- `presidential_polls.json`
- `presidential_head_to_head.json`
- `presidential_runoff_pairs.json`
- `presidential_snapshot_runoff_pairs.json` ← Computed at last poll date (Dec 19)

**Key Results (as of Dec 22, 2025):**
- André Ventura: 44.6% leading probability ("if elections today")
- Marques Mendes: 40.5% leading probability
- Gouveia e Melo: 9.0% leading probability
- António José Seguro: 5.4% leading probability

**Election Day Forecast:**
- André Ventura: 20.9% (35.9% win prob)
- Marques Mendes: 20.6% (34.4% win prob)
- Gouveia e Melo: 17.3% (15.3% win prob)
- António José Seguro: 16.2% (11.3% win prob)

---

## Previous Versions

### ACCEPTED_presidential_2026_v1_zerosumnormal (Dec 8, 2024)

**Status:** Superseded by `ACCEPTED_presidential_dec_2025_polls`  
**Last Poll Date:** 2025-12-03  
**Notes:** Did not include Dec 12-16 polls. Showed Gouveia e Melo higher.

---

## Deployment Process

1. Train model with latest polls
2. Export all dashboard JSON files
3. Copy JSON files to `estimador-web/public/data/`
4. Update this log
5. Push both repositories

### Quick Deploy Commands

```bash
# 1. Train presidential model (if not already done)
pixi run presidential-train --output-dir outputs/presidential_YYYYMMDD

# 2. Export all dashboard files from trace
pixi run presidential-export --trace-path outputs/presidential_YYYYMMDD/trace.zarr

# 3. Export AND copy to estimador-web in one command
pixi run presidential-export-web --trace-path outputs/presidential_YYYYMMDD/trace.zarr

# 4. Or manually copy files
cp outputs/presidential_YYYYMMDD/presidential_*.json ../estimador-web/public/data/

# 5. Commit and push estimador-web
cd ../estimador-web
git add public/data/presidential_*.json
git commit -m "feat: update presidential forecast with new polls"
git push
```

### Export Script Details

The export script (`scripts/export_presidential_dashboard.py`) generates all 10 required JSON files:

```bash
# See all options
pixi run python scripts/export_presidential_dashboard.py --help

# Export from specific trace
pixi run python scripts/export_presidential_dashboard.py \
    --trace-path outputs/presidential_dec_2025_v2/trace.zarr \
    --output-dir outputs/presidential_dec_2025_v2

# Export from latest and copy to web
pixi run presidential-export-web
```

Generated files:
- `presidential_forecast.json` - Election day forecast with credible intervals
- `presidential_win_probabilities.json` - Win/leading probabilities per candidate
- `presidential_trends.json` - Time series of support over campaign
- `presidential_snapshot_probabilities.json` - "If elections were today" probabilities
- `presidential_trajectories.json` - Spaghetti plot data (100 posterior samples)
- `presidential_house_effects.json` - Pollster bias estimates
- `presidential_polls.json` - Raw poll data for overlay
- `presidential_runoff_pairs.json` - Second round matchup probabilities (election day)
- `presidential_snapshot_runoff_pairs.json` - Second round matchups (as of last poll)
- `presidential_head_to_head.json` - Top 2 candidate head-to-head over time

