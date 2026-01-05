# Website Deployment Log

This file tracks which model runs are currently deployed to [estimador.pt](https://estimador.pt).

## Current Production Version

### Presidential Election 2026

**Model Run:** `ACCEPTED_presidential_jan2026_poll`  
**Deployed:** 2026-01-05  
**Last Poll Date:** 2026-01-04 (Pitagórica for TVI/CNN/TSF/JN)  
**Total Polls:** 13

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
- `presidential_snapshot_runoff_pairs.json` ← Computed at last poll date (Jan 4)
- `presidential_changes.json`

**Key Results (as of Jan 4, 2026):**
- André Ventura: 45.3% leading probability ("if elections today")
- Marques Mendes: 20.7% leading probability
- Gouveia e Melo: 18.4% leading probability
- António José Seguro: 13.5% leading probability

**Election Day Forecast:**
- André Ventura: 20.4% (38.2% win prob)
- Marques Mendes: 18.4% (21.2% win prob)
- Gouveia e Melo: 18.0% (20.1% win prob)
- António José Seguro: 17.2% (16.3% win prob)

**New poll added (Jan 2-4, 2026 - Pitagórica):**
- António José Seguro: 19.3%
- Gouveia e Melo: 19.2%
- André Ventura: 18.9%
- Cotrim Figueiredo: 18.0%
- Marques Mendes: 15.4%

---

## Previous Versions

### presidential_dec_2025_v2 (Dec 22, 2025)

**Status:** Superseded by `ACCEPTED_presidential_jan2026_poll`  
**Last Poll Date:** 2025-12-19  
**Notes:** 12 polls. Ventura 44.6%, Mendes 40.5% leading.

### ACCEPTED_presidential_2026_v1_zerosumnormal (Dec 8, 2024)

**Status:** Superseded  
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

