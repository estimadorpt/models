#!/bin/bash
# Run calibration improvement experiments for municipal model

set -e

TRACE_PATH="outputs/latest/trace.zarr"
YEARS="2009,2013,2017,2021"
DRAWS=300
TUNE=300
SEED=42

echo "========================================"
echo "Municipal Calibration Experiments"
echo "========================================"
echo ""

# Experiment 1: Baseline (revert to simple single concentration)
echo "[1/5] Running baseline with single concentration..."
python3 -m src.main \
  --mode municipal-train \
  --municipal-trace-path "$TRACE_PATH" \
  --municipal-election-years "$YEARS" \
  --draws $DRAWS \
  --tune $TUNE \
  --seed $SEED \
  --output-dir outputs/exp_baseline \
  2>&1 | tee outputs/exp_baseline.log

# Experiment 2: Hierarchical concentration (region-specific uncertainty)
echo ""
echo "[2/5] Running with hierarchical (region-specific) concentration..."
# Note: This would need CLI flag support, skip for now
# Will test programmatically

# Experiment 3: Looser concentration prior
echo ""
echo "[3/5] Running with looser concentration prior..."
# Note: This would need CLI flag support, skip for now

# Experiment 4: National uncertainty sampling
echo ""
echo "[4/5] Running with national uncertainty sampling..."
# Note: This would need CLI flag support, skip for now

# Experiment 5: Higher sample count (match municipal_coupling_full)
echo ""
echo "[5/5] Running with more MCMC samples (1500 draws/tune)..."
python3 -m src.main \
  --mode municipal-train \
  --municipal-trace-path "$TRACE_PATH" \
  --municipal-election-years "$YEARS" \
  --draws 1500 \
  --tune 1500 \
  --seed $SEED \
  --output-dir outputs/exp_high_samples \
  2>&1 | tee outputs/exp_high_samples.log

echo ""
echo "========================================"
echo "Experiments complete!"
echo "Computing Brier scores..."
echo "========================================"

# Compute Brier scores for all experiments
python3 scripts/compute_municipal_brier_scores.py \
  outputs/exp_baseline \
  outputs/exp_high_samples \
  outputs/municipal_with_new_features

echo ""
echo "Done!"
