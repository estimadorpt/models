"""Run calibration improvement experiments for municipal coupling model."""

import subprocess
from pathlib import Path

# Import the model to test programmatically
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.municipal_coupling_model import train_coupling_model

TRACE_PATH = "outputs/latest/trace.zarr"
YEARS = [2009, 2013, 2017, 2021]
TRAIN_YEARS = [2009, 2013, 2017]
SEED = 42

experiments = [
    {
        "name": "baseline",
        "description": "Baseline: single global concentration",
        "draws": 400,
        "tune": 400,
        "use_national_uncertainty": False,
        "use_hierarchical_concentration": False,
        "concentration_results_prior": (100.0, 0.1),
    },
    {
        "name": "national_uncertainty",
        "description": "With national uncertainty sampling",
        "draws": 400,
        "tune": 400,
        "use_national_uncertainty": True,
        "use_hierarchical_concentration": False,
        "concentration_results_prior": (100.0, 0.1),
    },
    {
        "name": "hierarchical_conc",
        "description": "Hierarchical (region-specific) concentration",
        "draws": 400,
        "tune": 400,
        "use_national_uncertainty": False,
        "use_hierarchical_concentration": True,
        "concentration_results_prior": (100.0, 0.1),
    },
    {
        "name": "looser_prior",
        "description": "Looser concentration prior (more uncertainty)",
        "draws": 400,
        "tune": 400,
        "use_national_uncertainty": False,
        "use_hierarchical_concentration": False,
        "concentration_results_prior": (50.0, 0.5),  # Looser: mean=100 but more variance
    },
    {
        "name": "high_samples",
        "description": "High sample count (1200 draws/tune)",
        "draws": 1200,
        "tune": 1200,
        "use_national_uncertainty": False,
        "use_hierarchical_concentration": False,
        "concentration_results_prior": (100.0, 0.1),
    },
]

def run_experiment(exp_config: dict):
    """Run a single experiment."""
    output_dir = Path(f"outputs/exp_{exp_config['name']}")

    print(f"\n{'='*70}")
    print(f"📊 Running: {exp_config['description']}")
    print(f"   Output: {output_dir}")
    print('='*70)

    try:
        model, idata, evaluation = train_coupling_model(
            trace_path=TRACE_PATH,
            election_years=YEARS,
            train_years=TRAIN_YEARS,
            output_dir=output_dir,
            draws=exp_config["draws"],
            tune=exp_config["tune"],
            target_accept=0.9,
            random_seed=SEED,
            use_national_uncertainty=exp_config["use_national_uncertainty"],
            use_hierarchical_concentration=exp_config["use_hierarchical_concentration"],
            concentration_results_prior=exp_config["concentration_results_prior"],
        )

        print(f"\n✅ {exp_config['name']} complete!")
        print(f"   Winner accuracy: {evaluation.winner_accuracy:.2%}")
        print(f"   MAE: {evaluation.mean_vote_share_mae:.4f}")

        return True
    except Exception as e:
        print(f"\n❌ {exp_config['name']} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compute_brier_scores():
    """Compute Brier scores for all experiments."""
    print(f"\n\n{'='*70}")
    print("🔍 Computing Brier scores for all experiments")
    print('='*70)

    exp_dirs = [f"outputs/exp_{exp['name']}" for exp in experiments]
    # Add comparison with the earlier run
    exp_dirs.append("outputs/municipal_with_new_features")

    cmd = [
        "python3",
        "scripts/compute_municipal_brier_scores.py",
    ] + exp_dirs

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

if __name__ == "__main__":
    print("""
    ========================================================================
    🔬 Municipal Coupling Model: Calibration Experiments
    ========================================================================

    Testing different approaches to improve probabilistic calibration:
    1. Baseline (single global concentration)
    2. National uncertainty sampling
    3. Hierarchical (region-specific) concentration
    4. Looser concentration prior
    5. High MCMC sample count

    """)

    results = {}
    for exp in experiments:
        success = run_experiment(exp)
        results[exp["name"]] = success

    print(f"\n\n{'='*70}")
    print("📋 Experiment Summary")
    print('='*70)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {name:25s}: {status}")

    # Compute Brier scores
    if all(results.values()):
        compute_brier_scores()
    else:
        print("\n⚠️  Some experiments failed. Skipping Brier score comparison.")
