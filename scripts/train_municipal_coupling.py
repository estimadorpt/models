"""Training entry point for the municipal coupling model (issue #38)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.municipal_coupling_model import train_coupling_model


def parse_years(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the municipal coupling model")
    parser.add_argument(
        "--trace-path",
        type=str,
        default="outputs/latest/trace.zarr",
        help="Path to the national PyMC trace.zarr directory (source of national_trend_pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/municipal_coupling",
        help="Directory where training artefacts will be stored",
    )
    parser.add_argument(
        "--election-years",
        type=parse_years,
        default=parse_years("2009,2013,2017,2021"),
        help="Comma separated list of election years to include",
    )
    parser.add_argument(
        "--train-years",
        type=parse_years,
        default=parse_years("2009,2013,2017"),
        help="Comma separated list of training election years",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Number of posterior draws")
    parser.add_argument("--tune", type=int, default=1000, help="Number of tuning steps")
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.9,
        help="Target acceptance rate for NUTS",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    _, _, evaluation = train_coupling_model(
        trace_path=args.trace_path,
        election_years=args.election_years,
        train_years=args.train_years,
        output_dir=output_dir,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        random_seed=args.seed,
    )

    print("Holdout year:", evaluation.election_year)
    print("Winner accuracy:", f"{evaluation.winner_accuracy:.3f}")
    print("Vote-share MAE:", f"{evaluation.mean_vote_share_mae:.3f}")
    print(f"Artefacts written to {output_dir}")


if __name__ == "__main__":
    main()
