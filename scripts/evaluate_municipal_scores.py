#!/usr/bin/env python3
"""Compute Brier and log scores for municipal coupling runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import arviz as az

from src.data.municipal_coupling import build_municipal_coupling_dataset
from src.models.municipal_coupling_model import MunicipalCouplingModel


EPS = 1e-12


def _aggregate_observed(
    model: MunicipalCouplingModel,
    shares: np.ndarray,
    coalition_mapping: Dict[str, List[str]],
) -> Dict[str, float]:
    return model._aggregate_vector_to_coalitions(shares, coalition_mapping)


def compute_scores(
    model_dir: Path,
    election_year: int,
    trace_path: str,
    election_years: Iterable[int],
    train_years: Iterable[int],
    poll_paths: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    dataset = build_municipal_coupling_dataset(
        election_years,
        trace_path,
        train_years,
        poll_paths=poll_paths,
    )
    model = MunicipalCouplingModel(dataset, train_years=train_years, test_year=election_year)

    posterior_path = model_dir / "posterior.zarr"
    if not posterior_path.exists():
        raise FileNotFoundError(f"posterior.zarr not found in {model_dir}")

    idata = az.from_zarr(str(posterior_path))
    samples = model.predict_vote_shares(idata, election_year)

    municipalities = model.municipality_codes
    records: List[Dict[str, object]] = []

    holdout = dataset.results[dataset.results["election_year"] == election_year]
    holdout = holdout.set_index("municipality_code").loc[municipalities]

    for idx, muni_code in enumerate(municipalities):
        coalition_map = model._get_coalition_mapping(election_year, muni_code)
        predicted_samples = model.aggregate_samples_to_coalitions(
            samples[:, idx, :], election_year, muni_code
        )
        if not predicted_samples:
            continue
        predicted = {
            name: float(values.mean()) for name, values in predicted_samples.items()
        }

        share_cols = [f"share_{party}" for party in model.parties]
        observed_vector = holdout.loc[muni_code][share_cols]
        observed_vector = observed_vector.replace({np.nan: 0.0}).to_numpy(dtype=float)
        observed = _aggregate_observed(model, observed_vector, coalition_map)

        # Identify the observed winning coalition
        if observed:
            observed_winner = max(observed, key=observed.get)
        else:
            observed_winner = max(predicted, key=predicted.get)

        prob_actual = float(predicted.get(observed_winner, EPS))
        brier = (prob_actual - 1.0) ** 2
        log_score = -np.log(prob_actual + EPS)

        top_pred = max(predicted, key=predicted.get)
        records.append(
            {
                "municipality_code": muni_code,
                "municipality_name": model.municipality_names[idx],
                "observed_winner": observed_winner,
                "predicted_leader": top_pred,
                "prob_observed_winner": prob_actual,
                "prob_predicted_leader": float(predicted[top_pred]),
                "brier_score": brier,
                "log_score": log_score,
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate municipal coupling run with scoring rules.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--election-year", type=int, required=True)
    parser.add_argument("--trace-path", type=str, required=True)
    parser.add_argument("--election-years", type=str, required=True, help="Comma-separated list")
    parser.add_argument("--train-years", type=str, required=True, help="Comma-separated list")
    parser.add_argument("--poll-files", type=str, nargs="*")
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    election_years = [int(x.strip()) for x in args.election_years.split(",") if x.strip()]
    train_years = [int(x.strip()) for x in args.train_years.split(",") if x.strip()]

    df = compute_scores(
        model_dir=args.model_dir,
        election_year=args.election_year,
        trace_path=args.trace_path,
        election_years=election_years,
        train_years=train_years,
        poll_paths=args.poll_files,
    )

    if df.empty:
        print("No records produced; check inputs")
        return

    overall_brier = df["brier_score"].mean()
    overall_log = df["log_score"].mean()

    print(f"Brier score (mean): {overall_brier:.4f}")
    print(f"Log score (mean): {overall_log:.4f}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Details written to {args.output}")


if __name__ == "__main__":
    main()
