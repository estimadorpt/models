#!/usr/bin/env python3
"""
Extract win probabilities from municipal coupling model posterior samples.

This script demonstrates how to compute P(party X wins municipality Y) by
counting victories across posterior samples rather than just using mean shares.

This implementation follows the parliamentary model's pattern of loading from
zarr posteriors rather than separate numpy files.
"""

from pathlib import Path
from typing import Dict, List

import arviz as az
import numpy as np
import pandas as pd

# Import the model class to use its posterior slicing method
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.municipal_coupling_model import MunicipalCouplingModel
from src.data.municipal_coupling import build_municipal_coupling_dataset


def compute_win_probabilities(
    model_dir: Path,
    election_year: int,
    trace_path: str,
    election_years: List[int],
    train_years: List[int],
    min_probability: float = 0.01,
    poll_paths: List[Path] = None,
) -> pd.DataFrame:
    """
    Compute win probabilities for each party in each municipality.

    Parameters
    ----------
    model_dir : Path
        Directory containing the trained model posterior
    election_year : int
        Election year to compute win probabilities for
    trace_path : str
        Path to the parliamentary model trace
    election_years : List[int]
        All election years in the dataset
    train_years : List[int]
        Training years used in the model
    min_probability : float
        Minimum probability to include in output (filters noise)
    poll_paths : List[Path], optional
        Paths to poll data files

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: municipality_code, municipality_name, party,
        win_probability, mean_share, median_share
    """

    # Load posterior from zarr (following parliamentary model pattern)
    posterior_path = model_dir / "posterior.zarr"
    if not posterior_path.exists():
        raise FileNotFoundError(
            f"Posterior not found at {posterior_path}. "
            "Rerun the model with the latest code."
        )

    print(f"Loading posterior from {posterior_path}...")
    idata = az.from_zarr(str(posterior_path))

    # Rebuild the dataset and model to access the posterior slicing method
    print("Rebuilding dataset and model...")
    dataset = build_municipal_coupling_dataset(
        election_years,
        trace_path,
        train_years,
        poll_paths=poll_paths,
    )
    model = MunicipalCouplingModel(dataset, train_years=train_years, test_year=election_year)

    # Get posterior samples using the model's method (following parliamentary pattern)
    print(f"Extracting posterior samples for {election_year}...")
    shares_array = model.get_municipality_vote_share_posterior(idata, election_year)

    # shares_array shape: (samples, municipalities, parties)
    municipalities = model.municipality_codes
    parties = model.parties

    records = []

    for muni_idx, muni_code in enumerate(municipalities):
        muni_samples = shares_array[:, muni_idx, :]
        coalition_samples = model.aggregate_samples_to_coalitions(
            muni_samples, election_year, municipality_code=str(muni_code)
        )

        if not coalition_samples:
            continue

        coalition_names = list(coalition_samples.keys())
        coalition_matrix = np.column_stack(
            [coalition_samples[name] for name in coalition_names]
        )

        winners_per_sample = coalition_matrix.argmax(axis=1)
        unique, counts = np.unique(winners_per_sample, return_counts=True)
        total_samples = len(winners_per_sample)

        mean_shares = coalition_matrix.mean(axis=0)
        median_shares = np.median(coalition_matrix, axis=0)

        for coalition_idx, count in zip(unique, counts):
            probability = count / total_samples

            if probability >= min_probability:
                coalition_name = coalition_names[coalition_idx]
                records.append({
                    "municipality_code": str(muni_code),
                    "entity": coalition_name,
                    "win_probability": probability,
                    "mean_share": mean_shares[coalition_idx],
                    "median_share": median_shares[coalition_idx],
                    "victory_count": int(count),
                    "total_samples": total_samples,
                })

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["party"] = df["entity"]

    # Load municipality names if available
    predictions_file = model_dir / "predictions.csv"
    if predictions_file.exists():
        predictions = pd.read_csv(predictions_file)
        if "municipality_name" in predictions.columns:
            name_map = predictions.set_index("municipality_code")["municipality_name"].to_dict()
            df["municipality_name"] = df["municipality_code"].map(name_map)
            # Reorder columns
            cols = ["municipality_code", "municipality_name", "entity", "win_probability",
                   "mean_share", "median_share", "victory_count", "total_samples"]
            df = df[[c for c in cols if c in df.columns]]

    return df.sort_values(["municipality_code", "win_probability"], ascending=[True, False])


def summarize_competitive_municipalities(
    win_probs: pd.DataFrame,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Find municipalities where multiple parties have non-negligible win probability.

    Parameters
    ----------
    win_probs : pd.DataFrame
        Output from compute_win_probabilities
    threshold : float
        Minimum win probability to consider a party competitive

    Returns
    -------
    pd.DataFrame
        Municipalities with uncertainty metric
    """

    competitive = (
        win_probs[win_probs["win_probability"] >= threshold]
        .groupby("municipality_code")
        .agg({
            "municipality_name": "first",
            "entity": "count",  # Number of competitive coalitions
            "win_probability": lambda x: 1 - x.max(),  # Uncertainty (1 - p_leader)
        })
        .rename(columns={"entity": "n_competitive_entities", "win_probability": "uncertainty"})
        .reset_index()
    )

    competitive["n_competitive_parties"] = competitive["n_competitive_entities"]

    return competitive[competitive["n_competitive_entities"] > 1].sort_values(
        "uncertainty", ascending=False
    )


def compute_coalition_win_probabilities(
    model_dir: Path,
    election_year: int,
    trace_path: str,
    election_years: List[int],
    train_years: List[int],
    coalitions: Dict[str, List[str]],
    poll_paths: List[Path] = None,
) -> pd.DataFrame:
    """
    Compute win probabilities for party coalitions.

    Parameters
    ----------
    model_dir : Path
        Directory containing the trained model posterior
    election_year : int
        Election year to compute win probabilities for
    trace_path : str
        Path to the parliamentary model trace
    election_years : List[int]
        All election years in the dataset
    train_years : List[int]
        Training years used in the model
    coalitions : Dict[str, List[str]]
        Mapping of coalition name to list of party names
        Example: {"Centro-Direita": ["PSD", "CDS-PP"], "Esquerda": ["PS", "BE", "CDU"]}
    poll_paths : List[Path], optional
        Paths to poll data files

    Returns
    -------
    pd.DataFrame
        Win probabilities for each coalition in each municipality
    """

    # Load posterior from zarr (following parliamentary model pattern)
    posterior_path = model_dir / "posterior.zarr"
    if not posterior_path.exists():
        raise FileNotFoundError(
            f"Posterior not found at {posterior_path}. "
            "Rerun the model with the latest code."
        )

    print(f"Loading posterior from {posterior_path} for coalition analysis...")
    idata = az.from_zarr(str(posterior_path))

    # Rebuild the dataset and model
    print("Rebuilding dataset and model for coalition analysis...")
    dataset = build_municipal_coupling_dataset(
        election_years,
        trace_path,
        train_years,
        poll_paths=poll_paths,
    )
    model = MunicipalCouplingModel(dataset, train_years=train_years, test_year=election_year)

    # Get posterior samples using the model's method
    print(f"Extracting posterior samples for {election_year}...")
    shares_array = model.get_municipality_vote_share_posterior(idata, election_year)

    # shares_array shape: (samples, municipalities, parties)
    municipalities = model.municipality_codes
    parties = model.parties

    records = []

    for muni_idx, muni_code in enumerate(municipalities):
        muni_shares = shares_array[:, muni_idx, :]  # (samples, parties)

        # Compute coalition shares for each sample
        coalition_shares = {}
        for coalition_name, member_parties in coalitions.items():
            # Find indices of member parties
            member_indices = [parties.index(p) for p in member_parties if p in parties]
            if member_indices:
                # Sum shares across coalition members
                coalition_shares[coalition_name] = muni_shares[:, member_indices].sum(axis=1)

        if not coalition_shares:
            continue

        # Stack into array: (samples, n_coalitions)
        coalition_names = list(coalition_shares.keys())
        coalition_array = np.stack([coalition_shares[name] for name in coalition_names], axis=1)

        # Find winning coalition per sample
        winners = coalition_array.argmax(axis=1)
        unique, counts = np.unique(winners, return_counts=True)
        total_samples = len(winners)

        for coalition_idx, count in zip(unique, counts):
            coalition_name = coalition_names[coalition_idx]
            probability = count / total_samples
            mean_share = coalition_array[:, coalition_idx].mean()

            records.append({
                "municipality_code": str(muni_code),
                "coalition": coalition_name,
                "win_probability": probability,
                "mean_share": mean_share,
            })

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["municipality_code", "win_probability"], ascending=[True, False])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute municipal win probabilities")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/municipal_coupling"),
        help="Directory containing posterior.zarr",
    )
    parser.add_argument(
        "--election-year",
        type=int,
        required=True,
        help="Election year to compute win probabilities for (e.g., 2021)",
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        required=True,
        help="Path to parliamentary model trace (e.g., outputs/latest/trace.zarr)",
    )
    parser.add_argument(
        "--election-years",
        type=str,
        required=True,
        help="Comma-separated list of all election years (e.g., 2009,2013,2017,2021)",
    )
    parser.add_argument(
        "--train-years",
        type=str,
        required=True,
        help="Comma-separated list of training years (e.g., 2009,2013,2017)",
    )
    parser.add_argument(
        "--poll-files",
        type=str,
        nargs="*",
        help="Paths to poll parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: model_dir/win_probabilities.csv)",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.01,
        help="Minimum win probability to include",
    )
    parser.add_argument(
        "--coalitions",
        action="store_true",
        help="Also compute coalition win probabilities",
    )

    args = parser.parse_args()

    # Parse election years and train years
    election_years = [int(y.strip()) for y in args.election_years.split(",")]
    train_years = [int(y.strip()) for y in args.train_years.split(",")]
    poll_paths = [Path(p) for p in args.poll_files] if args.poll_files else None

    print(f"Computing win probabilities from {args.model_dir} for {args.election_year}...")
    print(f"Election years: {election_years}")
    print(f"Train years: {train_years}")

    # Compute party win probabilities
    win_probs = compute_win_probabilities(
        args.model_dir,
        args.election_year,
        args.trace_path,
        election_years,
        train_years,
        args.min_probability,
        poll_paths,
    )

    output_path = args.output or args.model_dir / "win_probabilities.csv"
    win_probs.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Show summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total municipalities analyzed: {win_probs['municipality_code'].nunique()}")
    print(f"Total party-municipality combinations: {len(win_probs)}")

    # Show most competitive municipalities
    competitive = summarize_competitive_municipalities(win_probs)
    if not competitive.empty:
        print("\n=== Most Competitive Municipalities ===")
        print(competitive.head(10).to_string(index=False))

    # Show municipalities with clear winners (>90% probability)
    clear_winners = win_probs[win_probs["win_probability"] > 0.90]
    if not clear_winners.empty:
        print(f"\n=== Clear Winners (>90% probability): {len(clear_winners)} municipalities ===")
        print(clear_winners.head(10).to_string(index=False))

    # Compute coalition probabilities if requested
    if args.coalitions:
        print("\n=== Computing Coalition Win Probabilities ===")
        coalitions = {
            "Centro-Direita": ["PSD", "CDS-PP"],
            "Esquerda": ["PS", "BE", "CDU"],
            "Extremos": ["CH", "IL"],
        }
        coalition_probs = compute_coalition_win_probabilities(
            args.model_dir,
            args.election_year,
            args.trace_path,
            election_years,
            train_years,
            coalitions,
            poll_paths,
        )
        coalition_output = args.model_dir / "coalition_win_probabilities.csv"
        coalition_probs.to_csv(coalition_output, index=False)
        print(f"Saved coalition probabilities to {coalition_output}")
