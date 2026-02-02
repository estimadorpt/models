#!/usr/bin/env python
"""
Export Presidential Dashboard Files

This script generates all JSON files needed for the estimador.pt web dashboard
from a saved presidential model trace.

Usage:
    # Export from latest presidential run
    pixi run python scripts/export_presidential_dashboard.py
    
    # Export from specific trace
    pixi run python scripts/export_presidential_dashboard.py \
        --trace-path outputs/presidential_dec_2025_v2/trace.zarr \
        --output-dir outputs/presidential_dec_2025_v2
    
    # Export and copy to estimador-web
    pixi run python scripts/export_presidential_dashboard.py --copy-to-web
    
    # Mark the run as accepted
    pixi run python scripts/export_presidential_dashboard.py --mark-accepted my_run_name

Generated files:
    - presidential_forecast.json
    - presidential_win_probabilities.json
    - presidential_trends.json
    - presidential_snapshot_probabilities.json
    - presidential_trajectories.json
    - presidential_house_effects.json
    - presidential_polls.json
    - presidential_runoff_pairs.json
    - presidential_snapshot_runoff_pairs.json
    - presidential_head_to_head.json
    - presidential_changes.json (changes since last poll)
"""

import argparse
import os
import shutil
import sys
import re
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.presidential_forecasting import export_dashboard_from_trace
from src.data.presidential_loaders import load_presidential_polls


# Default paths
DEFAULT_TRACE_PATH = "outputs/latest_presidential/trace.zarr"
DEFAULT_OUTPUT_DIR = "outputs/latest_presidential"
DEFAULT_POLLS_FILE = "presidenciais_polls_2026.parquet"
DEFAULT_ELECTION_DATE = "2026-01-18"
ESTIMADOR_WEB_DATA_PATH = "../estimador-web/public/data"


def export_dashboard(
    trace_path: str,
    output_dir: str,
    polls_file: str = DEFAULT_POLLS_FILE,
    election_date: str = DEFAULT_ELECTION_DATE,
    n_trajectory_samples: int = 100,
) -> dict:
    """
    Export all dashboard JSON files from a presidential model trace.
    
    Args:
        trace_path: Path to the saved zarr trace directory
        output_dir: Directory to save JSON files
        polls_file: Name of the polls parquet file in data/
        election_date: Election date string (e.g., '2026-01-18')
        n_trajectory_samples: Number of samples for trajectory JSON
        
    Returns:
        Dictionary mapping output type to file path
    """
    print(f"Loading polls from {polls_file}...")
    polls_df = load_presidential_polls(polls_file)
    
    print(f"\nExporting dashboard files from {trace_path}...")
    output_files = export_dashboard_from_trace(
        trace_path=trace_path,
        output_dir=output_dir,
        election_date=election_date,
        polls_df=polls_df,
        n_trajectory_samples=n_trajectory_samples,
    )
    
    return output_files


def copy_to_estimador_web(output_dir: str, web_data_path: str = ESTIMADOR_WEB_DATA_PATH):
    """
    Copy presidential JSON files to the estimador-web public/data folder.
    
    Args:
        output_dir: Source directory with presidential JSON files
        web_data_path: Destination path (relative to models repo root)
    """
    # Resolve paths
    models_root = Path(__file__).parent.parent
    source_dir = Path(output_dir)
    dest_dir = models_root / web_data_path
    
    if not dest_dir.exists():
        print(f"Warning: Destination {dest_dir} does not exist.")
        print("Make sure estimador-web is checked out at ../estimador-web")
        return False
    
    # Find all presidential JSON files
    json_files = list(source_dir.glob("presidential_*.json"))
    
    if not json_files:
        print(f"No presidential_*.json files found in {source_dir}")
        return False
    
    print(f"\nCopying {len(json_files)} files to {dest_dir}...")
    for src_file in json_files:
        dest_file = dest_dir / src_file.name
        shutil.copy2(src_file, dest_file)
        print(f"  Copied {src_file.name}")
    
    print(f"\nDone! Files copied to {dest_dir}")
    print("\nNext steps:")
    print(f"  cd {dest_dir.parent.parent}")
    print("  git add public/data/presidential_*.json")
    print('  git commit -m "feat: update presidential forecast"')
    print("  git push")
    
    return True


def mark_run_as_accepted(output_dir: str, accepted_name: str) -> str:
    """
    Mark a model run as accepted by renaming the directory with ACCEPTED_ prefix.
    
    Also updates the latest_presidential symlink to point to the new location.
    
    Args:
        output_dir: Current output directory path
        accepted_name: Name for the accepted run (without ACCEPTED_ prefix)
        
    Returns:
        New path to the accepted run directory
    """
    output_path = Path(output_dir).resolve()
    outputs_root = output_path.parent
    
    # Clean up the accepted name (remove any existing ACCEPTED_ prefix)
    clean_name = re.sub(r'^ACCEPTED_', '', accepted_name)
    new_name = f"ACCEPTED_{clean_name}"
    new_path = outputs_root / new_name
    
    if output_path == new_path:
        print(f"Run is already marked as accepted: {new_path}")
        return str(new_path)
    
    if new_path.exists():
        print(f"Warning: {new_path} already exists. Not renaming.")
        return str(output_path)
    
    # Rename the directory
    print(f"Renaming {output_path.name} -> {new_name}")
    output_path.rename(new_path)
    
    # Update the latest_presidential symlink
    symlink_path = outputs_root / "latest_presidential"
    if symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(new_path)
    print(f"Updated latest_presidential symlink -> {new_name}")
    
    # Remove any broken symlink inside the directory
    internal_symlink = new_path / "latest_presidential"
    if internal_symlink.is_symlink():
        internal_symlink.unlink()
    
    return str(new_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export presidential dashboard JSON files from a model trace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--trace-path",
        default=DEFAULT_TRACE_PATH,
        help=f"Path to the zarr trace directory (default: {DEFAULT_TRACE_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSON files (default: same as trace parent dir)",
    )
    parser.add_argument(
        "--polls-file",
        default=DEFAULT_POLLS_FILE,
        help=f"Polls parquet file in data/ (default: {DEFAULT_POLLS_FILE})",
    )
    parser.add_argument(
        "--election-date",
        default=DEFAULT_ELECTION_DATE,
        help=f"Election date (default: {DEFAULT_ELECTION_DATE})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of trajectory samples for spaghetti plot (default: 100)",
    )
    parser.add_argument(
        "--copy-to-web",
        action="store_true",
        help="Copy generated files to estimador-web/public/data/",
    )
    parser.add_argument(
        "--web-path",
        default=ESTIMADOR_WEB_DATA_PATH,
        help=f"Path to estimador-web data folder (default: {ESTIMADOR_WEB_DATA_PATH})",
    )
    parser.add_argument(
        "--mark-accepted",
        metavar="NAME",
        help="Mark the run as accepted with the given name (e.g., 'presidential_2026_v2_12polls')",
    )
    
    args = parser.parse_args()
    
    # Resolve trace path
    trace_path = args.trace_path
    if not os.path.exists(trace_path):
        # Try resolving as relative to models repo
        models_root = Path(__file__).parent.parent
        trace_path = models_root / args.trace_path
        if not trace_path.exists():
            print(f"Error: Trace not found at {args.trace_path}")
            print("Make sure to run presidential-train first, or specify --trace-path")
            sys.exit(1)
        trace_path = str(trace_path)
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(trace_path).parent)
    
    # Export dashboard files
    print("=" * 60)
    print("PRESIDENTIAL DASHBOARD EXPORT")
    print("=" * 60)
    
    output_files = export_dashboard(
        trace_path=trace_path,
        output_dir=output_dir,
        polls_file=args.polls_file,
        election_date=args.election_date,
        n_trajectory_samples=args.n_samples,
    )
    
    print("\n" + "=" * 60)
    print("GENERATED FILES")
    print("=" * 60)
    for key, path in output_files.items():
        print(f"  {key}: {path}")
    
    # Optionally mark as accepted
    if args.mark_accepted:
        print("\n" + "=" * 60)
        print("MARKING RUN AS ACCEPTED")
        print("=" * 60)
        output_dir = mark_run_as_accepted(output_dir, args.mark_accepted)
    
    # Optionally copy to estimador-web
    if args.copy_to_web:
        print("\n" + "=" * 60)
        print("COPYING TO ESTIMADOR-WEB")
        print("=" * 60)
        copy_to_estimador_web(output_dir, args.web_path)


if __name__ == "__main__":
    main()

