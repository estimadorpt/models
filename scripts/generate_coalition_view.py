"""Generate proper coalition-level predictions for 2025 municipal elections."""

from pathlib import Path
import pandas as pd
import numpy as np

from src.data.municipal_common import canonicalize_party_tokens, TOKEN_TO_PARTY

TARGET_PARTIES = ("PS", "PSD", "CDU", "CDS-PP", "BE", "CH", "IL", "OTHER", "LOCAL_INC")

def map_coalition_to_parties(coalition_name: str, components_str: str) -> set:
    """Map a coalition list to TARGET_PARTIES."""
    tokens = set()

    # Try explicit components first
    if components_str and str(components_str) != 'nan':
        raw_components = str(components_str).split(';')
        for comp in raw_components:
            comp = comp.strip()
            if comp:
                comp_tokens = canonicalize_party_tokens(comp)
                tokens.update(comp_tokens)

    # Fall back to coalition name
    if not tokens:
        tokens = set(canonicalize_party_tokens(coalition_name))

    # Map to TARGET_PARTIES
    mapped = set()
    for token in tokens:
        party = TOKEN_TO_PARTY.get(token, token)
        if party in TARGET_PARTIES:
            mapped.add(party)

    # If still nothing, it's OTHER
    if not mapped:
        mapped.add("OTHER")

    return mapped


def generate_coalition_predictions(
    party_predictions_path: Path,
    coalition_definitions_path: Path,
    output_path: Path,
):
    """Generate coalition-level predictions."""

    # Load party-level predictions
    party_preds = pd.read_csv(party_predictions_path)

    # Load coalition definitions
    coalitions = pd.read_parquet(coalition_definitions_path)

    # For each municipality, aggregate parties into coalitions
    results = []

    for municipality_code in party_preds['municipality_code'].unique():
        muni_preds = party_preds[party_preds['municipality_code'] == municipality_code].iloc[0]
        muni_coalitions = coalitions[coalitions['municipality_code'] == municipality_code]

        if muni_coalitions.empty:
            # No coalition data - skip or use party-level?
            continue

        municipality_name = muni_preds['municipality_name']
        district = muni_preds.get('district_name', '')

        # Build coalition predictions
        coalition_shares = {}
        for _, coal_row in muni_coalitions.iterrows():
            list_name = coal_row['party']
            components = coal_row.get('coalition_parties', '')

            # Map to TARGET_PARTIES
            parties_in_coalition = map_coalition_to_parties(list_name, components)

            # Sum party shares
            total_share = 0.0
            for party in parties_in_coalition:
                if party in TARGET_PARTIES:
                    total_share += muni_preds[party]

            coalition_shares[list_name] = total_share

        # Sort by share descending
        sorted_coalitions = sorted(coalition_shares.items(), key=lambda x: x[1], reverse=True)

        if sorted_coalitions:
            winner = sorted_coalitions[0][0]
            winner_share = sorted_coalitions[0][1]

            # Add to results
            for list_name, share in sorted_coalitions:
                results.append({
                    'municipality_code': municipality_code,
                    'municipality_name': municipality_name,
                    'district_name': district,
                    'coalition_list': list_name,
                    'predicted_share': share,
                    'is_predicted_winner': (list_name == winner),
                })

    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['municipality_code', 'predicted_share'], ascending=[True, False])
    results_df.to_csv(output_path, index=False)

    # Also create a winners-only file
    winners_df = results_df[results_df['is_predicted_winner']].copy()
    winners_only_path = output_path.parent / "coalition_winners.csv"
    winners_df.to_csv(winners_only_path, index=False)

    print(f"✓ Coalition predictions: {len(results_df)} entries")
    print(f"✓ Coalition winners: {len(winners_df)} municipalities")

    # Summary statistics
    winner_counts = winners_df['coalition_list'].value_counts()
    print(f"\n🏆 Top coalition winners:")
    for coalition, count in winner_counts.head(10).items():
        print(f"   {coalition:40s}: {count:3d} municipalities")

    return results_df, winners_df


if __name__ == "__main__":
    PARTY_PREDS = Path("outputs/municipal_2025_forecast/predictions_simple.csv")
    COALITION_DEFS = Path("data/municipal_coalitions_2025.parquet")
    OUTPUT = Path("outputs/municipal_2025_forecast/coalition_predictions_corrected.csv")

    print("Generating proper coalition-level predictions...")
    print(f"  Party predictions: {PARTY_PREDS}")
    print(f"  Coalition definitions: {COALITION_DEFS}")
    print(f"  Output: {OUTPUT}")
    print()

    results_df, winners_df = generate_coalition_predictions(PARTY_PREDS, COALITION_DEFS, OUTPUT)

    print(f"\n✅ Done! Files created:")
    print(f"   - {OUTPUT}")
    print(f"   - {OUTPUT.parent / 'coalition_winners.csv'}")
