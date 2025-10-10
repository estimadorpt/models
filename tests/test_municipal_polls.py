import numpy as np
import pandas as pd

from src.data.municipal_common import TARGET_PARTIES
from src.data.municipal_polls import load_erc_municipal_polls


def test_load_erc_municipal_polls_basic(tmp_path):
    data = pd.DataFrame(
        {
            "deposit_number": [123, 123],
            "pollster": ["TestOrg", "TestOrg"],
            "fieldwork_start": [pd.Timestamp("2021-09-01"), pd.Timestamp("2021-09-01")],
            "fieldwork_end": [pd.Timestamp("2021-09-05"), pd.Timestamp("2021-09-05")],
            "election_type": ["Autárquicas", "Autárquicas"],
            "geographic_scope": ["Municipal", "Municipal"],
            "region_name": ["Aveiro", "Aveiro"],
            "sample_size": [500, 500],
            "methodology": ["face-to-face", "face-to-face"],
            "margin_of_error": [np.nan, np.nan],
            "extraction_confidence": ["high", "high"],
            "crawl_date": [pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-01")],
            "party": ["PS", "AD – Aliança Democrática"],
            "party_original": ["PS", "AD"],
            "vote_intention_pct": [45.0, 35.0],
            "coalition_members": [np.array([], dtype=str), np.array(["PSD", "CDS"])],
            "municipality_code_full": ["LOCAL-010500", "LOCAL-010500"],
            "municipality_name_official": ["Aveiro", "Aveiro"],
            "district_code": ["LOCAL-010000", "LOCAL-010000"],
            "district_name": ["Aveiro", "Aveiro"],
            "territory_region": ["Centro", "Centro"],
            "municipality_codes": [["LOCAL-010500"], ["LOCAL-010500"]],
            "municipality_names": [["Aveiro"], ["Aveiro"]],
        }
    )

    parquet_path = tmp_path / "polls.parquet"
    data.to_parquet(parquet_path)

    polls = load_erc_municipal_polls([parquet_path], election_dates={2021: "2021-09-26"})

    assert len(polls) == 1
    poll_row = polls.iloc[0]
    assert poll_row["municipality_code"] == "LOCAL-010500"
    assert poll_row["is_poll"] == 1
    assert poll_row["election_year"] == 2021
    assert poll_row["sample_size"] == 500

    # Check effective sample size calculation
    assert "effective_sample" in poll_row.index
    assert "response_rate" in poll_row.index
    assert poll_row["effective_sample"] <= poll_row["sample_size"]
    assert 0 < poll_row["response_rate"] <= 1.0

    # total_votes should equal effective_sample
    assert poll_row["total_votes"] == poll_row["effective_sample"]

    votes_total = sum(poll_row[f"votes_{party}"] for party in TARGET_PARTIES)
    assert votes_total == poll_row["effective_sample"]

    # Check countdown field
    assert "countdown" in poll_row.index
    assert poll_row["countdown"] >= 0  # Should be positive (poll before election)

    assert poll_row["available_PSD"] == 1
    assert poll_row["available_CDS-PP"] == 1
    assert poll_row["votes_PS"] > poll_row["votes_PSD"]
