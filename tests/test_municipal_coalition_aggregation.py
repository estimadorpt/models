import numpy as np

from src.models.municipal_coupling_model import MunicipalCouplingModel


def _make_stub_model(parties, coalition_mappings=None):
    model = MunicipalCouplingModel.__new__(MunicipalCouplingModel)
    model.parties = parties
    model.party_to_index = {party: idx for idx, party in enumerate(parties)}
    model.coalition_mappings = coalition_mappings or {}
    return model


def test_aggregate_vector_to_coalitions_merges_components():
    parties = [
        "PS",
        "PSD",
        "CDS-PP",
        "IL",
        "LOCAL_INC",
        "OTHER",
    ]
    model = _make_stub_model(parties)

    vector = np.array([0.35, 0.28, 0.12, 0.08, 0.10, 0.07])
    mapping = {
        "PPD/PSD.CDS-PP.A.MPT.PPM": ["PSD", "CDS-PP"],
        "PS.L": ["PS"],
    }

    aggregated = model._aggregate_vector_to_coalitions(vector, mapping)

    assert np.isclose(aggregated["PPD/PSD.CDS-PP.A.MPT.PPM"], 0.40)
    assert np.isclose(aggregated["PS.L"], 0.35)
    assert np.isclose(sum(aggregated.values()), 1.0)


def test_aggregate_samples_to_coalitions_normalizes_rows():
    parties = ["PS", "PSD", "CDS-PP", "OTHER"]
    mapping = {
        "CentroDireita": ["PSD", "CDS-PP"],
        "PS": ["PS"],
    }
    model = _make_stub_model(parties, coalition_mappings={2021: {"dummy": mapping}})
    samples = np.array(
        [
            [0.30, 0.25, 0.05, 0.40],
            [0.28, 0.32, 0.10, 0.30],
        ]
    )
    aggregated = model.aggregate_samples_to_coalitions(samples, 2021, "dummy")

    assert set(aggregated.keys()) == {"CentroDireita", "PS", "OTHER"}
    stacked = np.column_stack(list(aggregated.values()))
    row_sums = stacked.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
    assert np.isclose(aggregated["CentroDireita"][0], 0.30)
    assert np.isclose(aggregated["CentroDireita"][1], 0.42)
