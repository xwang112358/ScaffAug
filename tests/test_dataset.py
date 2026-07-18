"""Smoke test for the WelQrate dataset loader.

Skips automatically when ./welqrate_datasets is absent (e.g. fresh checkout / CI),
so `pytest` is green without a multi-GB download. Run locally after the data is present
to exercise the real loader + split API.
"""
import os
import pytest

torch = pytest.importorskip("torch")

DATA_ROOT = os.path.join(os.getcwd(), "welqrate_datasets")
_has_data = os.path.isdir(os.path.join(DATA_ROOT, "AID1798"))

pytestmark = pytest.mark.skipif(
    not _has_data, reason="welqrate_datasets/AID1798 not present; skipping real-data smoke test"
)


def test_load_dataset_and_split():
    from welqrate.dataset import WelQrateDataset
    ds = WelQrateDataset(dataset_name="AID1798", root=DATA_ROOT, mol_repr="2dmol")
    assert len(ds) > 0
    g0 = ds[0]
    assert hasattr(g0, "x") and hasattr(g0, "edge_index") and hasattr(g0, "y")

    split = ds.get_idx_split("random_cv1")
    for key in ("train", "valid", "test"):
        assert key in split and len(split[key]) > 0
    # splits are disjoint
    train, test = set(split["train"].tolist()), set(split["test"].tolist())
    assert train.isdisjoint(test)
