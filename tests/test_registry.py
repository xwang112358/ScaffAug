"""Unit tests for the model registry — the swap point of the unified framework.

Every registered backbone must build from unified config keys and produce one
prediction per graph. No dataset/GPU needed.
"""
import pytest

torch = pytest.importorskip("torch")
from torch_geometric.data import Batch  # noqa: E402

from welqrate.registry import build_model, AVAILABLE_MODELS, DEFAULTS


def test_expected_models_registered():
    for name in ("deepgin", "gin", "gcn", "gat"):
        assert name in AVAILABLE_MODELS


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_build_and_forward(name, synthetic_graphs):
    model = build_model(name, in_channels=12, cfg={}).eval()
    batch = Batch.from_data_list(synthetic_graphs)
    with torch.no_grad():
        out = model(batch)
    n_graphs = int(batch.batch.max().item()) + 1
    assert out.shape[0] == n_graphs
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_defaults_exist(name):
    assert name in DEFAULTS and "hidden_channels" in DEFAULTS[name]


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("not_a_model", in_channels=12, cfg={})
