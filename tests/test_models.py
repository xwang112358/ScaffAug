"""Unit tests for the 2D GNN backbones: a forward pass yields one prediction per graph."""
import pytest

torch = pytest.importorskip("torch")
from torch_geometric.data import Batch  # noqa: E402

from tests.conftest import IN_CHANNELS


def _build(name):
    if name == "gin":
        from welqrate.models.gnn2d.GIN import GIN_Model
        return GIN_Model(in_channels=IN_CHANNELS, hidden_channels=32, num_layers=3)
    if name == "gcn":
        from welqrate.models.gnn2d.GCN import GCN_Model
        return GCN_Model(in_channels=IN_CHANNELS, hidden_channels=32, num_layers=3)
    if name == "gat":
        from welqrate.models.gnn2d.GAT import GAT_Model
        return GAT_Model(in_channels=IN_CHANNELS, hidden_channels=32, num_layers=3, heads=2)
    raise ValueError(name)


@pytest.mark.parametrize("name", ["gin", "gcn", "gat"])
def test_forward_output_shape(name, synthetic_graphs):
    model = _build(name).eval()
    batch = Batch.from_data_list(synthetic_graphs)
    with torch.no_grad():
        out = model(batch)
    n_graphs = int(batch.batch.max().item()) + 1
    assert out.shape[0] == n_graphs           # one prediction per graph
    assert torch.isfinite(out).all()


def test_forward_is_deterministic_in_eval(synthetic_batch):
    model = _build("gin").eval()
    with torch.no_grad():
        a = model(synthetic_batch)
        b = model(synthetic_batch)
    assert torch.allclose(a, b)
