"""Smoke test: the training loop runs end-to-end on synthetic graphs.

No dataset, no GPU required (CPU is fine). Verifies a few optimization steps produce
finite losses and actually update the model parameters.
"""
import pytest

torch = pytest.importorskip("torch")
from torch.nn import BCEWithLogitsLoss           # noqa: E402
from torch_geometric.loader import DataLoader     # noqa: E402

from tests.conftest import IN_CHANNELS


def test_train_steps_update_params(synthetic_graphs):
    from welqrate.models.gnn2d.GIN import GIN_Model
    model = GIN_Model(in_channels=IN_CHANNELS, hidden_channels=16, num_layers=2)
    loader = DataLoader(synthetic_graphs, batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = BCEWithLogitsLoss()

    before = [p.detach().clone() for p in model.parameters()]
    losses = []
    model.train()
    for _ in range(3):
        for batch in loader:
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out.view(-1), batch.y.view(-1).float())
            loss.backward()
            opt.step()
            losses.append(loss.item())

    assert all(l == l and l != float("inf") for l in losses)   # finite (no NaN/inf)
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))  # params moved
