"""Shared pytest fixtures for ScaffAug tests.

Provides small synthetic molecular graphs so unit/smoke tests run with NO dataset
download and (mostly) no GPU. Real-data tests live in test_dataset.py and self-skip
when ./welqrate_datasets is absent.
"""
import os
import numpy as np
import pytest

try:
    import torch
    from torch_geometric.data import Data, Batch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

IN_CHANNELS = 12  # WelQrate 2dmol node-feature dim


def _random_graph(n_nodes, in_channels=IN_CHANNELS, label=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, in_channels, generator=g)
    # a simple connected-ish random edge set (undirected)
    src = torch.randint(0, n_nodes, (2 * n_nodes,), generator=g)
    dst = torch.randint(0, n_nodes, (2 * n_nodes,), generator=g)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([float(label)]))


@pytest.fixture
def synthetic_graphs():
    """A tiny imbalanced list of graphs (like a screening set: few actives)."""
    if not HAS_TORCH:
        pytest.skip("torch / torch_geometric not installed")
    graphs = []
    for i in range(24):
        label = 1 if i < 4 else 0            # 4 actives / 24
        graphs.append(_random_graph(n_nodes=8 + (i % 5), label=label, seed=i))
    return graphs


@pytest.fixture
def synthetic_batch(synthetic_graphs):
    return Batch.from_data_list(synthetic_graphs)


@pytest.fixture
def ranking_example():
    """Known (true_y, score) rankings for metric tests: 10 actives in 1000 (~1%,
    a realistic virtual-screening active rate that keeps BEDROC within [0, 1])."""
    n, n_act = 1000, 10
    true_y = np.array([1] * n_act + [0] * (n - n_act), dtype=float)
    perfect = np.linspace(1.0, 0.0, n)                      # actives first
    reverse = np.linspace(0.0, 1.0, n)                      # actives last
    rng = np.random.default_rng(0)
    random_ = rng.permutation(perfect)
    return true_y, perfect, reverse, random_, n, n_act
