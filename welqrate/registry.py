"""Model registry for the unified ScaffAug framework.

Add a new backbone by writing a builder and registering it in ``_BUILDERS``.
All builders take (in_channels, cfg) and return an ``nn.Module`` whose ``forward(batch)``
returns one prediction per graph.

Unified hyperparameter keys (config['MODEL']):
    hidden_channels, num_layers, dropout, graph_pooling
These map to each backbone's native argument names.
"""

# Default MODEL hyperparameters per backbone (overridable via config / CLI).
DEFAULTS = {
    "deepgin": {"hidden_channels": 64, "num_layers": 5, "dropout": 0.2, "graph_pooling": "sum"},
    "gin":     {"hidden_channels": 64, "num_layers": 3, "dropout": 0.2},
    "gcn":     {"hidden_channels": 64, "num_layers": 3, "dropout": 0.2},
    "gat":     {"hidden_channels": 64, "num_layers": 3, "dropout": 0.2, "heads": 2},
}


def _build_deepgin(in_channels, cfg):
    # DeepGIN (UpGIN) — the paper's main backbone; uses an internal atom encoder.
    from welqrate.models.gnn2d.UpGIN import GIN
    return GIN(
        num_layer=int(cfg.get("num_layers", 5)),
        emb_dim=int(cfg.get("hidden_channels", 64)),
        drop_ratio=float(cfg.get("dropout", 0.2)),
        graph_pooling=cfg.get("graph_pooling", "sum"),
    )


def _build_gin(in_channels, cfg):
    from welqrate.models.gnn2d.GIN import GIN_Model
    return GIN_Model(in_channels=in_channels, hidden_channels=int(cfg.get("hidden_channels", 64)),
                     num_layers=int(cfg.get("num_layers", 3)), dropout=float(cfg.get("dropout", 0.2)))


def _build_gcn(in_channels, cfg):
    from welqrate.models.gnn2d.GCN import GCN_Model
    return GCN_Model(in_channels=in_channels, hidden_channels=int(cfg.get("hidden_channels", 64)),
                     num_layers=int(cfg.get("num_layers", 3)), dropout=float(cfg.get("dropout", 0.2)))


def _build_gat(in_channels, cfg):
    from welqrate.models.gnn2d.GAT import GAT_Model
    return GAT_Model(in_channels=in_channels, hidden_channels=int(cfg.get("hidden_channels", 64)),
                     num_layers=int(cfg.get("num_layers", 3)), heads=int(cfg.get("heads", 2)),
                     dropout=float(cfg.get("dropout", 0.2)))


_BUILDERS = {
    "deepgin": _build_deepgin,   # <-- paper's main backbone
    "gin": _build_gin,
    "gcn": _build_gcn,
    "gat": _build_gat,
}

AVAILABLE_MODELS = sorted(_BUILDERS)


def build_model(name, in_channels, cfg=None):
    """Instantiate a backbone by name. `cfg` is config['MODEL'] (dict); defaults fill gaps."""
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"unknown model '{name}'. available: {AVAILABLE_MODELS}")
    merged = {**DEFAULTS.get(key, {}), **(cfg or {})}
    return _BUILDERS[key](in_channels, merged)
