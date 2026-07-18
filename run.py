#!/usr/bin/env python3
"""Unified ScaffAug runner: swap dataset x model x method from one entry point.

Examples
--------
# Single run — ScaffAug self-training with the DeepGIN backbone (the paper's main setting):
python run.py --dataset AID1798 --split random_cv1 --model deepgin --method selftrain

# Plain baseline / generative augmentation with any backbone:
python run.py --dataset AID2689 --split scaffold_seed1 --model gcn --method baseline
python run.py --dataset AID2689 --split scaffold_seed1 --model gin  --method augment

# Hyperparameter search (Optuna), then evaluate the best config over N seeds:
python run.py --dataset AID1798 --split random_cv1 --model deepgin --method selftrain --tune --n-trials 20

Methods
-------
  baseline  : train on the labeled set only            (welqrate.train.train)
  augment   : add labeled G-DSA graphs                 (welqrate.train_aug.train_aug)
  selftrain : pseudo-label unlabeled G-DSA graphs      (welqrate.scaffaug_st.train_pseudo_label)
"""
import argparse
import copy
import csv
import os

import yaml

METRICS = ["logAUC", "EF100", "DCG100", "BEDROC", "EF500", "EF1000", "DCG500", "DCG1000"]
BEDROC = 3  # index of BEDROC in METRICS (selection objective)


def load_config(path, args):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["DATA"]["dataset_name"] = args.dataset
    cfg["DATA"]["split_scheme"] = args.split
    cfg["MODEL"]["model_name"] = args.model
    if args.epochs is not None:
        cfg["TRAIN"]["num_epochs"] = args.epochs
    if args.seed is not None:
        cfg["GENERAL"]["seed"] = args.seed
    for k in ("hidden_channels", "num_layers", "dropout", "graph_pooling"):
        v = getattr(args, k)
        if v is not None:
            cfg["MODEL"][k] = v
    return cfg


def train_eval(cfg, args, dataset, device):
    """Build the model from cfg['MODEL'] and run the chosen method. Returns the 8 test metrics."""
    import torch
    from welqrate.registry import build_model
    from welqrate.loader import get_valid_loader, get_test_loader

    in_channels = dataset[0].x.size(1)
    model = build_model(args.model, in_channels, cfg["MODEL"]).to(device)

    if args.method == "baseline":
        from welqrate.train import train
        return train(model, dataset, cfg, device)

    if args.method == "augment":
        from welqrate.train_aug import train_aug
        path = os.path.join(args.aug_root, f"{args.dataset}_{args.split}_{args.ratio}_augment_pyg_graphs_labels.pt")
        return train_aug(model, dataset, torch.load(path, weights_only=False), cfg, device)

    if args.method == "selftrain":
        from welqrate.scaffaug_st import train_pseudo_label
        bs = int(cfg["TRAIN"]["batch_size"]); nw = int(cfg["GENERAL"]["num_workers"]); seed = int(cfg["GENERAL"]["seed"])
        path = os.path.join(args.st_aug_root, f"{args.dataset}_{args.split}_{args.ratio}_generated_pyg_graphs.pt")
        aug = torch.load(path, weights_only=False)
        split_dict = dataset.get_idx_split(args.split)
        orig_train = [dataset[int(i)] for i in split_dict["train"]]
        valid_loader = get_valid_loader(dataset[split_dict["valid"]], bs, nw, seed)
        test_loader = get_test_loader(dataset[split_dict["test"]], bs, nw, seed)
        save_path = os.path.join(args.out_dir, "checkpoints", f"{args.model}_{args.dataset}_{args.split}")
        os.makedirs(save_path, exist_ok=True)
        return train_pseudo_label(model, cfg, device, aug, orig_train, valid_loader, test_loader, save_path=save_path)

    raise ValueError(args.method)


# ----------------------------- hyperparameter search -----------------------------

def suggest_hp(trial, model, method):
    """Optuna search space (mirrors the paper's hyperparameter pools)."""
    hp = {"MODEL": {}, "TRAIN": {}, "AUGMENTATION": {}}
    if model == "deepgin":
        hp["MODEL"]["hidden_channels"] = trial.suggest_categorical("hidden_channels", [64, 128, 256])
        hp["MODEL"]["num_layers"] = trial.suggest_int("num_layers", 3, 5)
        hp["MODEL"]["dropout"] = trial.suggest_float("dropout", 0.1, 0.3)
        hp["MODEL"]["graph_pooling"] = "sum"
    else:
        hp["MODEL"]["hidden_channels"] = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        hp["MODEL"]["num_layers"] = trial.suggest_int("num_layers", 2, 4)
    hp["TRAIN"]["peak_lr"] = trial.suggest_float("peak_lr", 1e-4, 1e-2, log=True)
    if method == "selftrain":
        hp["AUGMENTATION"]["confidence_threshold"] = trial.suggest_categorical(
            "confidence_threshold", [0.75, 0.8, 0.85, 0.9])
        hp["AUGMENTATION"]["pseudo_label_freq"] = trial.suggest_int("pseudo_label_freq", 3, 10)
        hp["AUGMENTATION"]["start_epoch"] = trial.suggest_int("start_epoch", 20, 30)
    return hp


def apply_hp(cfg, hp):
    out = copy.deepcopy(cfg)
    for section, kv in hp.items():
        out.setdefault(section, {}).update(kv)
    return out


def run_tune(args, base_cfg, dataset, device):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        cfg = apply_hp(base_cfg, suggest_hp(trial, args.model, args.method))
        return train_eval(cfg, args, dataset, device)[BEDROC]  # maximize validation-selected test BEDROC

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    print(f"best trial: BEDROC={study.best_value:.4f}  params={study.best_params}")

    best_cfg = apply_hp(base_cfg, {  # reconstruct sectioned hp from flat best_params
        "MODEL": {k: v for k, v in study.best_params.items()
                  if k in ("hidden_channels", "num_layers", "dropout")},
        "TRAIN": {k: v for k, v in study.best_params.items() if k == "peak_lr"},
        "AUGMENTATION": {k: v for k, v in study.best_params.items()
                         if k in ("confidence_threshold", "pseudo_label_freq", "start_epoch")},
    })
    best_cfg["MODEL"].setdefault("graph_pooling", "sum")

    import numpy as np
    rows = []
    for seed in range(1, args.n_seeds + 1):
        best_cfg["GENERAL"]["seed"] = seed
        m = train_eval(best_cfg, args, dataset, device)
        rows.append(m)
        print(f"  seed {seed}: BEDROC={m[BEDROC]:.4f}")
    arr = np.array(rows)
    mean, std = arr.mean(0), arr.std(0)
    print("FINAL (mean+/-std over seeds): " + "  ".join(
        f"{k}={mean[i]:.4f}+/-{std[i]:.4f}" for i, k in enumerate(METRICS)))
    _write(args, best_cfg, mean, std, study.best_params)


def _write(args, cfg, mean, std=None, params=None):
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.model}_{args.method}_{args.dataset}_{args.split}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        header = ["dataset", "split", "model", "method"] + METRICS + [f"{m}_std" for m in METRICS] * (std is not None)
        w.writerow(header)
        row = [args.dataset, args.split, args.model, args.method] + list(mean) + (list(std) if std is not None else [])
        w.writerow(row)
    print(f"wrote {out}")


def run(args):
    import torch
    from welqrate.dataset import WelQrateDataset

    base_cfg = load_config(args.config, args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = WelQrateDataset(dataset_name=args.dataset, root=args.data_root, mol_repr="2dmol")

    if args.tune:
        run_tune(args, base_cfg, dataset, device)
    else:
        metrics = train_eval(base_cfg, args, dataset, device)
        print("RESULT " + "  ".join(f"{k}={v:.4f}" for k, v in zip(METRICS, metrics)))
        import numpy as np
        _write(args, base_cfg, np.array(metrics))


def build_parser():
    from welqrate.registry import AVAILABLE_MODELS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="AID1798 | AID2689 | AID463087 | AID485290 | AID488997")
    p.add_argument("--split", required=True, help="random_cv1..5 | scaffold_seed1..5")
    p.add_argument("--model", default="deepgin", choices=AVAILABLE_MODELS, help="backbone (default: deepgin)")
    p.add_argument("--method", default="selftrain", choices=["baseline", "augment", "selftrain"])
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--ratio", default="0.1", help="augmentation-ratio tag in the graph filenames")
    p.add_argument("--data-root", default="./welqrate_datasets")
    p.add_argument("--aug-root", default="./augment_pyg_graphs_labels", help="labeled G-DSA graphs (method=augment)")
    p.add_argument("--st-aug-root", default="./augment_pyg_datasets", help="unlabeled G-DSA graphs (method=selftrain)")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--epochs", type=int, default=None, help="override TRAIN.num_epochs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    # hyperparameter search
    p.add_argument("--tune", action="store_true", help="run Optuna search, then eval best over --n-seeds")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-seeds", type=int, default=3)
    # manual model-hp overrides (single-run mode)
    p.add_argument("--hidden-channels", dest="hidden_channels", type=int, default=None)
    p.add_argument("--num-layers", dest="num_layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--graph-pooling", dest="graph_pooling", default=None)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
