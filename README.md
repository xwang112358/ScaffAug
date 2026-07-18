# ScaffAug

**Scaffold-Aware Generative Augmentation and Reranking for Enhanced Virtual Screening.**

ScaffAug improves ligand-based virtual screening (VS) under severe class and scaffold imbalance
through three modules:

1. **Augmentation** — a scaffold-aware sampling (SAS) algorithm builds a scaffold library favoring
   underrepresented active scaffolds, and a graph diffusion model (DiGress) generates
   scaffold-conditioned molecules, forming a *generative diverse scaffold-augmented* (G-DSA) dataset.
2. **Self-training** — the G-DSA molecules are integrated into training via confidence-based
   pseudo-labeling (only chemically valid, confidently-predicted molecules are used).
3. **Reranking** — Maximal Marginal Relevance (MMR) reranks the top predictions to increase scaffold
   diversity while preserving early enrichment.

This repository contains the **training / self-training / evaluation** code (modules 2–3 and the
consumer of module 1). The **generation** module (SAS + DiGress) lives in a companion repository —
see [Augmentation module](#augmentation-module).

---

## Installation

```bash
conda create -n scaffaug python=3.9 -c conda-forge rdkit=2023.03.2
conda activate scaffaug

# PyTorch (match your CUDA; example: CUDA 12.1)
conda install pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# PyG extensions (match torch/CUDA)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install -e .
pip install -r requirements.txt
```

## Data

**WelQrate screening datasets** (5 assays: AID1798, AID2689, AID463087, AID485290, AID488997) are
downloaded automatically on first use by the `welqrate` loader into `./welqrate_datasets/`:

```python
from welqrate.dataset import WelQrateDataset
ds = WelQrateDataset(dataset_name="AID1798", root="./welqrate_datasets", mol_repr="2dmol")
split = ds.get_idx_split("random_cv1")   # or scaffold_seed1, ...
```

Each dataset provides 5 random (`random_cv1..5`) and 5 scaffold (`scaffold_seed1..5`) splits.

**Augmentation graphs (G-DSA).** The self-training / augmentation modes consume pre-generated
scaffold-conditioned graphs, expected under:

```
augment_pyg_datasets/<AID>_<split>_0.1_generated_pyg_graphs.pt             # unlabeled, for self-training
augment_pyg_graphs_labels/<AID>_<split>_0.1_augment_pyg_graphs_labels.pt   # labeled, for --aug
augment_valid_pyg_graphs_labels/<AID>_<split>_0.1_..._graphs_labels.pt     # valid-only, for --valid
```

Generate these with the [augmentation module](#augmentation-module).

## Usage — unified runner

Everything runs through one entry point, `run.py`, where you **swap dataset × model × method**
from the command line:

```bash
python run.py --dataset <AID> --split <split> --model <backbone> --method <method>
```

- `--dataset` : `AID1798 | AID2689 | AID463087 | AID485290 | AID488997`
- `--split`   : `random_cv1..5 | scaffold_seed1..5`
- `--model`   : `deepgin` (default, the paper's backbone) `| gin | gcn | gat`
- `--method`  : `selftrain` (default) `| baseline | augment`

**The paper's main setting** is ScaffAug self-training with the DeepGIN backbone:

```bash
python run.py --dataset AID1798 --split random_cv1 --model deepgin --method selftrain
```

Other examples:

```bash
python run.py --dataset AID2689 --split scaffold_seed1 --model gcn --method baseline
python run.py --dataset AID2689 --split scaffold_seed1 --model gin --method augment
```

Sweep all datasets × splits:

```bash
bash scripts/run.sh selftrain deepgin     # paper setting, all datasets x splits
bash scripts/run.sh baseline gcn
```

Each run writes a per-split result CSV to `./results/` (`logAUC, BEDROC, EF100, DCG100, ...`).
Aggregate them into the paper tables with `plot_result.ipynb`.

> New backbones plug in by adding a builder to `welqrate/registry.py`; new methods by adding a
> branch in `run.py`. Config defaults live in `configs/base.yaml` (+ per-model defaults in the registry).

### Testing

```bash
bash scripts/smoke_test.sh          # unit + smoke tests (metrics, registry, models, training loop)
pytest                              # same; real-data test auto-skips without ./welqrate_datasets
```

### Hyperparameter search

Add `--tune` to run an Optuna search over the paper's hyperparameter pools (backbone size, depth,
learning rate, and — for self-training — confidence threshold, pseudo-label frequency, warm-up),
then evaluate the best configuration over `--n-seeds` seeds:

```bash
python run.py --dataset AID1798 --split random_cv1 --model deepgin --method selftrain \
    --tune --n-trials 20 --n-seeds 3
```

The result CSV then reports the mean ± std of every metric over the seeds. Without `--tune`, `run.py`
performs a single run with the config / CLI hyperparameters.

## Evaluation metrics

Early-recognition VS metrics (in `welqrate/utils/evaluation.py`): **logAUC** (`[0.001, 0.1]`),
**BEDROC** (α=20), **EF₁₀₀**, **DCG₁₀₀** (plus EF/DCG at 500/1000). Model selection uses validation
BEDROC.

## Repository layout

```
run.py                    # unified entry point: swap dataset x model x method
welqrate/                 # core package
  dataset.py              #   WelQrate dataset + splits
  registry.py             #   model registry (build_model, AVAILABLE_MODELS) — the swap point
  models/gnn2d/           #   backbones: UpGIN (DeepGIN), GIN, GCN, GAT
  train.py                #   baseline trainer
  train_aug.py            #   augmentation trainer
  scaffaug_st.py          #   ScaffAug self-training (pseudo-labeling)
  rerank.py               #   MMR reranking + SD_100 scaffold-diversity metric
  loader.py, scheduler.py #   data loaders + LR schedule
  utils/evaluation.py     #   logAUC / BEDROC / EF / DCG
configs/base.yaml         # base config (TRAIN / GENERAL / AUGMENTATION defaults)
tests/                    # unit + smoke tests (pytest)
scripts/run.sh            # sweep run.py over all datasets x splits
scripts/smoke_test.sh     # quick test runner
plot_result.ipynb         # aggregate result CSVs -> tables
```

## Augmentation module

The SAS sampling + DiGress scaffold-conditioned generation that produce the G-DSA graphs are in the
companion repository **Scaffolding_Digress** (separate conda environment). See that repo to build the
scaffold library and generate the augmentation graphs, then place the resulting `.pt` files in the
directories listed under [Data](#data).

## Citation

If you use this code, please cite:

> Wang, Xin, et al. "Scaffold-Aware Generative Augmentation and Reranking for Enhanced Virtual
> Screening." *arXiv preprint arXiv:2510.16306* (2025).

```bibtex
@article{wang2025scaffaug,
  title   = {Scaffold-Aware Generative Augmentation and Reranking for Enhanced Virtual Screening},
  author  = {Wang, Xin and Wang, Yu and Liu, Yunchao and Meiler, Jens and Derr, Tyler},
  journal = {arXiv preprint arXiv:2510.16306},
  year    = {2025}
}
```
