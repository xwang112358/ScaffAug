#!/usr/bin/env bash
# Sweep the unified runner over all datasets x splits for a given (model, method).
#
# Usage:
#   bash scripts/run.sh <method> <model>
#     method: baseline | augment | selftrain     (default: selftrain)
#     model:  deepgin | gin | gcn | gat          (default: deepgin)
#
# Examples:
#   bash scripts/run.sh selftrain deepgin     # ScaffAug self-training, DeepGIN (paper setting)
#   bash scripts/run.sh baseline gcn
#
# Edit DATASETS / SPLITS to run a subset. Runs sequentially; pick a GPU with
# CUDA_VISIBLE_DEVICES=<id> before the command.
set -euo pipefail
cd "$(dirname "$0")/.."

METHOD="${1:-selftrain}"
MODEL="${2:-deepgin}"

DATASETS=(AID1798 AID2689 AID463087 AID485290 AID488997)
SPLITS=(random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 \
        scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5)

echo "Running: model=$MODEL  method=$METHOD  over ${#DATASETS[@]} datasets x ${#SPLITS[@]} splits"
for dataset in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "==> $dataset / $split"
    python run.py --dataset "$dataset" --split "$split" --model "$MODEL" --method "$METHOD"
  done
done
echo "Done. Aggregate ./results/*.csv with plot_result.ipynb."
