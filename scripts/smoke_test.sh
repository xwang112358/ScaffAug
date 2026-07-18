#!/usr/bin/env bash
# Quick smoke + unit test run. No dataset download, CPU is fine.
#   bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Running unit + smoke tests (metrics, models, training loop)..."
python -m pytest tests/ -v
