#!/usr/bin/env bash
# Full-scale contrastive pretraining. Only run after scripts/run_pilot.sh reports GO.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train_pretrain --config configs/full_run.yaml
