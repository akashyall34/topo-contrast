#!/usr/bin/env bash
# Runs the full pilot: preprocess (if needed) -> contrastive pretrain ->
# branch-identity probe -> prints the GO / NO-GO decision.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=configs/pilot.yaml

if [ ! -d "data/processed/pilot" ] || [ -z "$(ls -A data/processed/pilot 2>/dev/null)" ]; then
    echo "No preprocessed pilot data found in data/processed/pilot/."
    echo "Run: python data/prepare_imagecas.py --subset pilot --n-cases 40"
    exit 1
fi

echo "== Contrastive pretraining (pilot) =="
python -m src.train_pretrain --config "$CONFIG"

echo
echo "== Branch-identity probe =="
python -m src.probes.branch_identity_probe --config "$CONFIG"
