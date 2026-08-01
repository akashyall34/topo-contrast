#!/usr/bin/env bash
# Fine-tune a segmentation model from the pretrained image encoder, then
# evaluate (Dice, clDice, Betti errors, NSD, data-efficiency curves).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train_finetune --config configs/full_run.yaml
