# Image–Topology Contrastive Pretraining for Tubular Anatomy Segmentation

[Paper link — TBD] · [MICCAI 2027 target]

## Overview

Pretraining a 3D image encoder against a graph encoder of the vessel
skeleton (nodes = bifurcations/endpoints, edges = centerline segments),
via contrastive alignment, so the image encoder learns branch-identity
and topology-aware features before segmentation fine-tuning.

## Installation

```bash
conda env create -f environment.yml
conda activate tubular-topo
```

## Data

- [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT) — primary
- [ASOCA](https://asoca.grand-challenge.org/) — secondary validation

Run `data/prepare_imagecas.py` to download/preprocess into `data/processed/`.

## Pilot: branch-identity probe (go/no-go gate)

Before investing in the full pretraining pipeline, this pilot answers one
question on a ~30-50 volume subset: **does an image encoder aligned to
skeleton-graph embeddings actually learn branch identity better than an
unaligned baseline?**

```bash
bash scripts/run_pilot.sh
```

See `docs/project-proposal.md` for the full rationale and success criteria.

## Full pretraining

```bash
bash scripts/run_pretrain.sh
```

## Fine-tuning / evaluation

```bash
bash scripts/run_finetune.sh
```

Metrics: Dice, clDice, Betti number errors, NSD, and data-efficiency curves
(performance vs. % of labeled fine-tuning data).

## Citation

TBD on submission.
