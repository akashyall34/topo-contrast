#!/usr/bin/env python
"""Pilot-gate step 2 (docs/PROJECT.md Section 8): the shortcut-learning
falsification check.

Topological connectivity correlates with Euclidean proximity (vessels are
spatially continuous curves), so a model could solve the connectivity-
prediction pretext task by implicitly estimating crop-center distance
rather than reasoning about actual anatomy. This script trains a
classifier using *only* the Euclidean distance between crop centers (no
image content at all) and reports its cross-validated accuracy predicting
the 3-class connectivity label.

This number is the bar any real image-based model must clear by a real
margin (docs/PROJECT.md Section 8 point 2) — if a trained siamese encoder
doesn't beat this, the task is a distance regression in disguise.

Only needs skeleton.npy per case (via ConnectivityPairSampler) — no image
volumes loaded, matching scripts/check_sampling_feasibility.py's economy.

Usage:
    python scripts/check_shortcut_baseline.py --data-dir data/processed/pilot --config configs/pilot.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.connectivity_dataset import (  # noqa: E402
    LABEL_TO_ID,
    ConnectivityPairSampler,
    InsufficientPairsError,
    SamplerConfig,
    load_case_graph,
)


def collect_distance_label_pairs(
    data_dir: Path, config: SamplerConfig, case_ids: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (distances, label_ids) across every pool, every case —
    skips (with a warning) any case that fails pilot-gate step 1, rather
    than letting one bad case abort the whole run.

    `case_ids`: restrict to these case directory names only (e.g. a val
    split) — used by src/train_connectivity_pretrain.py to re-fit this
    baseline on the same held-out cases the trained encoder is evaluated
    on, so the comparison is apples-to-apples rather than reusing the
    whole-pilot-set number from a different split.

    `rng`: without this, each `ConnectivityPairSampler` below defaulted to
    an unseeded `np.random.default_rng()` (OS entropy) — meaning this
    function, unlike every other seeded piece of this pipeline, silently
    returned different sampled pairs (and thus a slightly different
    baseline accuracy) on every run. Pass a seeded generator for
    reproducible pilot-gate decisions.
    """
    distances, labels = [], []
    if case_ids is not None:
        cases = [data_dir / c for c in sorted(case_ids)]
        missing = [c for c in cases if not c.is_dir()]
        if missing:
            raise FileNotFoundError(f"case directories not found: {missing}")
    else:
        cases = sorted(p for p in data_dir.iterdir() if p.is_dir())
    for case in cases:
        try:
            vg, arclength_index = load_case_graph(case)
            sampler = ConnectivityPairSampler(vg, arclength_index, config, rng=rng)
            pools = sampler.build_pools()
        except InsufficientPairsError as e:
            print(f"skipping {case.name} (failed pilot-gate step 1): {e}")
            continue
        for pool in pools.values():
            for coord_a, coord_b, label in pool:
                distances.append(ConnectivityPairSampler._euclidean(coord_a, coord_b))
                labels.append(LABEL_TO_ID[label])
    return np.asarray(distances, dtype=np.float64).reshape(-1, 1), np.asarray(labels, dtype=np.int64)


def main(data_dir: Path, config: SamplerConfig, cv_folds: int, seed: int = 0) -> float:
    distances, labels = collect_distance_label_pairs(data_dir, config, rng=np.random.default_rng(seed))
    print(f"collected {len(labels)} labeled pairs across the pilot set")

    class_counts = {c: int((labels == c).sum()) for c in sorted(set(labels.tolist()))}
    print(f"label distribution (0=SAME_BRANCH, 1=NEAR_ANCESTOR, 2=DISCONNECTED_OR_FAR): {class_counts}")
    majority_baseline = max(class_counts.values()) / len(labels)
    print(f"majority-class baseline accuracy: {majority_baseline:.4f}")

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, distances, labels, cv=cv_folds)
    distance_only_acc = float(scores.mean())
    print(f"distance-only classifier accuracy ({cv_folds}-fold CV): {distance_only_acc:.4f} (+/- {scores.std():.4f})")

    print()
    print(
        "This is the number a real image-based connectivity-prediction encoder must beat "
        "by a real margin — not majority-class accuracy, and not chance. If it doesn't, "
        "the pretext task is a distance regression in disguise (docs/PROJECT.md Section 8)."
    )
    return distance_only_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sampler_config = SamplerConfig(**cfg["sampler"])

    main(Path(args.data_dir), sampler_config, args.cv_folds, args.seed)
