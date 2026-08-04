"""Connectivity-prediction pretraining (docs/PROJECT.md Section 2): trains
a siamese image encoder + relational head to classify the topological
connectivity relationship (SAME_BRANCH / NEAR_ANCESTOR / DISCONNECTED_OR_FAR)
between two crops from the same volume.

Answers pilot-gate step 3's core question (docs/PROJECT.md Section 8): does
this beat the Euclidean-distance-only floor established by
scripts/check_shortcut_baseline.py (59.8% on the full pilot set)? Both the
trained encoder and a val-split-restricted re-fit of that same baseline are
evaluated here, on identical held-out cases, so the comparison is fair.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from scripts.check_shortcut_baseline import collect_distance_label_pairs
from src.datasets.connectivity_dataset import ConnectivityPairDataset, SamplerConfig
from src.models.image_encoder import PilotImageEncoder
from src.models.relational_head import RelationalHead


def split_cases(data_dir: Path, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Case-level train/val split — pairs from the same case are
    correlated (same anatomy/acquisition), so splitting at the pair level
    would leak information between train and val."""
    cases = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if len(cases) < 2:
        raise ValueError(f"need at least 2 cases to split, found {len(cases)} under {data_dir}")
    rng = np.random.default_rng(seed)
    shuffled = cases.copy()
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    if n_val >= len(shuffled):
        raise ValueError(
            f"val_fraction={val_fraction} leaves no train cases out of {len(shuffled)} total"
        )
    return shuffled[n_val:], shuffled[:n_val]


def class_weights_from_labels(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss. Without this,
    the model can match the distance-only baseline's failure mode — collapse
    to the majority class (DISCONNECTED_OR_FAR, ~60% of pairs) — and look
    like it's "learning" in raw accuracy while doing nothing of the sort."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"a class has zero examples in this split: counts={counts}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(image_encoder, head, loader, device, log_every_n_batches: int = 20) -> float:
    image_encoder.eval()
    head.eval()
    correct, total = 0, 0
    for i, batch in enumerate(loader):
        patch_a = batch["patch_a"].to(device, non_blocking=True)
        patch_b = batch["patch_b"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = head(image_encoder(patch_a), image_encoder(patch_b))
        pred = logits.argmax(dim=-1)
        correct += (pred == label).sum().item()
        total += label.shape[0]
        if (i + 1) % log_every_n_batches == 0:
            print(f"    eval batch {i + 1}", flush=True)
    if total == 0:
        # A silent 0.0 here would be indistinguishable from "the model
        # performs at floor" — but it actually means every val case failed
        # pilot-gate step 1 at this config (empty val_dataset), a distinct
        # and much more important failure to surface loudly.
        raise RuntimeError(
            "evaluate() received an empty loader — every val case must have "
            "failed pilot-gate step 1 at this sampler config; check the "
            "'skipping <case>' warnings printed during dataset construction"
        )
    return correct / total


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Input shape (patch_size, batch_size) is fixed across the run, so
        # cudnn can safely autotune the fastest conv algorithm once instead
        # of re-picking generically every call.
        torch.backends.cudnn.benchmark = True
    data_dir = Path(cfg["data_dir"])
    pretrain_cfg = cfg["pretrain"]
    sampler_config = SamplerConfig(**cfg["sampler"])

    train_cases, val_cases = split_cases(data_dir, pretrain_cfg["val_fraction"], pretrain_cfg["seed"])
    print(f"train cases: {len(train_cases)}  val cases: {len(val_cases)}", flush=True)

    # Default 0, not >0: the model is already moved to `device` (CUDA)
    # before the DataLoader is ever iterated, and PyTorch's default
    # multiprocessing start method on Linux is `fork` — forking a process
    # that already holds an active CUDA context corrupts the child, which
    # is exactly what a bare "DataLoader worker killed by signal" with no
    # further Python traceback means (observed in practice on a real T4
    # Colab run). This is a different, more fundamental problem than the
    # cache-memory-multiplication issue `max_cache_size` guards against
    # below — bounding the cache does not fix it. If num_workers>0 is
    # requested anyway (e.g. via config, for a model large enough to
    # actually benefit), `multiprocessing_context="spawn"` avoids the
    # fork+CUDA corruption since spawned workers start a fresh
    # interpreter rather than forking the CUDA-initialized parent.
    num_workers = pretrain_cfg.get("num_workers", 0)
    # Each DataLoader worker gets its own *separate* copy of the dataset,
    # including its own image cache — with shuffle=True, every worker
    # eventually touches most/all cases over an epoch, so an unbounded
    # cache gets duplicated per worker. Bound it only when workers exist;
    # with num_workers=0 (single process) there's no multiplication risk.
    max_cache_size = 8 if num_workers > 0 else None

    t0 = time.time()
    rng = np.random.default_rng(pretrain_cfg["seed"])
    train_dataset = ConnectivityPairDataset(data_dir, sampler_config, rng=rng, case_ids=train_cases,
                                             max_cache_size=max_cache_size)
    val_dataset = ConnectivityPairDataset(data_dir, sampler_config, rng=rng, case_ids=val_cases,
                                           max_cache_size=max_cache_size)
    print(f"train pairs: {len(train_dataset)}  val pairs: {len(val_dataset)}  "
          f"(dataset construction: {time.time() - t0:.1f}s)", flush=True)

    # pin_memory speeds up the host->device copy; num_workers>0 overlaps
    # CPU-side patch extraction with GPU compute instead of the GPU
    # idling between batches. multiprocessing_context="spawn" only matters
    # (and is only set) when num_workers>0 — see the comment above on why
    # fork (the default) is unsafe once CUDA is already initialized.
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(train_dataset, batch_size=pretrain_cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=pretrain_cfg["batch_size"], shuffle=False, **loader_kwargs)

    image_encoder = PilotImageEncoder(embed_dim=pretrain_cfg["embed_dim"]).to(device)
    head = RelationalHead(embed_dim=pretrain_cfg["embed_dim"], hidden_dim=pretrain_cfg["hidden_dim"]).to(device)
    weights = class_weights_from_labels(train_dataset.all_labels(), num_classes=3).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        list(image_encoder.parameters()) + list(head.parameters()), lr=pretrain_cfg["lr"]
    )

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print at least this often within an epoch, not just once at epoch end
    # — Colab (and similar hosted notebooks) can treat a cell with no
    # output for several minutes as idle and interrupt/disconnect the
    # runtime, even though training is actively progressing.
    log_every_n_batches = 20

    val_acc = 0.0
    for epoch in range(pretrain_cfg["epochs"]):
        epoch_t0 = time.time()
        image_encoder.train()
        head.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            patch_a = batch["patch_a"].to(device, non_blocking=True)
            patch_b = batch["patch_b"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            logits = head(image_encoder(patch_a), image_encoder(patch_b))
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % log_every_n_batches == 0:
                elapsed = time.time() - epoch_t0
                print(f"  epoch {epoch + 1} batch {n_batches}  "
                      f"running_loss={epoch_loss / n_batches:.4f}  ({elapsed:.1f}s elapsed)", flush=True)

        train_time = time.time() - epoch_t0
        eval_t0 = time.time()
        val_acc = evaluate(image_encoder, head, val_loader, device)
        eval_time = time.time() - eval_t0
        print(f"epoch {epoch + 1}/{pretrain_cfg['epochs']}  "
              f"train_loss={epoch_loss / max(n_batches, 1):.4f}  val_acc={val_acc:.4f}  "
              f"(train {train_time:.1f}s, eval {eval_time:.1f}s, {n_batches} batches)", flush=True)

        torch.save(
            {"image_encoder": image_encoder.state_dict(), "relational_head": head.state_dict()},
            output_dir / "latest.pth",
        )

    final_val_acc = val_acc  # from the last completed epoch's evaluate() call above

    t0 = time.time()
    # Seeded, not the default unseeded rng — without this, the baseline
    # comparison silently changes on every run even with the rest of the
    # pipeline fixed by `seed`.
    baseline_rng = np.random.default_rng(pretrain_cfg["seed"])
    train_dist, train_labels = collect_distance_label_pairs(data_dir, sampler_config, case_ids=train_cases,
                                                              rng=baseline_rng)
    val_dist, val_labels = collect_distance_label_pairs(data_dir, sampler_config, case_ids=val_cases,
                                                          rng=baseline_rng)
    baseline = LogisticRegression(max_iter=1000).fit(train_dist, train_labels)
    baseline_val_acc = baseline.score(val_dist, val_labels)
    print(f"(baseline re-fit: {time.time() - t0:.1f}s)", flush=True)

    print()
    print(f"trained encoder val accuracy:        {final_val_acc:.4f}")
    print(f"distance-only baseline val accuracy: {baseline_val_acc:.4f}")
    print(f"gain over distance-only floor:       {final_val_acc - baseline_val_acc:+.4f}")
    return {"encoder_val_acc": final_val_acc, "distance_baseline_val_acc": baseline_val_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
