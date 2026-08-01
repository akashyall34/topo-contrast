"""Sweeps % of labeled fine-tuning data and reports segmentation metrics at
each fraction, for pretrained vs. from-scratch encoders — the headline plot
for the full paper (does pretraining reduce labeled-data requirements)."""
import json
from pathlib import Path

DATA_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)


def run_sweep(train_fn, eval_fn, fractions=DATA_FRACTIONS) -> dict:
    """train_fn(fraction) -> trained model; eval_fn(model) -> metrics dict.
    Kept generic so it can drive either the pretrained or from-scratch
    fine-tuning entrypoint without duplicating the sweep logic."""
    results = {}
    for frac in fractions:
        model = train_fn(frac)
        results[frac] = eval_fn(model)
    return results


def save_results(results: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
