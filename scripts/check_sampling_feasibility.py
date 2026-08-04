#!/usr/bin/env python
"""Pilot-gate step 1 (docs/PROJECT.md Section 8): before writing any
encoder/training code, confirm that hard-negative and hard-positive crop
pairs actually exist in usable quantity at the configured crop size and
radii. Only reads `skeleton.npy` (and optional `radius.npy`) per case — no
image volumes are loaded, no GPU needed.

Usage:
    python scripts/check_sampling_feasibility.py --data-dir data/processed/pilot --config configs/pilot.yaml [--n-cases 10]

Exits non-zero if any case fails to produce all three pools at the
configured sizes, printing the InsufficientPairsError diagnostic for each
failure so the radii/crop_size can be retuned before proceeding.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.connectivity_dataset import ConnectivityPairSampler, InsufficientPairsError, SamplerConfig, load_case_graph  # noqa: E402


def main(data_dir: Path, config: SamplerConfig, n_cases: int | None) -> bool:
    cases = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    if not cases:
        print(f"FAIL: no case directories found under {data_dir}")
        return False
    if n_cases is not None:
        cases = cases[:n_cases]

    all_ok = True
    for case in cases:
        try:
            vg, arclength_index = load_case_graph(case)
            sampler = ConnectivityPairSampler(vg, arclength_index, config)
            pools = sampler.build_pools()
        except InsufficientPairsError as e:
            print(f"{case.name}: FAIL — {e}")
            all_ok = False
            continue
        except Exception as e:  # noqa: BLE001 — surface any unexpected error per-case, don't abort the whole sweep
            print(f"{case.name}: ERROR — {type(e).__name__}: {e}")
            all_ok = False
            continue
        sizes = {name: len(pool) for name, pool in pools.items()}
        print(f"{case.name}: OK — {sizes}")

    print()
    print("PASS: all cases produced usable pool sizes" if all_ok else
          "FAIL: at least one case could not produce usable pool sizes — "
          "retune crop_size/near_radius/far_radius/hop_k in configs/pilot.yaml "
          "before writing any encoder/training code")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-cases", type=int, default=None, help="limit to the first N cases (default: all)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sampler_config = SamplerConfig(**cfg["sampler"])

    ok = main(Path(args.data_dir), sampler_config, args.n_cases)
    sys.exit(0 if ok else 1)
