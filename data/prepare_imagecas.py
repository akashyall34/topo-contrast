"""Preprocess ImageCAS volumes into data/processed/<subset>/<case_id>/.

For each case, writes:
    image.npy      (D, H, W) float32, intensity-normalized
    skeleton.npy   (D, H, W) bool, vessel centerline (from Frangi vesselness
                   on raw intensity only — see docs/PROJECT.md Section 8
                   point 5, no pretraining-time label leakage)
    radius.npy     (D, H, W) float32, distance transform of the Frangi-
                   derived vessel mask
    label.npy      (D, H, W) bool, ground-truth vessel mask — saved for
                   later supervised fine-tuning (src/train_finetune.py),
                   not read anywhere in the pretraining path above

Expects the raw ImageCAS Kaggle layout: `<raw_root>` searched recursively
for `<case_id>.img.nii.gz` / `<case_id>.label.nii.gz` pairs (as unzipped
from the Kaggle "1-200"/"201-400"/... chunks, e.g.
data/raw/imagecas_1-200/1-200/1.img.nii.gz) — not a flat per-case
subdirectory.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from src.preprocessing.hessian_frangi import keep_largest_components, threshold_vesselness, vesselness
from src.preprocessing.skeletonize import skeletonize_mask

# Coronary CTA vessel-visualization HU window. Placeholder — chosen to keep
# contrast-enhanced lumen and soft tissue while dropping bone/air extremes;
# not validated against a windowing study. Revisit before a real run.
_HU_CLIP_MIN = -200.0
_HU_CLIP_MAX = 800.0

# Fixed, dataset-level crop (fraction of each axis), applied identically to
# every case — NOT derived per-case from that case's own label, so this
# does not violate the no-pretraining-time-label-leakage constraint
# (docs/PROJECT.md Section 8 point 5). Whole-volume Frangi vesselness
# ranks ribs/aorta/other thoracic vasculature above the actual coronary
# tree (measured: Dice ~0.03-0.07 against ground truth at every percentile
# threshold tried, on case 1); this crop, derived by inspecting where the
# label sits across a handful of cases, removes rib-cage periphery and
# roughly triples Dice (~0.06 -> ~0.156) while retaining 100% of the label
# in the one case checked. Still a real limitation, not a solved problem —
# revisit with a proper heart-localization step before a non-pilot run.
_CROP_Z = (0.00, 0.75)
_CROP_Y = (0.15, 0.85)
_CROP_X = (0.30, 0.95)

# Percentile threshold for the Frangi-derived mask, tuned against ground
# truth on the crop above (case 1: Dice peaks near here, falls off both
# directions). See same caveat as above.
_VESSELNESS_PERCENTILE = 99.5

# Keep only the N largest connected components of the thresholded mask
# (see keep_largest_components docstring) — label-free, roughly doubles
# Dice on top of the crop+threshold tuning above (case 1: 0.156 -> 0.229).
_KEEP_N_COMPONENTS = 3

# Binary opening before skeletonizing. Without this, case 1's graph had
# 319 nodes / 460 edges with median edge length 1.7 voxels and 211
# degree>=3 "bifurcations" — almost entirely surface-roughness artifacts,
# not real branch points (a real coronary tree has on the order of tens,
# not hundreds). Opening trades raw mask Dice against ground truth (0.229
# -> 0.110) for a graph whose *structure* is usable for the connectivity-
# prediction task (76 nodes, 3 bifurcations, edges up to 111 voxels):
# structural cleanliness matters more here than voxel-level overlap, since
# spurious bifurcations directly corrupt the pretraining connectivity
# labels this whole pipeline exists to produce. Only checked on one case —
# revisit before scaling past the pilot.
_OPENING_STRUCTURE_SIZE = 3


def _fixed_crop(volume: np.ndarray) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    """Applies the fixed dataset-level crop (see _CROP_Z/Y/X above) to a
    (D, H, W) volume. Returns (cropped_volume, slices) so the same slices
    can be applied consistently to image/label/etc for one case."""
    d, h, w = volume.shape
    z0, z1 = int(_CROP_Z[0] * d), int(_CROP_Z[1] * d)
    y0, y1 = int(_CROP_Y[0] * h), int(_CROP_Y[1] * h)
    x0, x1 = int(_CROP_X[0] * w), int(_CROP_X[1] * w)
    slices = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    return volume[slices], slices


def _find_cases(raw_root: Path) -> dict[str, Path]:
    """Maps case_id -> directory containing that case's NIfTI pair,
    searching recursively (the real Kaggle layout nests cases several
    directories deep, unlike the flat per-case-subdirectory layout this
    script originally assumed)."""
    cases: dict[str, Path] = {}
    for img_path in sorted(raw_root.rglob("*.img.nii.gz")):
        case_id = img_path.name[: -len(".img.nii.gz")]
        label_path = img_path.parent / f"{case_id}.label.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"missing label for case {case_id}: expected {label_path}")
        if case_id in cases:
            raise ValueError(
                f"duplicate case id {case_id!r} found at both {cases[case_id]} "
                f"and {img_path.parent} — case ids must be unique under {raw_root}"
            )
        cases[case_id] = img_path.parent
    return cases


def _load_raw_case(case_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads one ImageCAS case's image + ground-truth vessel mask.

    Returns (image, ground_truth_mask) as (D, H, W) float32 / bool arrays.
    nibabel loads NIfTI data as (X, Y, Z); this codebase's skeleton/graph
    code treats coords as (z, y, x) triples with axis 0 = slice, so axes
    are reordered here to (Z, Y, X) once, at the load boundary.
    """
    img_nii = nib.load(case_dir / f"{case_id}.img.nii.gz")
    label_nii = nib.load(case_dir / f"{case_id}.label.nii.gz")

    image = np.asarray(img_nii.get_fdata(), dtype=np.float32)
    label = np.asarray(label_nii.get_fdata(), dtype=np.float32)
    if image.shape != label.shape:
        raise ValueError(f"case {case_id}: image shape {image.shape} != label shape {label.shape}")

    image = np.transpose(image, (2, 0, 1))  # (X, Y, Z) -> (Z, Y, X)
    label = np.transpose(label, (2, 0, 1))

    image = np.clip(image, _HU_CLIP_MIN, _HU_CLIP_MAX)
    image = (image - _HU_CLIP_MIN) / (_HU_CLIP_MAX - _HU_CLIP_MIN)

    ground_truth_mask = label > 0.5
    return image.astype(np.float32), ground_truth_mask


def process_case(case_id: str, case_dir: Path, out_dir: Path):
    image, ground_truth_mask = _load_raw_case(case_dir, case_id)

    # Fixed dataset-level crop (not derived from this case's own label —
    # see _CROP_Z/Y/X above), applied identically to image and label so
    # all saved arrays stay in the same coordinate frame.
    image, slices = _fixed_crop(image)
    ground_truth_mask = ground_truth_mask[slices]

    # Pretraining-time vessel mask/skeleton must come from raw intensity
    # only (Frangi vesselness) — never from `ground_truth_mask` — see
    # docs/PROJECT.md Section 8 point 5.
    vmap = vesselness(image)
    frangi_mask = threshold_vesselness(vmap, percentile=_VESSELNESS_PERCENTILE)
    frangi_mask = keep_largest_components(frangi_mask, n=_KEEP_N_COMPONENTS)
    frangi_mask = ndimage.binary_opening(
        frangi_mask, structure=np.ones((_OPENING_STRUCTURE_SIZE,) * 3)
    ).astype(np.uint8)

    skeleton = skeletonize_mask(frangi_mask)
    radius = ndimage.distance_transform_edt(frangi_mask)

    case_out = out_dir / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    np.save(case_out / "image.npy", image.astype(np.float32))
    np.save(case_out / "skeleton.npy", skeleton.astype(bool))
    np.save(case_out / "radius.npy", radius.astype(np.float32))
    np.save(case_out / "label.npy", ground_truth_mask.astype(bool))


def _process_case_worker(args: tuple[str, Path, Path]) -> tuple[str, str | None]:
    """Picklable wrapper for multiprocessing: returns (case_id, error_message)
    — error_message is None on success. Errors are caught per-case rather
    than left to kill the whole pool, since one corrupt/oddly-shaped case
    shouldn't lose the rest of a multi-hour batch."""
    case_id, case_dir, out_dir = args
    try:
        process_case(case_id, case_dir, out_dir)
        return case_id, None
    except Exception as e:  # noqa: BLE001 — deliberately broad, see docstring
        return case_id, f"{type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--raw-root", type=str, default="data/raw",
                         help="Root searched recursively for '<id>.img.nii.gz' / '<id>.label.nii.gz' pairs")
    parser.add_argument("--case-ids-file", type=str, default=None,
                         help="Optional text file, one case id per line. "
                              "Overrides --n-cases if given.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Parallel worker processes. Frangi vesselness on a "
                              "cropped volume peaks at a few GB per worker — pick "
                              "this based on available RAM, not just core count.")
    args = parser.parse_args()

    out_dir = Path("data/processed") / args.subset
    raw_root = Path(args.raw_root)
    cases = _find_cases(raw_root)
    if not cases:
        raise FileNotFoundError(f"no '*.img.nii.gz' files found under {raw_root}")

    if args.case_ids_file:
        case_ids = Path(args.case_ids_file).read_text().split()
        missing = [c for c in case_ids if c not in cases]
        if missing:
            raise FileNotFoundError(f"case ids not found under {raw_root}: {missing}")
    else:
        case_ids = sorted(cases, key=lambda c: int(c) if c.isdigit() else c)[: args.n_cases]

    work_items = [(case_id, cases[case_id], out_dir) for case_id in case_ids]

    if args.workers <= 1:
        for case_id, case_dir, out in work_items:
            print(f"processing {case_id}")
            err = _process_case_worker((case_id, case_dir, out))[1]
            if err:
                print(f"  FAILED {case_id}: {err}")
        return

    import multiprocessing as mp
    failures = []
    with mp.Pool(processes=args.workers) as pool:
        for case_id, err in pool.imap_unordered(_process_case_worker, work_items):
            if err is None:
                print(f"done: {case_id}")
            else:
                print(f"FAILED: {case_id}: {err}")
                failures.append((case_id, err))

    print(f"\n{len(work_items) - len(failures)}/{len(work_items)} cases succeeded")
    if failures:
        print("Failed cases:")
        for case_id, err in failures:
            print(f"  {case_id}: {err}")


if __name__ == "__main__":
    main()
