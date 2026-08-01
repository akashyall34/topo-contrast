"""Download/preprocess ImageCAS volumes into data/processed/<subset>/<case_id>/.

For each case, writes:
    image.npy      (D, H, W) float32, intensity-normalized
    skeleton.npy   (D, H, W) bool, vessel centerline
    radius.npy     (D, H, W) float32, distance transform of the vessel mask

Fill in `_download_case` / `_load_raw_case` for your actual ImageCAS access
(local NAS copy, S3 bucket, etc.) — this is left unimplemented since data
access is environment-specific.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from src.preprocessing.hessian_frangi import threshold_vesselness, vesselness
from src.preprocessing.skeletonize import skeletonize_mask


def _load_raw_case(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (image, vessel_mask) as (D, H, W) float32 / bool arrays.

    Replace with actual ImageCAS loading (e.g. nibabel/SimpleITK read from
    data/raw/<case_id>/) — left as a stub since raw data is not bundled.
    """
    raise NotImplementedError(
        f"Implement raw-volume loading for case {case_id}: read from data/raw/, "
        "intensity-normalize, and return (image, vessel_mask)."
    )


def process_case(case_id: str, out_dir: Path):
    image, vessel_mask = _load_raw_case(case_id)

    if vessel_mask is None:
        vmap = vesselness(image)
        vessel_mask = threshold_vesselness(vmap)

    skeleton = skeletonize_mask(vessel_mask)
    radius = ndimage.distance_transform_edt(vessel_mask)

    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "image.npy", image.astype(np.float32))
    np.save(case_dir / "skeleton.npy", skeleton.astype(bool))
    np.save(case_dir / "radius.npy", radius.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--case-ids-file", type=str, default=None,
                         help="Optional text file, one case id per line. "
                              "Overrides --n-cases if given.")
    args = parser.parse_args()

    out_dir = Path("data/processed") / args.subset

    if args.case_ids_file:
        case_ids = Path(args.case_ids_file).read_text().split()
    else:
        raw_dir = Path("data/raw")
        case_ids = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())[: args.n_cases]

    for case_id in case_ids:
        print(f"processing {case_id}")
        process_case(case_id, out_dir)


if __name__ == "__main__":
    main()
