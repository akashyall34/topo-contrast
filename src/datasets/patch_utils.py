"""Shared 3D patch-extraction helper, used by both the superseded
image-node contrastive dataset and the current connectivity-prediction
dataset (docs/PROJECT.md Section 1)."""
import numpy as np


def extract_patch(image: np.ndarray, coord: tuple[int, int, int], patch_size: int) -> np.ndarray:
    """Extract a cubic patch of `patch_size` centered on `coord`, zero-padding
    at volume boundaries.

    Args:
        image: (D, H, W) array.
        coord: (z, y, x) center voxel, in `image`'s original (unpadded)
            coordinate frame.
        patch_size: cube side length. Must be >= 1.

    Returns:
        (patch_size, patch_size, patch_size) array.
    """
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")

    r_lo = patch_size // 2
    r_hi = patch_size - r_lo  # handles odd patch_size without truncating a voxel
    z, y, x = (int(c) for c in coord)
    padded = np.pad(image, r_lo, mode="constant", constant_values=0)
    z, y, x = z + r_lo, y + r_lo, x + r_lo  # shift for padding
    return padded[z - r_lo:z + r_hi, y - r_lo:y + r_hi, x - r_lo:x + r_hi]
