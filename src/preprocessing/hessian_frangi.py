"""Multi-scale Frangi vesselness filtering for 3D tubular structures."""
import numpy as np
from scipy import ndimage
from skimage.filters import frangi


def vesselness(volume: np.ndarray, sigmas=(0.5, 1.0, 1.5, 2.0), black_ridges=False) -> np.ndarray:
    """Compute a Frangi vesselness map for a 3D CT/MR volume.

    Args:
        volume: (D, H, W) float array, intensity-normalized.
        sigmas: scales to probe, in voxels — should roughly span expected vessel radii.
        black_ridges: True if vessels are darker than background.

    Returns:
        (D, H, W) float array in [0, 1], high where tubular structure is likely.
    """
    return frangi(volume, sigmas=sigmas, black_ridges=black_ridges)


def threshold_vesselness(vesselness_map: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    """Binarize a vesselness map by percentile threshold. Simple baseline;
    swap for a learned/adaptive threshold if this proves too coarse."""
    thresh = np.percentile(vesselness_map, percentile)
    return (vesselness_map >= thresh).astype(np.uint8)


def keep_largest_components(mask: np.ndarray, n: int = 3) -> np.ndarray:
    """Keeps only the `n` largest 26-connected components of a binary mask.

    Percentile-thresholding a whole-volume Frangi map picks up scattered
    non-vessel structure (rib edges, noise) as many small components
    alongside the real, larger, connected vessel tree — label-free, so
    this doesn't touch ground truth. Measured on one ImageCAS case: after
    a fixed ROI crop + threshold_vesselness, n=3 roughly doubles Dice
    against the ground-truth mask versus no filtering (0.156 -> 0.229);
    n=1 alone was worse than no filtering at all (0.0) — the single
    largest component was not the coronary tree, so don't assume n=1 is
    ever the safe default. Not re-validated across cases; revisit before
    scaling past the pilot.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    labeled, n_components = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n_components == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
    top_labels = np.argsort(sizes)[::-1][:n] + 1
    return np.isin(labeled, top_labels).astype(np.uint8)
