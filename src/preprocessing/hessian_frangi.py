"""Multi-scale Frangi vesselness filtering for 3D tubular structures."""
import numpy as np
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
