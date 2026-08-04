"""3D thinning of a binary vessel mask into a 1-voxel-wide centerline skeleton."""
import numpy as np
from skimage.morphology import skeletonize


def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Reduce a binary tubular mask to its centerline skeleton.

    Args:
        binary_mask: (D, H, W) uint8/bool array, foreground = vessel.

    Returns:
        (D, H, W) bool array, True on skeleton voxels.
    """
    # skimage >= 0.19 merged the separate skeletonize_3d into skeletonize
    # (n-dimensional); skeletonize_3d was removed entirely in 0.25.
    return skeletonize(binary_mask.astype(bool))
