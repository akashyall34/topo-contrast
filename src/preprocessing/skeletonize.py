"""3D thinning of a binary vessel mask into a 1-voxel-wide centerline skeleton."""
import numpy as np
from skimage.morphology import skeletonize_3d


def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Reduce a binary tubular mask to its centerline skeleton.

    Args:
        binary_mask: (D, H, W) uint8/bool array, foreground = vessel.

    Returns:
        (D, H, W) bool array, True on skeleton voxels.
    """
    return skeletonize_3d(binary_mask.astype(bool))
