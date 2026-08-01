"""Segmentation + topology metrics for fine-tuning evaluation."""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize_3d


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


def cl_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Centerline Dice (Shit et al., 2021): measures topological correctness
    by checking mutual skeleton coverage rather than raw voxel overlap."""
    pred, target = pred.astype(bool), target.astype(bool)
    pred_skel = skeletonize_3d(pred)
    target_skel = skeletonize_3d(target)

    tprec = _masked_skeleton_overlap(pred_skel, target)
    tsens = _masked_skeleton_overlap(target_skel, pred)
    denom = tprec + tsens
    return float(2 * tprec * tsens / denom) if denom > 0 else 1.0


def _masked_skeleton_overlap(skeleton: np.ndarray, mask: np.ndarray) -> float:
    total = skeleton.sum()
    return float(np.logical_and(skeleton, mask).sum() / total) if total > 0 else 0.0


def betti_number_error(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compares connected-component count (b0) between prediction and
    target as a coarse topology-correctness signal. A full b0/b1/b2
    computation needs a proper homology library (e.g. gudhi); this is the
    cheap b0-only version sufficient for the pilot/eval loop."""
    pred_labels, pred_b0 = ndimage.label(pred.astype(bool))
    target_labels, target_b0 = ndimage.label(target.astype(bool))
    return {"b0_pred": pred_b0, "b0_target": target_b0, "b0_error": abs(pred_b0 - target_b0)}


def normalized_surface_distance(pred: np.ndarray, target: np.ndarray, tolerance_mm: float = 1.0,
                                 spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Fraction of predicted/target surface voxels within `tolerance_mm` of
    each other's surface. Requires voxel spacing for physical distances."""
    pred, target = pred.astype(bool), target.astype(bool)
    pred_surface = pred ^ ndimage.binary_erosion(pred)
    target_surface = target ^ ndimage.binary_erosion(target)

    dt_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    pred_within = (dt_target[pred_surface] <= tolerance_mm).sum()
    target_within = (dt_pred[target_surface] <= tolerance_mm).sum()

    denom = pred_surface.sum() + target_surface.sum()
    return float((pred_within + target_within) / denom) if denom > 0 else 1.0
