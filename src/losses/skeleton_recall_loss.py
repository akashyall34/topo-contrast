"""Skeleton recall loss for segmentation fine-tuning (not used in the pilot).

Placeholder for the full-run fine-tuning stage: penalizes predicted masks
that fail to cover the ground-truth centerline skeleton, complementing
Dice/CE with a topology-sensitive term. Fill in once the pilot gate passes
and fine-tuning work starts (see docs/project-proposal.md).
"""
import torch


def skeleton_recall_loss(*args, **kwargs) -> torch.Tensor:
    raise NotImplementedError("Implement after the pilot gate passes — see docs/project-proposal.md")
