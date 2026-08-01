"""Pairs each vessel-graph node with a 3D image patch centered on its voxel
coordinate, for image<->graph contrastive pretraining.

Expects preprocessed volumes under `data/processed/<case_id>/`:
    image.npy       (D, H, W) float32, intensity-normalized CT volume
    skeleton.npy    (D, H, W) bool, output of skeletonize_mask
    radius.npy      (D, H, W) float32, distance transform (optional)
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.graph_extraction import build_vessel_graph, graph_to_pyg_features


class ContrastiveVesselDataset(Dataset):
    def __init__(self, case_dir: Path, patch_size: int = 32):
        self.case_dir = Path(case_dir)
        self.patch_size = patch_size
        self.cases = sorted(p for p in self.case_dir.iterdir() if p.is_dir())

    def __len__(self):
        return len(self.cases)

    def _extract_patch(self, image: np.ndarray, coord: tuple[int, int, int]) -> np.ndarray:
        r = self.patch_size // 2
        z, y, x = coord
        padded = np.pad(image, r, mode="constant", constant_values=0)
        z, y, x = z + r, y + r, x + r  # shift for padding
        patch = padded[z - r:z + r, y - r:y + r, x - r:x + r]
        return patch

    def __getitem__(self, idx):
        case = self.cases[idx]
        image = np.load(case / "image.npy")
        skeleton = np.load(case / "skeleton.npy")
        radius = np.load(case / "radius.npy") if (case / "radius.npy").exists() else None

        vg = build_vessel_graph(skeleton, radius_map=radius)
        node_feats, edge_index, edge_feats = graph_to_pyg_features(vg)

        patches = np.stack([
            self._extract_patch(image, vg.node_coords[n]) for n in sorted(vg.node_coords)
        ])[:, None, ...]  # (N, 1, d, h, w)

        return {
            "case_id": case.name,
            "patches": torch.from_numpy(patches).float(),
            "node_feats": torch.from_numpy(node_feats).float(),
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_feats": torch.from_numpy(edge_feats).float(),
            "node_coords": vg.node_coords,
        }


def collate_single_case(batch):
    """Pilot dataloader uses batch_size=1 at the case level (graphs vary in
    size); this collate just unwraps the singleton list."""
    assert len(batch) == 1, "pilot dataset expects batch_size=1 (one case = one variable-size graph)"
    return batch[0]
