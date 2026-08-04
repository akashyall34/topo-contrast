"""DEPRECATED — superseded design, kept for reference only, not wired into
any current entrypoint. See docs/PROJECT.md Section 1: aligning an image
patch against the graph node extracted from that same patch/volume injects
no information the image encoder couldn't already get via reconstruction,
since the graph is a deterministic function of the same intensity data.
Current pretraining uses src/datasets/connectivity_dataset.py instead.

Pairs each vessel-graph node with a 3D image patch centered on its voxel
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

from src.datasets.patch_utils import extract_patch
from src.preprocessing.graph_extraction import build_vessel_graph, graph_to_pyg_features


class ContrastiveVesselDataset(Dataset):
    def __init__(self, case_dir: Path, patch_size: int = 32):
        self.case_dir = Path(case_dir)
        self.patch_size = patch_size
        self.cases = sorted(p for p in self.case_dir.iterdir() if p.is_dir())

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        image = np.load(case / "image.npy")
        skeleton = np.load(case / "skeleton.npy")
        radius = np.load(case / "radius.npy") if (case / "radius.npy").exists() else None

        vg = build_vessel_graph(skeleton, radius_map=radius)
        node_feats, edge_index, edge_feats = graph_to_pyg_features(vg)

        patches = np.stack([
            extract_patch(image, vg.node_coords[n], self.patch_size) for n in sorted(vg.node_coords)
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
