"""Standard 3D image/mask pair dataset for fine-tuning, with an optional
`fraction` to subsample cases for data-efficiency sweeps."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, case_dir: Path, fraction: float = 1.0, seed: int = 0):
        self.case_dir = Path(case_dir)
        cases = sorted(p for p in self.case_dir.iterdir() if p.is_dir())
        if fraction < 1.0:
            rng = random.Random(seed)
            k = max(1, int(len(cases) * fraction))
            cases = rng.sample(cases, k)
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        image = np.load(case / "image.npy")[None, ...]  # (1, D, H, W)
        mask = np.load(case / "mask.npy")[None, ...]
        return {
            "case_id": case.name,
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float(),
        }
