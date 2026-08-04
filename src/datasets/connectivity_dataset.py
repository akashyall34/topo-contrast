"""Samples pairs of image crops from the same volume and labels them with
their topological connectivity relationship (docs/PROJECT.md Section 2).
The vessel graph is used only to generate this label — it is not consumed
as a parallel modality or fed to a graph encoder anywhere downstream.

Expects preprocessed volumes under `data/processed/<case_id>/`:
    image.npy       (D, H, W) float32, intensity-normalized CT volume
    skeleton.npy    (D, H, W) bool, output of skeletonize_mask
    radius.npy      (D, H, W) float32, distance transform (optional)

`ConnectivityPairSampler.build_pools` only needs `skeleton.npy` (via the
graph it induces) — `image.npy` is loaded lazily, only when a patch is
actually extracted, so `scripts/check_sampling_feasibility.py` (pilot-gate
step 1) never has to load full CT volumes just to check pool sizes.
"""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from src.datasets.patch_utils import extract_patch
from src.preprocessing.graph_extraction import (
    DISCONNECTED_OR_FAR,
    NEAR_ANCESTOR,
    SAME_BRANCH,
    ArcLengthIndex,
    VesselGraph,
    build_arclength_index,
    build_vessel_graph,
    connectivity_label,
    geodesic_distance,
)

_CONNECTED_LABELS = (SAME_BRANCH, NEAR_ANCESTOR)
LABEL_TO_ID = {SAME_BRANCH: 0, NEAR_ANCESTOR: 1, DISCONNECTED_OR_FAR: 2}


class InsufficientPairsError(RuntimeError):
    """Raised when a pool (random/hard_negative/hard_positive) can't reach
    its configured minimum size within the sampling budget. This is the
    exact failure pilot-gate step 1 (docs/PROJECT.md Section 8) exists to
    surface — it must propagate, not be papered over with a smaller or
    empty pool."""


@dataclass
class SamplerConfig:
    crop_size: int = 32
    near_radius: float = 40.0   # Euclidean threshold (voxels) for hard negatives: d < near_radius
    far_radius: float = 150.0   # Euclidean threshold (voxels) for hard positives: d > far_radius
    hop_k: int = 2
    pairs_per_pool: int = 200
    min_pool_size: int = 20
    max_attempts_multiplier: int = 50  # sampling budget per pool = pairs_per_pool * this

    def __post_init__(self):
        if self.crop_size < 1:
            raise ValueError(f"crop_size must be >= 1, got {self.crop_size}")
        if self.near_radius <= self.crop_size:
            raise ValueError(
                f"near_radius ({self.near_radius}) must exceed crop_size "
                f"({self.crop_size}) or the hard_negative pool is unsatisfiable "
                f"by construction (all pools require d >= crop_size to avoid "
                f"overlapping crops)"
            )
        if self.far_radius <= self.crop_size:
            raise ValueError(f"far_radius ({self.far_radius}) must exceed crop_size ({self.crop_size})")
        if self.pairs_per_pool < 1 or self.min_pool_size < 1:
            raise ValueError("pairs_per_pool and min_pool_size must be >= 1")
        if self.min_pool_size > self.pairs_per_pool:
            raise ValueError("min_pool_size cannot exceed pairs_per_pool")
        if self.hop_k < 0:
            raise ValueError(f"hop_k must be >= 0, got {self.hop_k}")


class ConnectivityPairSampler:
    """Builds labeled crop-pair pools for one preprocessed case, using only
    the vessel graph (no image content needed)."""

    def __init__(self, vg: VesselGraph, arclength_index: ArcLengthIndex,
                 config: SamplerConfig, rng: np.random.Generator | None = None):
        self.vg = vg
        self.arclength_index = arclength_index
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

        self._candidates = np.array(list(arclength_index.point_to_edge.keys()), dtype=np.int64)
        if len(self._candidates) < 2:
            raise InsufficientPairsError(
                f"case has only {len(self._candidates)} indexed skeleton voxel(s); "
                f"need at least 2 to sample any pair"
            )
        # Used by _build_hard_negative_pool to query spatial neighbors
        # directly instead of hoping uniform random pairs happen to land
        # close together — see that method's docstring.
        self._kdtree = cKDTree(self._candidates)

    def _sample_candidate_pair(self):
        i, j = self.rng.integers(0, len(self._candidates), size=2)
        return tuple(self._candidates[i]), tuple(self._candidates[j])

    @staticmethod
    def _euclidean(coord_a, coord_b) -> float:
        return float(np.linalg.norm(np.asarray(coord_a, dtype=float) - np.asarray(coord_b, dtype=float)))

    def _build_pool(self, accept_fn) -> list[tuple[tuple, tuple, str]]:
        """accept_fn(coord_a, coord_b, euclid_dist, label) -> bool. Raises
        InsufficientPairsError if the pool can't reach `min_pool_size`
        within the sampling budget."""
        cfg = self.config
        pool = []
        max_attempts = cfg.pairs_per_pool * cfg.max_attempts_multiplier
        attempts = 0
        while len(pool) < cfg.pairs_per_pool and attempts < max_attempts:
            attempts += 1
            coord_a, coord_b = self._sample_candidate_pair()
            if coord_a == coord_b:
                continue
            euclid_dist = self._euclidean(coord_a, coord_b)
            if euclid_dist < cfg.crop_size:
                continue  # crops would overlap / sit within each other's receptive field
            _, hop_dist = geodesic_distance(self.vg, self.arclength_index, coord_a, coord_b)
            label = connectivity_label(hop_dist, cfg.hop_k)
            if accept_fn(coord_a, coord_b, euclid_dist, label):
                pool.append((coord_a, coord_b, label))
        if len(pool) < cfg.min_pool_size:
            raise InsufficientPairsError(
                f"only found {len(pool)}/{cfg.min_pool_size} required pairs in "
                f"{attempts} attempts (pairs_per_pool={cfg.pairs_per_pool}, "
                f"near_radius={cfg.near_radius}, far_radius={cfg.far_radius}, "
                f"hop_k={cfg.hop_k}) — crop size/spacing or radii are not "
                f"feasible for this case's vessel tree"
            )
        return pool

    def _build_hard_negative_pool(self) -> list[tuple[tuple, tuple, str]]:
        """Structured version of `_build_pool` for hard negatives: queries
        the KD-tree for candidates within [crop_size, near_radius) of a
        sampled point directly, instead of relying on uniform-random pair
        sampling to land in that narrow distance band by chance.

        This satisfies the distance constraint by construction, leaving
        only the DISCONNECTED_OR_FAR label check as a real rejection
        condition. Uniform rejection sampling here was the dominant cost
        of dataset construction in practice (one real training run's
        one-time setup took 60+ minutes) since the near-radius band is a
        small fraction of the full coordinate space regardless of how
        sparse or plentiful genuine hard negatives are.
        """
        cfg = self.config
        pool = []
        max_attempts = cfg.pairs_per_pool * cfg.max_attempts_multiplier
        attempts = 0
        n = len(self._candidates)
        while len(pool) < cfg.pairs_per_pool and attempts < max_attempts:
            attempts += 1
            i = int(self.rng.integers(0, n))
            coord_a = tuple(self._candidates[i])
            # query_ball_point is inclusive (d <= near_radius); the pool's
            # contract is strict (d < near_radius), so filter the boundary
            # case explicitly rather than silently admitting points at
            # exactly d == near_radius.
            neighbor_idxs = self._kdtree.query_ball_point(coord_a, cfg.near_radius)
            valid = [
                j for j in neighbor_idxs
                if j != i
                and cfg.crop_size <= self._euclidean(coord_a, tuple(self._candidates[j])) < cfg.near_radius
            ]
            if not valid:
                continue
            j = valid[int(self.rng.integers(0, len(valid)))]
            coord_b = tuple(self._candidates[j])
            _, hop_dist = geodesic_distance(self.vg, self.arclength_index, coord_a, coord_b)
            label = connectivity_label(hop_dist, cfg.hop_k)
            if label == DISCONNECTED_OR_FAR:
                pool.append((coord_a, coord_b, label))
        if len(pool) < cfg.min_pool_size:
            raise InsufficientPairsError(
                f"only found {len(pool)}/{cfg.min_pool_size} required pairs in "
                f"{attempts} attempts (pairs_per_pool={cfg.pairs_per_pool}, "
                f"near_radius={cfg.near_radius}, far_radius={cfg.far_radius}, "
                f"hop_k={cfg.hop_k}) — crop size/spacing or radii are not "
                f"feasible for this case's vessel tree"
            )
        return pool

    def build_pools(self) -> dict[str, list[tuple[tuple, tuple, str]]]:
        cfg = self.config
        return {
            "random": self._build_pool(lambda a, b, d, label: True),
            "hard_negative": self._build_hard_negative_pool(),
            "hard_positive": self._build_pool(
                lambda a, b, d, label: d > cfg.far_radius and label in _CONNECTED_LABELS
            ),
        }


def load_case_graph(case_dir: Path) -> tuple[VesselGraph, ArcLengthIndex]:
    """Builds the vessel graph + arc-length index for one case without
    touching image.npy — used by both the dataset and the standalone
    feasibility-check CLI."""
    skeleton = np.load(Path(case_dir) / "skeleton.npy")
    radius_path = Path(case_dir) / "radius.npy"
    radius = np.load(radius_path) if radius_path.exists() else None
    vg = build_vessel_graph(skeleton, radius_map=radius)
    return vg, build_arclength_index(vg)


class ConnectivityPairDataset(Dataset):
    """Flattens the per-case pair pools (all three pools, all cases) into a
    flat, indexable dataset of (patch_a, patch_b, label, euclidean_distance).

    Caches each case's `image.npy` in memory the first time it's touched
    (see `_get_image`) rather than reloading per pair — at pilot scale
    (~30-50 cases, each well under 1GB) this comfortably fits in RAM and
    was necessary in practice: with shuffled batches drawing from many
    cases, reloading full volumes per `__getitem__` call made training
    I/O-bound (observed ~39% CPU utilization, no progress after 5+ minutes
    on the real 38-case pilot set) rather than compute-bound.

    `max_cache_size` bounds the cache to an LRU of that many images
    (`None` = unbounded). This matters more than it looks: with a
    `DataLoader` using `num_workers > 0`, each worker process forks its
    own *separate* copy of this dataset (and thus its own cache) — with an
    unbounded cache and `shuffle=True`, every worker eventually touches
    (and caches) most/all cases over an epoch, multiplying memory by
    `num_workers`. Observed in practice: 4 workers each building a ~2.9GB
    unbounded cache OOM-killed a worker process on a 30-case train split.
    """

    def __init__(self, case_dir: Path, config: SamplerConfig, rng: np.random.Generator | None = None,
                 case_ids: list[str] | None = None, max_cache_size: int | None = None):
        """`case_ids`: restrict to these case directory names only (e.g. a
        train/val split) — default (None) uses every case directory found."""
        self.case_dir = Path(case_dir)
        self.config = config
        if case_ids is not None:
            self.cases = [self.case_dir / c for c in sorted(case_ids)]
            missing = [c for c in self.cases if not c.is_dir()]
            if missing:
                raise FileNotFoundError(f"case directories not found: {missing}")
        else:
            self.cases = sorted(p for p in self.case_dir.iterdir() if p.is_dir())
        self.rng = rng if rng is not None else np.random.default_rng()
        if max_cache_size is not None and max_cache_size < 1:
            raise ValueError(f"max_cache_size must be >= 1 or None, got {max_cache_size}")
        self.max_cache_size = max_cache_size
        self._image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

        self._entries = []  # (case_path, coord_a, coord_b, label)
        for case in self.cases:
            vg, arclength_index = load_case_graph(case)
            sampler = ConnectivityPairSampler(vg, arclength_index, config, rng=self.rng)
            try:
                pools = sampler.build_pools()
            except InsufficientPairsError as e:
                print(f"skipping {case.name} (failed pilot-gate step 1 at this config): {e}")
                continue
            for pool in pools.values():
                for coord_a, coord_b, label in pool:
                    self._entries.append((case, coord_a, coord_b, label))

    def __len__(self):
        return len(self._entries)

    def _get_image(self, case: Path) -> np.ndarray:
        if case in self._image_cache:
            self._image_cache.move_to_end(case)  # mark as most-recently-used
            return self._image_cache[case]
        image = np.load(case / "image.npy")
        self._image_cache[case] = image
        if self.max_cache_size is not None and len(self._image_cache) > self.max_cache_size:
            self._image_cache.popitem(last=False)  # evict least-recently-used
        return image

    def all_labels(self) -> list[int]:
        """Integer label ids for every entry, without touching any image
        data — used to compute class weights for the training loss."""
        return [LABEL_TO_ID[label] for _, _, _, label in self._entries]

    def __getitem__(self, idx):
        case, coord_a, coord_b, label = self._entries[idx]
        image = self._get_image(case)
        patch_a = extract_patch(image, coord_a, self.config.crop_size)
        patch_b = extract_patch(image, coord_b, self.config.crop_size)
        return {
            "case_id": case.name,
            "patch_a": torch.from_numpy(patch_a).float().unsqueeze(0),
            "patch_b": torch.from_numpy(patch_b).float().unsqueeze(0),
            "label": LABEL_TO_ID[label],
            "euclidean_distance": ConnectivityPairSampler._euclidean(coord_a, coord_b),
        }
