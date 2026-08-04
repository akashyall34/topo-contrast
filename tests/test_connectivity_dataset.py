import numpy as np
import pytest

from src.datasets.connectivity_dataset import (
    ConnectivityPairDataset,
    ConnectivityPairSampler,
    InsufficientPairsError,
    SamplerConfig,
    load_case_graph,
)
from src.datasets.patch_utils import extract_patch
from src.preprocessing.graph_extraction import build_arclength_index, build_vessel_graph

# A short line offset from arm1 by 3 voxels in y: close in Euclidean space
# (distance 3) but a separate skeleton component (not 26-adjacent, since the
# y-offset of 3 exceeds the 26-connectivity radius of 1) -> the intended
# "hard negative" case (spatially close, topologically disconnected).
_PARALLEL_OFFSET_Y = 3


def _add_parallel_segment(skeleton: np.ndarray, y_skeleton) -> np.ndarray:
    skel = skeleton.copy()
    for z, y, x in y_skeleton.arm_points["arm1"][:3]:
        skel[z, y + _PARALLEL_OFFSET_Y, x] = True
    return skel


@pytest.fixture
def feasible_config():
    return SamplerConfig(
        crop_size=1,
        near_radius=4.0,
        far_radius=5.0,
        hop_k=1,
        pairs_per_pool=20,
        min_pool_size=5,
        max_attempts_multiplier=400,
    )


def test_build_pools_succeeds_with_feasible_radii(y_skeleton, feasible_config):
    skeleton = _add_parallel_segment(y_skeleton.skeleton, y_skeleton)
    vg = build_vessel_graph(skeleton)
    idx = build_arclength_index(vg)
    sampler = ConnectivityPairSampler(vg, idx, feasible_config, rng=np.random.default_rng(0))

    pools = sampler.build_pools()

    assert len(pools["random"]) >= feasible_config.min_pool_size
    assert len(pools["hard_negative"]) >= feasible_config.min_pool_size
    assert len(pools["hard_positive"]) >= feasible_config.min_pool_size

    from src.preprocessing.graph_extraction import DISCONNECTED_OR_FAR, NEAR_ANCESTOR, SAME_BRANCH
    for _, _, label in pools["hard_negative"]:
        assert label == DISCONNECTED_OR_FAR
    for _, _, label in pools["hard_positive"]:
        assert label in (SAME_BRANCH, NEAR_ANCESTOR)


def test_build_pools_raises_when_radii_infeasible(y_skeleton, feasible_config):
    """far_radius set far beyond anything reachable in this tiny volume ->
    the hard_positive pool can never be satisfied; proves the failure path
    is real, not just present in the code."""
    skeleton = _add_parallel_segment(y_skeleton.skeleton, y_skeleton)
    vg = build_vessel_graph(skeleton)
    idx = build_arclength_index(vg)
    infeasible_config = SamplerConfig(
        crop_size=feasible_config.crop_size,
        near_radius=feasible_config.near_radius,
        far_radius=1000.0,  # unreachable in a 21^3 volume
        hop_k=feasible_config.hop_k,
        pairs_per_pool=feasible_config.pairs_per_pool,
        min_pool_size=feasible_config.min_pool_size,
        max_attempts_multiplier=feasible_config.max_attempts_multiplier,
    )
    sampler = ConnectivityPairSampler(vg, idx, infeasible_config, rng=np.random.default_rng(0))

    with pytest.raises(InsufficientPairsError):
        sampler.build_pools()


def test_sampler_config_rejects_radii_not_exceeding_crop_size():
    with pytest.raises(ValueError):
        SamplerConfig(crop_size=32, near_radius=32.0, far_radius=150.0)
    with pytest.raises(ValueError):
        SamplerConfig(crop_size=32, near_radius=40.0, far_radius=32.0)


def test_load_case_graph_does_not_require_image(tmp_path, y_skeleton):
    case_dir = tmp_path / "case_0"
    case_dir.mkdir()
    np.save(case_dir / "skeleton.npy", y_skeleton.skeleton)
    # deliberately no image.npy written — load_case_graph must not need it
    vg, idx = load_case_graph(case_dir)
    assert vg.graph.number_of_nodes() > 0
    assert len(idx.point_to_edge) > 0


def test_connectivity_pair_dataset_getitem_shapes(tmp_path, y_skeleton, feasible_config):
    skeleton = _add_parallel_segment(y_skeleton.skeleton, y_skeleton)
    case_dir = tmp_path / "case_0"
    case_dir.mkdir()
    np.save(case_dir / "skeleton.npy", skeleton)
    rng = np.random.default_rng(0)
    image = rng.random(skeleton.shape).astype(np.float32)
    np.save(case_dir / "image.npy", image)

    dataset = ConnectivityPairDataset(tmp_path, feasible_config, rng=np.random.default_rng(1))
    assert len(dataset) > 0

    item = dataset[0]
    c = feasible_config.crop_size
    assert item["patch_a"].shape == (1, c, c, c)
    assert item["patch_b"].shape == (1, c, c, c)
    assert item["label"] in (0, 1, 2)
    assert item["euclidean_distance"] >= c
    assert item["case_id"] == "case_0"


def test_image_cache_respects_max_cache_size(tmp_path, y_skeleton, feasible_config):
    """With max_cache_size < number of cases, the cache must never grow
    past that bound — this is what protects DataLoader worker processes
    (each forking their own cache copy) from unbounded memory growth."""
    skeleton = _add_parallel_segment(y_skeleton.skeleton, y_skeleton)
    n_cases = 3
    for i in range(n_cases):
        case_dir = tmp_path / f"case_{i}"
        case_dir.mkdir()
        np.save(case_dir / "skeleton.npy", skeleton)
        image = np.random.default_rng(i).random(skeleton.shape).astype(np.float32)
        np.save(case_dir / "image.npy", image)

    dataset = ConnectivityPairDataset(tmp_path, feasible_config, rng=np.random.default_rng(1),
                                       max_cache_size=2)

    # touch every case_id's entries so all 3 cases get loaded at some point
    case_ids_seen = set()
    for idx in range(len(dataset)):
        item = dataset[idx]
        case_ids_seen.add(item["case_id"])
        assert len(dataset._image_cache) <= 2
    assert case_ids_seen == {"case_0", "case_1", "case_2"}


def test_image_cache_correctness_survives_eviction(tmp_path, y_skeleton, feasible_config):
    """Cache eviction must not corrupt results — re-fetching an evicted
    case's image should reload it correctly, not return stale/wrong data."""
    skeleton = _add_parallel_segment(y_skeleton.skeleton, y_skeleton)
    images = {}
    for i in range(3):
        case_dir = tmp_path / f"case_{i}"
        case_dir.mkdir()
        np.save(case_dir / "skeleton.npy", skeleton)
        image = np.random.default_rng(i).random(skeleton.shape).astype(np.float32)
        images[f"case_{i}"] = image
        np.save(case_dir / "image.npy", image)

    dataset = ConnectivityPairDataset(tmp_path, feasible_config, rng=np.random.default_rng(1),
                                       max_cache_size=1)  # forces eviction on almost every access

    for idx in range(len(dataset)):
        case, coord_a, coord_b, label = dataset._entries[idx]
        item = dataset[idx]
        expected_patch_a = extract_patch(images[case.name], coord_a, feasible_config.crop_size)
        assert np.allclose(item["patch_a"].squeeze(0).numpy(), expected_patch_a)
