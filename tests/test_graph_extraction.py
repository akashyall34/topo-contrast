import math

import networkx as nx
import numpy as np
import pytest

from src.preprocessing.graph_extraction import (
    DISCONNECTED_OR_FAR,
    NEAR_ANCESTOR,
    SAME_BRANCH,
    build_arclength_index,
    build_vessel_graph,
    connectivity_label,
    geodesic_distance,
    graph_to_pyg_features,
)


def _line_skeleton(length: int = 10) -> np.ndarray:
    """A straight 1-voxel line: 2 endpoints, 0 bifurcations, 1 edge."""
    skel = np.zeros((1, 1, length), dtype=bool)
    skel[0, 0, :] = True
    return skel


def test_straight_line_has_two_endpoints_one_edge():
    skel = _line_skeleton(10)
    vg = build_vessel_graph(skel)
    degrees = [d for _, d in vg.graph.degree()]
    assert sorted(degrees) == [1, 1]
    assert vg.graph.number_of_edges() == 1


def test_edge_features_present():
    skel = _line_skeleton(10)
    vg = build_vessel_graph(skel)
    u, v, data = next(iter(vg.graph.edges(data=True)))
    assert data["length"] > 0
    assert "direction" in data
    assert "mean_radius" in data


def test_graph_to_pyg_features_shapes():
    skel = _line_skeleton(10)
    vg = build_vessel_graph(skel)
    node_feats, edge_index, edge_feats = graph_to_pyg_features(vg)
    n_nodes = vg.graph.number_of_nodes()
    n_edges_directed = vg.graph.number_of_edges() * 2
    assert node_feats.shape == (n_nodes, 4)
    assert edge_index.shape == (2, n_edges_directed)
    assert edge_feats.shape == (n_edges_directed, 5)


# --- Y-tree fixture: bifurcation + geodesic distance / connectivity label ---

def test_y_tree_has_one_bifurcation_and_three_endpoints(y_skeleton):
    vg = build_vessel_graph(y_skeleton.skeleton)
    # Restrict to the Y-tree's own connected component — the fixture also
    # contains a disjoint 4-voxel line (its own 2 endpoints + 2 degree-2
    # nodes) and an isolated degree-0 point, which would otherwise pollute
    # the degree distribution being asserted here.
    b_node = _node_id_for_coord(vg, y_skeleton.bifurcation)
    y_nodes = next(c for c in nx.connected_components(vg.graph) if b_node in c)
    degrees = sorted(vg.graph.degree(n) for n in y_nodes)
    assert degrees.count(3) == 1  # the bifurcation
    assert degrees.count(1) == 3  # three arm tips
    assert vg.graph.subgraph(y_nodes).number_of_edges() == 3


def _node_id_for_coord(vg, coord):
    for node_id, c in vg.node_coords.items():
        if tuple(c) == tuple(coord):
            return node_id
    raise AssertionError(f"no node at {coord}")


def test_arclength_index_monotonic_along_arm(y_skeleton):
    vg = build_vessel_graph(y_skeleton.skeleton)
    idx = build_arclength_index(vg)
    arm1_points = y_skeleton.arm_points["arm1"]  # step 1..5 outward from B, axis-aligned unit steps
    arclens = [idx.point_arclength_from_u[tuple(p)] for p in arm1_points]
    diffs = np.diff(arclens)
    # axis-aligned unit steps -> each consecutive pair is exactly 1.0 apart,
    # monotonic in whichever direction arc length increases from `u`.
    assert np.allclose(np.abs(diffs), 1.0)
    assert np.all(diffs > 0) or np.all(diffs < 0)


def test_geodesic_distance_same_edge_matches_euclidean(y_skeleton):
    vg = build_vessel_graph(y_skeleton.skeleton)
    idx = build_arclength_index(vg)
    arm1_points = y_skeleton.arm_points["arm1"]
    p_a, p_b = arm1_points[0], arm1_points[3]  # steps 1 and 4 -> both on the same edge
    arc_dist, hop_dist = geodesic_distance(vg, idx, p_a, p_b)
    assert hop_dist == 0
    assert math.isclose(arc_dist, 3.0, rel_tol=1e-6)  # |4 - 1| unit steps


def test_geodesic_distance_across_bifurcation(y_skeleton):
    vg = build_vessel_graph(y_skeleton.skeleton)
    idx = build_arclength_index(vg)
    arm1_points = y_skeleton.arm_points["arm1"]
    arm2_points = y_skeleton.arm_points["arm2"]
    p_a = arm1_points[1]  # step 2 on arm1, distance 2.0 from B
    p_b = arm2_points[2]  # step 3 on arm2, distance 3*sqrt(2) from B
    arc_dist, hop_dist = geodesic_distance(vg, idx, p_a, p_b)
    assert hop_dist == 1
    assert math.isclose(arc_dist, 2.0 + 3 * math.sqrt(2), rel_tol=1e-6)


def test_geodesic_distance_disconnected_components(y_skeleton):
    vg = build_vessel_graph(y_skeleton.skeleton)
    idx = build_arclength_index(vg)
    arm1_points = y_skeleton.arm_points["arm1"]
    arc_dist, hop_dist = geodesic_distance(vg, idx, arm1_points[0], y_skeleton.disjoint_segment[0])
    assert arc_dist is None
    assert hop_dist is None


def test_geodesic_distance_isolated_point_is_disconnected(y_skeleton):
    vg = build_vessel_graph(y_skeleton.skeleton)
    idx = build_arclength_index(vg)
    arm1_points = y_skeleton.arm_points["arm1"]
    arc_dist, hop_dist = geodesic_distance(vg, idx, arm1_points[0], y_skeleton.isolated_point)
    assert arc_dist is None
    assert hop_dist is None


def test_geodesic_distance_rejects_off_skeleton_coord(y_skeleton):
    vg = build_vessel_graph(y_skeleton.skeleton)
    idx = build_arclength_index(vg)
    arm1_points = y_skeleton.arm_points["arm1"]
    off_skeleton_coord = (0, 0, 1)  # not on the skeleton at all
    with pytest.raises(ValueError):
        geodesic_distance(vg, idx, arm1_points[0], off_skeleton_coord)


@pytest.mark.parametrize(
    "hop_distance, hop_k, expected",
    [
        (0, 0, SAME_BRANCH),
        (0, 2, SAME_BRANCH),
        (1, 1, NEAR_ANCESTOR),
        (1, 0, DISCONNECTED_OR_FAR),  # boundary: hop_distance > hop_k
        (2, 2, NEAR_ANCESTOR),        # boundary: hop_distance == hop_k
        (None, 5, DISCONNECTED_OR_FAR),
    ],
)
def test_connectivity_label_boundaries(hop_distance, hop_k, expected):
    assert connectivity_label(hop_distance, hop_k) == expected
