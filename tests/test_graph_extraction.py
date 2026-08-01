import numpy as np

from src.preprocessing.graph_extraction import build_vessel_graph, graph_to_pyg_features


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
