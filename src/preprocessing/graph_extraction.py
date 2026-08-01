"""Convert a skeleton voxel mask into a topology graph.

Nodes = endpoints and bifurcations (degree != 2 in the 26-connected skeleton).
Edges = centerline segments connecting adjacent nodes, carrying geometric
features (length, mean radius, direction) used as edge attributes for the
graph encoder.
"""
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy import ndimage

_NEIGHBOR_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


@dataclass
class VesselGraph:
    graph: nx.Graph
    node_coords: dict = field(default_factory=dict)  # node_id -> (z, y, x)


def _neighbors(coord, skeleton_set):
    z, y, x = coord
    return [
        (z + dz, y + dy, x + dx)
        for dz, dy, dx in _NEIGHBOR_OFFSETS
        if (z + dz, y + dy, x + dx) in skeleton_set
    ]


def _voxel_degrees(skeleton_coords, skeleton_set):
    return {c: len(_neighbors(c, skeleton_set)) for c in skeleton_coords}


def build_vessel_graph(skeleton: np.ndarray, radius_map: np.ndarray | None = None) -> VesselGraph:
    """Trace a skeleton mask into a bifurcation graph.

    Args:
        skeleton: (D, H, W) bool array from `skeletonize_mask`.
        radius_map: optional (D, H, W) float array (e.g. distance transform
            of the original mask) used to attach a per-edge mean radius.

    Returns:
        VesselGraph with node ids -> voxel coords and edges carrying
        `length`, `mean_radius`, `direction` (unit vector, node_a -> node_b).
    """
    skeleton_coords = set(map(tuple, np.argwhere(skeleton)))
    if radius_map is None:
        radius_map = ndimage.distance_transform_edt(skeleton if skeleton.dtype == bool else skeleton.astype(bool))

    degrees = _voxel_degrees(skeleton_coords, skeleton_set=skeleton_coords)
    critical = {c for c, d in degrees.items() if d != 2}  # endpoints (d<=1) + bifurcations (d>=3)

    g = nx.Graph()
    node_coords = {}
    coord_to_node = {}
    for i, c in enumerate(sorted(critical)):
        node_coords[i] = c
        coord_to_node[c] = i
        g.add_node(i, coord=c, degree=degrees[c])

    visited_edges = set()
    for start_coord in critical:
        for nbr in _neighbors(start_coord, skeleton_coords):
            if (start_coord, nbr) in visited_edges or nbr in critical and (nbr, start_coord) in visited_edges:
                continue
            # walk the path from start_coord through nbr until hitting another critical voxel
            path = [start_coord, nbr]
            prev, cur = start_coord, nbr
            while cur not in critical:
                nxt_candidates = [n for n in _neighbors(cur, skeleton_coords) if n != prev]
                if not nxt_candidates:
                    break
                nxt = nxt_candidates[0]
                path.append(nxt)
                prev, cur = cur, nxt
            if cur in critical and cur != start_coord:
                a, b = coord_to_node[start_coord], coord_to_node[cur]
                visited_edges.add((start_coord, path[1]))
                if not g.has_edge(a, b):
                    coords_arr = np.array(path, dtype=float)
                    length = float(np.linalg.norm(np.diff(coords_arr, axis=0), axis=1).sum())
                    radii = [radius_map[tuple(p)] for p in path]
                    direction = coords_arr[-1] - coords_arr[0]
                    norm = np.linalg.norm(direction)
                    direction = direction / norm if norm > 0 else direction
                    g.add_edge(
                        a, b,
                        length=length,
                        mean_radius=float(np.mean(radii)),
                        direction=direction.tolist(),
                        path=path,
                    )

    return VesselGraph(graph=g, node_coords=node_coords)


def graph_to_pyg_features(vg: VesselGraph):
    """Extract (node_features, edge_index, edge_features) arrays for a
    PyTorch Geometric Data object. Kept separate from `build_vessel_graph`
    so the graph-building logic stays torch-free."""
    g = vg.graph
    nodes = sorted(g.nodes)
    node_feats = np.array([[g.nodes[n]["degree"], *vg.node_coords[n]] for n in nodes], dtype=np.float32)

    edge_index = []
    edge_feats = []
    for u, v, data in g.edges(data=True):
        for a, b in ((u, v), (v, u)):  # undirected -> both directions for PyG
            edge_index.append([a, b])
            edge_feats.append([data["length"], data["mean_radius"], *data["direction"]])

    edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
    edge_feats = np.array(edge_feats, dtype=np.float32) if edge_feats else np.zeros((0, 5), dtype=np.float32)
    return node_feats, edge_index, edge_feats
