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


# --- Geodesic/connectivity utilities -----------------------------------
# Used by src/datasets/connectivity_dataset.py to label pairs of crop
# centers with their topological relationship. The graph is consumed only
# as a label generator here (see docs/PROJECT.md Section 1) — nothing
# below feeds a trained graph encoder.

SAME_BRANCH = "SAME_BRANCH"
NEAR_ANCESTOR = "NEAR_ANCESTOR"
DISCONNECTED_OR_FAR = "DISCONNECTED_OR_FAR"


@dataclass
class ArcLengthIndex:
    # voxel coord -> canonical edge id. Isolated (degree-0) nodes get a
    # sentinel self-edge id (node_id, node_id) so they fold into the same
    # lookup path as real edges instead of needing special-casing below.
    point_to_edge: dict = field(default_factory=dict)
    # voxel coord -> arc length from the edge id's first node (`u` as
    # returned by `graph.edges(data=True)`, not necessarily `path[0]`).
    point_arclength_from_u: dict = field(default_factory=dict)
    edge_length: dict = field(default_factory=dict)  # edge id -> total length


def build_arclength_index(vg: VesselGraph) -> ArcLengthIndex:
    """Index every skeleton voxel on every edge's path by (edge id, arc
    length from that edge's `u` endpoint), so `geodesic_distance` can look
    up an arbitrary crop-center voxel in O(1) instead of re-walking paths.
    """
    idx = ArcLengthIndex()

    for u, v, data in vg.graph.edges(data=True):
        path = data["path"]
        edge_id = (u, v)
        coords_arr = np.array(path, dtype=float)
        seg_lengths = np.linalg.norm(np.diff(coords_arr, axis=0), axis=1)
        cum_from_path_start = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_length = float(cum_from_path_start[-1])
        idx.edge_length[edge_id] = total_length

        u_coord, v_coord = vg.node_coords[u], vg.node_coords[v]
        if path[0] == u_coord:
            path_start_is_u = True
        elif path[0] == v_coord:
            path_start_is_u = False
        else:
            raise ValueError(
                f"edge ({u}, {v}) path does not start at either endpoint's "
                f"coord — build_vessel_graph invariant violated"
            )

        for i, coord in enumerate(path):
            c = tuple(coord)
            arclen_from_start = float(cum_from_path_start[i])
            arclen_from_u = arclen_from_start if path_start_is_u else (total_length - arclen_from_start)
            idx.point_to_edge[c] = edge_id
            idx.point_arclength_from_u[c] = arclen_from_u

    # Isolated (degree-0) nodes never appear in graph.edges(), so they'd
    # otherwise be silently absent from the index — treat each as its own
    # zero-length sentinel "edge" from itself to itself.
    for n, data in vg.graph.nodes(data=True):
        if vg.graph.degree(n) == 0:
            coord = vg.node_coords[n]
            edge_id = (n, n)
            idx.edge_length[edge_id] = 0.0
            idx.point_to_edge[coord] = edge_id
            idx.point_arclength_from_u[coord] = 0.0

    return idx


def nearest_skeleton_point(coord, arclength_index: ArcLengthIndex):
    """Look up an already-on-skeleton voxel coord in the index.

    Deliberately does not snap to the nearest indexed point if `coord`
    isn't already one: samplers must only pick skeleton voxels as crop
    centers (see connectivity_dataset.py), and silently snapping an
    off-skeleton coord would mask that kind of sampling bug rather than
    surfacing it.
    """
    coord = tuple(coord)
    if coord not in arclength_index.point_to_edge:
        raise ValueError(
            f"coord {coord} is not an indexed skeleton voxel; the sampler "
            f"must only pass coordinates that lie on the skeleton"
        )
    return coord


def _edge_endpoints_with_distance(coord, edge_id, arclength_index: ArcLengthIndex):
    """Returns [(node_id, arc_distance_from coord to that node), ...] for
    both endpoints of coord's edge (both entries are the same node, twice,
    for the isolated-node sentinel edge — harmless, not a special case)."""
    u, v = edge_id
    total = arclength_index.edge_length[edge_id]
    dist_to_u = arclength_index.point_arclength_from_u[coord]
    dist_to_v = total - dist_to_u
    return [(u, dist_to_u), (v, dist_to_v)]


def geodesic_distance(vg: VesselGraph, arclength_index: ArcLengthIndex, coord_a, coord_b):
    """Geodesic (along-tree) distance and hop count between two skeleton
    voxel coords.

    Returns:
        (arc_distance, hop_distance): floats/int, or (None, None) if the
        two coords lie in different connected components (disconnected).
    """
    coord_a = nearest_skeleton_point(coord_a, arclength_index)
    coord_b = nearest_skeleton_point(coord_b, arclength_index)

    edge_a = arclength_index.point_to_edge[coord_a]
    edge_b = arclength_index.point_to_edge[coord_b]

    if edge_a == edge_b:
        arc_a = arclength_index.point_arclength_from_u[coord_a]
        arc_b = arclength_index.point_arclength_from_u[coord_b]
        return abs(arc_a - arc_b), 0

    options_a = _edge_endpoints_with_distance(coord_a, edge_a, arclength_index)
    options_b = _edge_endpoints_with_distance(coord_b, edge_b, arclength_index)

    best_arc = None
    best_hops = None
    for node_a, dist_to_node_a in options_a:
        for node_b, dist_to_node_b in options_b:
            try:
                path_nodes = nx.shortest_path(vg.graph, node_a, node_b, weight="length")
            except nx.NetworkXNoPath:
                continue
            path_len = sum(
                vg.graph[path_nodes[i]][path_nodes[i + 1]]["length"]
                for i in range(len(path_nodes) - 1)
            )
            total_arc = dist_to_node_a + path_len + dist_to_node_b
            # Number of junction nodes crossed getting from coord_a to
            # coord_b, i.e. how many bifurcations apart the two points are.
            # `len(path_nodes)` (not -1): when node_a == node_b (the two
            # edges share one immediate bifurcation), path_nodes = [node_a]
            # and that single shared node IS the one bifurcation between
            # coord_a and coord_b — hop must be 1 here, not 0 (0 is reserved
            # for same-edge, handled above before this loop is reached).
            hop_len = len(path_nodes)
            if best_arc is None or total_arc < best_arc:
                best_arc, best_hops = total_arc, hop_len

    if best_arc is None:
        return None, None  # different connected components
    return best_arc, best_hops


def connectivity_label(hop_distance: int | None, hop_k: int) -> str:
    """Bucket a hop distance (from `geodesic_distance`) into the 3-class
    connectivity label used for pretraining (docs/PROJECT.md Section 2)."""
    if hop_distance is None or hop_distance > hop_k:
        return DISCONNECTED_OR_FAR
    if hop_distance == 0:
        return SAME_BRANCH
    return NEAR_ANCESTOR
