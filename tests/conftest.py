"""Shared synthetic fixtures for graph-extraction and connectivity-dataset
tests: a hand-built Y-shaped vessel skeleton (one bifurcation, three arms)
plus a disjoint line segment and a disjoint isolated single voxel, so tests
can exercise same-edge, cross-edge, and disconnected-component behavior
without needing any real image data.

Arm directions from the bifurcation `B` are chosen so that no two arms'
voxels are ever mutually 26-adjacent (each pair of arm directions has some
axis where they point in opposite unit directions, guaranteeing Chebyshev
distance >= 2 between any two points on different arms) — this keeps the
skeleton a clean tree (one degree-3 node, three degree-1 endpoints, no
accidental extra edges) rather than something with unintended adjacencies.
"""
from dataclasses import dataclass

import numpy as np
import pytest

_VOLUME_SHAPE = (21, 21, 21)
_B = (10, 10, 10)  # bifurcation, (z, y, x)
_ARM_DIRS = {
    "arm1": (1, 0, 0),
    "arm2": (-1, 1, 0),
    "arm3": (-1, -1, 0),
}
_ARM_LENGTH = 5
_ISOLATED_POINT = (0, 0, 0)  # far from B and from the disjoint segment below, no neighbors
_DISJOINT_SEGMENT = [(20, 20, 20 - i) for i in range(4)]  # 4-voxel line, far corner


@dataclass
class YSkeletonFixture:
    skeleton: np.ndarray
    volume_shape: tuple
    bifurcation: tuple
    arm_points: dict  # arm name -> list of (z, y, x), ordered from B outward, excluding B
    isolated_point: tuple
    disjoint_segment: list


def _arm_points(direction, length):
    dz, dy, dx = direction
    return [(_B[0] + dz * i, _B[1] + dy * i, _B[2] + dx * i) for i in range(1, length + 1)]


@pytest.fixture
def y_skeleton() -> YSkeletonFixture:
    skel = np.zeros(_VOLUME_SHAPE, dtype=bool)
    skel[_B] = True

    arm_points = {}
    for name, direction in _ARM_DIRS.items():
        pts = _arm_points(direction, _ARM_LENGTH)
        arm_points[name] = pts
        for p in pts:
            skel[p] = True

    for p in _DISJOINT_SEGMENT:
        skel[p] = True

    skel[_ISOLATED_POINT] = True

    return YSkeletonFixture(
        skeleton=skel,
        volume_shape=_VOLUME_SHAPE,
        bifurcation=_B,
        arm_points=arm_points,
        isolated_point=_ISOLATED_POINT,
        disjoint_segment=_DISJOINT_SEGMENT,
    )
