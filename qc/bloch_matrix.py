# Matrix transforming the Bloch vector
from __future__ import annotations
from enum import Enum
from scipy.spatial.transform import Rotation as R

import numpy as np

import qc.gates


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


def get_rotation_matrix(axis: Axis, angle):
    x = angle if axis == Axis.X else 0
    y = angle if axis == Axis.Y else 0
    z = angle if axis == Axis.Z else 0
    return R.from_rotvec([x, y, z]).as_matrix()


class BlochMatrix(object):
    def __init__(self):
        self._rot = [[], []]

    @property
    def rot(self) -> np.array:
        return self._rot

    # return rotation matrix from so(3) given a gate from su(2) with parametrisation
    # a + ib, -c + id
    # c + id, a - ib
    def set_by_gate(self, g: qc.gates.Gate) -> BlochMatrix:
        a = g.gate[0, 0].real
        b = g.gate[0, 0].imag
        c = g.gate[1, 0].real
        d = g.gate[1, 0].imag

        self._rot = np.array((a**2 - b**2 - c**2 + d**2, 2 * (a * b - c * d), 2 * (b * d + a * c)),
                             (-2 * (a * b + c * d), a**2 - b**2 + c**2 - d**2, 2 * (a * d - b * c)),
                             (2 * (b * d - a * c), 2 * ( b * c - a * d), a**2 + b**2 - c**2 - d**2))
        return self
