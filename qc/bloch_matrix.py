# Matrix transforming the Bloch vector
from __future__ import annotations
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as R
import qc.gates
from qc.gates import Gate


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


def get_rotation_matrix(axis: Axis, angle):
    x = angle if axis == Axis.X else 0
    y = angle if axis == Axis.Y else 0
    z = angle if axis == Axis.Z else 0
    return R.from_rotvec((x, y, z)).as_matrix()


class BlochMatrix(object):
    def __init__(self):
        self._rot = np.empty(shape=(3, 3))

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

        self._rot = np.array(((a ** 2 - b ** 2 - c ** 2 + d ** 2, 2 * (a * b - c * d), 2 * (b * d + a * c)),
                              (-2 * (a * b + c * d), a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * (a * d - b * c)),
                              (2 * (b * d - a * c), 2 * (b * c - a * d), a ** 2 + b ** 2 - c ** 2 - d ** 2)))
        return self

    def combine_with_noise(self, word: list[str], visibility: float):
        self._rot = np.identity(3)
        for g in word:
            gate = Gate().get_universal(g)
            self._rot = np.matmul(self._rot, visibility * self.set_by_gate(gate).rot)
        return self


# returns a list of 3x3 orthogonal matrices from a list of words
def get_bloch_matrices(words, visibility: float = 1.0):
    matrices = np.zeros(shape=(len(words), 3, 3))
    for i in range(len(words)):
        g = BlochMatrix().combine_with_noise(words[i], visibility).rot
        matrices[i] = g
    return matrices


def get_bloch_vectors(matrices, initial=np.array((0.0, 0.0, 1.0))):
    vectors = np.zeros(shape=(len(matrices), 3))
    for i in range(len(matrices)):
        vectors[i] = np.dot(matrices[i], initial)
    return vectors
