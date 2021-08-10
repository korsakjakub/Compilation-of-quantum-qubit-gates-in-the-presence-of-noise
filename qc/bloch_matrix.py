# Matrix transforming the Bloch vector
from __future__ import annotations

from enum import Enum
from typing import List

import numpy as np
import scipy.spatial.transform.rotation
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


def get_rotation_matrix(axis: Axis, angle):
    x = angle if axis == Axis.X else 0
    y = angle if axis == Axis.Y else 0
    z = angle if axis == Axis.Z else 0
    return R.from_rotvec((x, y, z)).as_matrix()


def get_bloch_vectors(matrices, initial=np.array((0.0, 0.0, 1.0))) -> np.ndarray:
    vectors = np.zeros(shape=(len(matrices), 3))
    for i in range(len(matrices)):
        vectors[i] = np.dot(matrices[i], initial)
    return np.unique(vectors, axis=0)


class BlochMatrix(object):
    gates = {
        'H': np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
        'X': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        'Y': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        'Z': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        'T': np.array(
            [[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0], [np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]]),
        'R': np.array(
            [[np.cos(np.pi / 4), np.sin(np.pi / 4), 0], [- np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]]),
        'I': np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        )
    }

    def __init__(self, vis: float = 1.0):
        self._rot = np.empty(shape=(3, 3))
        self.visibility = vis

    @property
    def rot(self) -> np.array:
        return self._rot

    def get_universal(self, name: str):
        return self.gates[name]

    # return rotation matrix from so(3) given a gate from su(2) with parametrisation
    # a + ib, -c + id
    # c + id, a - ib
    def set_by_gate(self, g: List) -> BlochMatrix:
        a = g[0]
        b = g[1]
        c = g[2]
        d = g[3]

        self._rot = np.array(((a ** 2 - b ** 2 - c ** 2 + d ** 2, 2 * (a * b - c * d), 2 * (b * d + a * c)),
                              (-2 * (a * b + c * d), a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * (a * d - b * c)),
                              (2 * (b * d - a * c), - 2 * (b * c + a * d), a ** 2 + b ** 2 - c ** 2 - d ** 2)))
        return self

    def get_random(self, seed):
        self._rot = R.random().as_matrix()
        return self

    def combine(self, word: List[str]):
        self._rot = np.identity(3)
        for g in word:
            self._rot = np.matmul(self._rot, self.get_universal(g))
        return self

    def add_noise(self, mat: np.ndarray, length: int) -> np.ndarray:
        for i in range(len(mat)):
            mat[i] *= self.visibility ** length
        return np.asarray(mat)

    # returns a list of 3x3 orthogonal matrices from a list of words
    def get_bloch_matrices(self, words) -> np.ndarray:
        matrices = []
        print("Generating matrices")
        for i in tqdm(range(len(words))):
            g = self.combine(words[i]).rot
            matrices.append(g)
        matrices = np.unique(matrices, axis=0)
        return matrices


if __name__ == "__main__":
    b = BlochMatrix()
    b.get_random(55)
