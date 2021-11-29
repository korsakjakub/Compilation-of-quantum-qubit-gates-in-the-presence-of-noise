from __future__ import annotations

import enum
import math
from typing import List

import numpy as np
import picos as pc

import qc


def affine_channel_norm(delta, c):
    problem = pc.Problem()
    v = pc.RealVariable('v', 3)
    t = pc.RealVariable('t', 1)
    problem.add_list_of_constraints(t <= pc.Norm(delta * v + c))
    # problem.add_constraint(pc.Norm(delta * v + c) <= t)
    # problem.set_objective('min', t)
    problem.set_objective('max', t)
    print(problem)
    solution = problem.solve(solver="cvxopt")
    print(solution.primals)


class Noise(enum.Enum):
    Depolarizing = 1
    PauliX = 2
    PauliY = 3
    PauliZ = 4
    Pauli = 5
    AmplitudeDamping = 6


class Channel(object):

    def __init__(self, noise: Noise, vis: float = 1.0):
        self._rot = np.empty(shape=(3, 3))
        self.visibility = vis
        self.noise_type = noise

    @property
    def rot(self) -> np.array:
        return self._rot

    def add_noise(self, input_channels: List[qc.lp_input.WordDict], length: int) -> List[qc.lp_input.WordDict]:
        m = np.eye(3)
        if self.noise_type == Noise.Depolarizing:
            m = np.eye(3) * self.visibility ** length
        elif self.noise_type == Noise.PauliX:
            m = np.array([[1, 0, 0], [0, 2 * self.visibility - 1, 0], [0, 0, 2 * self.visibility - 1]], dtype=float) \
                ** length
        elif self.noise_type == Noise.PauliY:
            m = np.array([[2 * self.visibility - 1, 0, 0], [0, 1, 0], [0, 0, 2 * self.visibility - 1]], dtype=float) \
                ** length
        elif self.noise_type == Noise.PauliZ:
            m = np.array([[2 * self.visibility - 1, 0, 0], [0, 2 * self.visibility - 1, 0], [0, 0, 1]], dtype=float) \
                ** length
        elif self.noise_type == Noise.Depolarizing:
            m = np.eye(3)
        for i in range(len(input_channels)):
            input_channels[i]['m'] = np.matmul(m, input_channels[i]['m'])
        return input_channels


if __name__ == "__main__":
    test_delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    test_c = np.array([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)])
    affine_channel_norm(test_delta, test_c)
