from __future__ import annotations

import enum

import numpy as np
from scipy.linalg import norm
from scipy.optimize import minimize


def affine_channel_distance(ch1: np.ndarray, ch2: np.ndarray):
    d = (ch1 - ch2)[:3, :3]
    c = (ch1 - ch2)[:3, 3].reshape((3, 1))

    def _objective(var_vec):
        return -norm(d @ var_vec + c) / (np.sqrt(3))

    def _constraint1(var_vec):
        return 1 - norm(var_vec)

    v0 = np.array([0.5, 0.5, 0.5], dtype=float)
    domain = (-1.0, 1.0)
    bound = (domain, domain, domain)
    con = {'type': 'ineq', 'fun': _constraint1}
    sol = minimize(_objective, v0, method='SLSQP', bounds=bound, constraints=con)
    return -sol.fun


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
        self.eta = vis
        self.noise_type = noise

    @property
    def rot(self) -> np.array:
        return self._rot

    def add_noise(self, input_channels, length: int):
        if self.noise_type == Noise.Depolarizing:
            m = np.eye(3) * (1 - self.eta) ** length
        elif self.noise_type == Noise.PauliX:
            m = np.array([[1, 0, 0], [0, 2 * (1 - self.eta) - 1, 0], [0, 0, 2 * (1 - self.eta) - 1]], dtype=float) \
                ** length
        elif self.noise_type == Noise.PauliY:
            m = np.array([[2 * (1 - self.eta) - 1, 0, 0], [0, 1, 0], [0, 0, 2 * (1 - self.eta) - 1]], dtype=float) \
                ** length
        elif self.noise_type == Noise.PauliZ:
            m = np.array([[2 * (1 - self.eta) - 1, 0, 0], [0, 2 * (1 - self.eta) - 1, 0], [0, 0, 1]], dtype=float) \
            ** length

        elif self.noise_type == Noise.AmplitudeDamping:
            # amplitude damping represented by 4x4 matrix -> affine transformation r' = N r + c
            m = np.array([[np.sqrt((1 - self.eta)), 0, 0, 0], [0, np.sqrt((1 - self.eta)), 0, 0],
                          [0, 0, (1 - self.eta), self.eta], [0, 0, 0, 1]], dtype=float) ** length
            for i in range(len(input_channels)):
                # append a column and a row at the end so the shapes match
                affine_mat = np.concatenate(
                    (np.concatenate((input_channels[i]['m'], np.array([[0], [0], [0]], dtype=float)), axis=1),
                     np.array([[0, 0, 0, 1]], dtype=float)), axis=0)
                input_channels[i]['m'] = np.matmul(m, affine_mat)
            return input_channels
        else:
            m = np.eye(3)
        for i in range(len(input_channels)):
            input_channels[i]['m'] = np.matmul(m, input_channels[i]['m'])
        return input_channels


if __name__ == "__main__":
    channel1 = np.array([[0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1]])
    channel2 = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0.3],
                         [0, 0, 0, 1]])
    print(affine_channel_distance(channel1, channel2))
