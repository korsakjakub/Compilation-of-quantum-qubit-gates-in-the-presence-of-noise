# Density matrix operations
import numpy as np


class DensityMatrix:

    def __init__(self, matrix: np.matrix) -> None:
        self.state = matrix

    @classmethod
    def from_diagonals(cls, diagonals: np.array) -> None:
        cls.state = np.diag(diagonals)

    def get_bloch_vector(self) -> np.array:
        return np.array((self.state[0, 1] + self.state[1, 0],
                         (self.state[0, 1] - self.state[1, 0]) * 1j,
                         self.state[0, 0] - self.state[1, 1]))
