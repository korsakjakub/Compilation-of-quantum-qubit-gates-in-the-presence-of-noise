# Density matrix operations
import numpy as np


class DensityMatrix(object):

    def __init__(self, matrix: np.array) -> None:
        self._state = matrix

    @property
    def state(self) -> np.array:
        return self._state

    def from_diagonals(self, diagonals: np.array) -> None:
        self._state = np.diag(diagonals)

    def get_bloch_vector(self) -> np.array:
        return np.array((self._state[0, 1] + self._state[1, 0],
                         (self._state[0, 1] - self._state[1, 0]) * 1j,
                         self._state[0, 0] - self._state[1, 1]))

    def apply_noise(self, visibility: float, times: int = 1):
        if 0.0 <= visibility <= 1.0:
            return visibility**times * self._state + (1.0 - visibility**times) * 0.5 * np.identity(2)
        return self._state
