from __future__ import annotations
import numpy as np
import qutip.qip.operations.gates as q


class Gate(object):
    gates = {
        'H': 1j * q.hadamard_transform().full(),
        'X': 1j * q.sigmax().full(),
        'Y': 1j * q.sigmay().full(),
        'Z': 1j * q.sigmaz().full(),
        'T': (np.cos(np.pi / 8) - np.sin(np.pi / 8) * 1j) * q.phasegate(np.pi / 4).full(),
        'R': (np.cos(np.pi / 8) + np.sin(np.pi / 8) * 1j) * q.phasegate(- np.pi / 4).full(),
        "I": np.array([[1, 0], [0, 1]])
    }  # TODO: Spróbować bez tych faz

    def __init__(self, vis: float = 1.0) -> None:
        self._gate = [[], []]
        self.visibility = vis

    @property
    def gate(self) -> np.array:
        return self._gate

    def get_universal(self, name: str):
        self._gate = self.gates[name]
        return self

    # given a word from the set of gates, return their product
    def set_by_word(self, g: str) -> Gate:
        if len(g) > 1:
            self._gate = self.__combine_gates(g)
        elif g[0] in self.gates:
            self._gate = self.gates[g[0]]
        return self

    # su(2) matrix is of the form
    # a, -b*
    # b, a*,
    # where a, and b are complex
    def set_by_params(self, a: complex, b: complex) -> Gate:
        if np.isclose(a.real ** 2 + a.imag ** 2 + b.real ** 2 + b.imag ** 2, 1.0):
            self._gate = np.array(((a, - np.conj(b)), (b, np.conj(a))))
        else:
            print("\nNormalize the parameters!\n")
        return self

    def __combine_gates(self, word: str) -> np.array:
        out = ((1, 0), (0, 1))
        for s in word:
            out = self.visibility * np.matmul(out, self.gates[s]) + (1 - self.visibility) * self.gates["I"] / 2
        return out

    def get_gates(self, words) -> np.ndarray:
        matrices = np.zeros(shape=(len(words), 2, 2))
        for i in range(len(words)):
            g = self.set_by_word(words[i])
            matrices[i] = g.gate
        return matrices
