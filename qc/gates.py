import qutip.qip.operations.gates as q


class Gate:

    gates = {
        'H': q.hadamard_transform,
        'X': q.sigmax,
        'Y': q.sigmay,
        'Z': q.sigmaz,
        'I': q.qeye
    }

    def __init__(self, key: str) -> None:
        self.gate = self.gates[key]

    def get(self):
        return self.gate
