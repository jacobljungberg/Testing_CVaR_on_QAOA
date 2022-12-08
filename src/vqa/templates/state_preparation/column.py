import numpy as np
import pennylane as qml

from .abstract_initial_state import InitialState


class Column(InitialState):
    def __init__(self, num_qubits, **kwargs):
        super().__init__(num_qubits, **kwargs)

    def __call__(self):
        for _, qubit in enumerate(
            range(0, int(np.sqrt(self.num_qubits)) ** 2, int(np.sqrt(self.num_qubits)))
        ):  # invalid solution
            qml.PauliX(wires=qubit)
