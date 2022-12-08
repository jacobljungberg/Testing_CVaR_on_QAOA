import numpy as np
import pennylane as qml

from .abstract_initial_state import InitialState


class Identity(InitialState):
    def __init__(self, num_qubits, **kwargs):
        super().__init__(num_qubits, **kwargs)

    def __call__(self):
        for i, qubit in enumerate(
            range(0, self.num_qubits, int(np.sqrt(self.num_qubits)))
        ):
            qml.PauliX(wires=qubit + i)
