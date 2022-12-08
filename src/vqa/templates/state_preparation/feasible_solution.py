import numpy as np
import pennylane as qml
import jax.numpy as jnp

from vqa.templates.state_preparation.abstract_initial_state import InitialState
from vqa.utils.tsp_utils import classify_bitstrings_tsp


class FeasibleSolution(InitialState):
    def __init__(self, num_qubits, **kwargs):
        super().__init__(num_qubits, **kwargs)
        self.costs = (
            self.kwargs["costs"]
            if "costs" in self.kwargs
            else print("no costs not in kwargs")
        )
        label, _ = classify_bitstrings_tsp(self.costs)
        self.index_ = list(*jnp.where(label != 2))  # index for all feasible solutions
        self.wires = range(self.num_qubits)
        self.state = np.zeros(2**self.num_qubits, dtype=np.complex128)
        for i in self.index_:
            self.state[i] = 1 / np.sqrt(len(self.index_)) * 1 + 0j

    def __call__(self):
        qml.QubitStateVector(self.state, wires=self.wires)
