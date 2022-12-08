from abc import ABC, abstractmethod

import numpy as np
from pennylane.operation import Operation


class AbstractCircuit(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @property
    @abstractmethod
    def wires(self):
        raise NotImplementedError("The circuit does not have wires.")

    @abstractmethod
    def init(self, rng_key=None) -> np.ndarray:
        raise NotImplementedError

    def _circuit_ansatz(self, x) -> Operation:
        raise NotImplementedError("The circuit does not have a circuit ansatz.")

    # only used if we want to transpile the circuit.
    def update_circuit_ansatz(
        self, circuit_ansatz: Operation, num_qubits: int
    ) -> Operation:
        self.num_qubits = num_qubits
        self._circuit_ansatz = circuit_ansatz
