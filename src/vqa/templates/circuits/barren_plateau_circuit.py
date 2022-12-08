# https://arxiv.org/pdf/1803.11173.pdf
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from pennylane.operation import Operation
from vqa.templates.circuits import AbstractCircuit
from vqa.utils.decorator import memory, obj_memory
from vqa.utils.utils import pairwise


class BarrenPlateauCircuit(AbstractCircuit):
    def __init__(self, num_layers, num_qubits: int = 5):
        super().__init__()
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.list_gate_set = []  # (n_qubits, layers)
        self.params = None

    @property
    def wires(self):
        return range(self.num_qubits)

    def init(self, rng_key=None):
        self.list_gate_set = self._pauli_gates(self.num_layers, self.wires, rng_key)
        return self._params(rng_key)

    def _pauli_gates(self, layer, wires, rng_key):
        if rng_key is not None:
            np.random.seed(rng_key)
        list_gate_set = []
        for _ in range(layer):
            gate_set = [qml.RX, qml.RY, qml.RZ]
            random_gate_sequence = {i: np.random.choice(gate_set) for i in wires}
            list_gate_set.append(random_gate_sequence)
        return list_gate_set

    def _params(self, rng_key):
        return qnp.array(
            qnp.random.random((self.num_layers * self.num_qubits)) * 2 * np.pi,
            requires_grad=True,
        )

    @property
    def H(self):
        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        coeffs = [1.0]
        H = qml.Hamiltonian(coeffs, obs)
        return H

    def __call__(self, x):
        return self._circuit_ansatz(x)

    def _circuit_ansatz(self, params) -> Operation:
        assert self.list_gate_set is not None
        assert (
            len(params.flatten()) == self.num_qubits * self.num_layers
        ), f"{len(params.flatten())} != {self.num_qubits * self.num_layers}"

        params = np.reshape(params, (self.num_layers, self.num_qubits))

        for i in self.wires:
            qml.RY(np.pi / 4, wires=i)

        for layer in range(self.num_layers):

            for wire in self.wires:
                self.list_gate_set[layer][wire](params[layer][wire], wires=wire)

            qml.Barrier(wires=self.wires)

            list_1 = [(u, v) for u, v in pairwise(self.wires)]
            list_2 = [(u, v) for u, v in pairwise(np.roll(self.wires, -1))]

            for (i, j) in list_1:
                qml.CZ(wires=[i, j])

            for (u, v) in list_2:
                qml.CZ(wires=[u, v])

            # for i in range(self.num_qubits - 1):
            # qml.CZ(wires=[i, i + 1])

            qml.Barrier(wires=self.wires)
