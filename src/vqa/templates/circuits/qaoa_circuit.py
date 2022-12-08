import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from beartype import beartype
from pennylane.operation import Operation
from vqa.qaoa.initial_parameter import x0
from vqa.qaoa.mixer_h import x_mixer
from vqa.templates.circuits import AbstractCircuit
from vqa.templates.state_preparation import Plus


class QAOACircuit(AbstractCircuit):
    def __init__(
        self,
        H: qml.Hamiltonian,
        initial_state: qml.qnode = Plus,
        mixer_h: qml.qnode = x_mixer,
        num_layers: int = 1,
    ):
        self.H = H
        self.num_qubits = len(self.H.wires)
        self.dev_qubit = qml.device("default.qubit", wires=np.arange(self.num_qubits))
        self.initial_state = initial_state
        self.mixer_h = mixer_h
        self.num_layers = num_layers

        assert (
            self.initial_state.num_qubits == self.num_qubits
        ), f"N qubits initial state :{self.initial_state.num_qubits}, N qubits H {self.num_qubits}"
        assert len(self.mixer_h.wires) == len(
            self.H.wires
        ), f"N qubits mixer_h: {len(self.mixer_h.wires)} N qubits H: {len(self.H.wires)} "

    @property
    def wires(self):
        return self.H.wires

    def init(self, rng_key=None):
        return self._params(rng_key)

    def _params(self, rng_key=None):
        if rng_key is not None:
            pnp.random.seed(rng_key)
        gamma_init = pnp.array(2 * np.pi * np.random.rand(self.num_layers))
        beta_init = pnp.array(np.pi * np.random.rand(self.num_layers))
        x0 = pnp.concatenate((gamma_init, beta_init), axis=0)
        return x0

    def __call__(self, x: pnp.ndarray) -> pnp.ndarray:
        return self._circuit_ansatz(x)

    def _circuit_ansatz(self, params) -> Operation:
        assert 2 * self.num_layers == len(
            params
        ), f"{len(params)} != {2 * self.num_layers}"
        self.initial_state()
        beta_list = params[self.num_layers :]
        gamma_list = params[: self.num_layers]
        for beta, gamma in zip(beta_list, gamma_list):
            qml.ApproxTimeEvolution(self.H, gamma, 1)
            qml.ApproxTimeEvolution(self.mixer_h, beta, 1)
