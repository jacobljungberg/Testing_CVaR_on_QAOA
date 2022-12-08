from typing import List

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from pennylane.operation import Operation
from pennylane.templates.layers import StronglyEntanglingLayers
from vqa.hamiltonian import molecular_hamiltonian
from vqa.templates.circuits import AbstractCircuit


class MolecularHamiltonianCircuit(AbstractCircuit):
    def __init__(
        self,
        num_layers: int = None,
        name: str = None,
        symbols: List = None,
        initial_state: pnp.ndarray = None,
        params: pnp.ndarray = None,
        coordinates: np.ndarray = None,
        active_electrons: int = None,
        active_orbitals: int = None,
        frozen: int = None,
    ):
        self.num_layers: int = num_layers
        self.name: str = name
        self.symbols: list = symbols
        self.initial_state: pnp.ndarray = initial_state
        self.params: pnp.ndarray = params
        self.coordinates: pnp.ndarray = coordinates
        self.active_electrons: int = active_electrons
        self.active_orbitals: int = active_orbitals
        self.frozen: int = frozen

        self.num_qubits: int = None
        self.H = None
        # self.exact_energy: float = None
        self.H, self.num_qubits, self.ground_state_energy = molecular_hamiltonian(
            symbols=self.symbols,
            coordinates=self.coordinates,
            name=self.name,
            frozen=self.frozen,
            active_electrons=self.active_electrons,
            active_orbitals=self.active_orbitals,
        )
        self.params_shape: List = (self.num_layers, self.num_qubits, 3)

    @property
    def wires(self):
        return self.H.wires

    def init(self, rng_key=None):
        return self._params(rng_key)

    def _params(self, rng_key=None):
        # If no params are given, initialize them randomly.
        if self.params is None:
            if rng_key is not None:
                pnp.random.seed(rng_key)
            self.params = pnp.random.rand(self.num_layers * self.num_qubits * 3)
        # Use Hartree-Fock parameters
        # assert self.params_shape == self.params.shape, "params shape is wrong"
        return self.params

    def __repr__(self) -> str:
        return f"{self.name}"

    def __call__(self, x: pnp.ndarray):
        return self._circuit_ansatz(x)

    def _circuit_ansatz(self, params: pnp.ndarray) -> Operation:
        assert isinstance(self.num_qubits, int), "num_qubits is not an int"
        params = np.reshape(params, newshape=self.params_shape)
        assert self.params_shape == params.shape, "params shape is wrong"
        wires = range(self.num_qubits)
        qml.BasisState(self.initial_state, wires=wires)
        StronglyEntanglingLayers(
            weights=params, wires=wires, ranges=[1 for _ in range(self.num_layers)]
        )
