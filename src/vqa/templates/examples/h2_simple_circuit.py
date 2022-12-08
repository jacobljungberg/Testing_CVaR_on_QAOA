from typing import List

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from vqa.templates.circuits import MolecularHamiltonianCircuit


class H2SimpleCircuit(MolecularHamiltonianCircuit):
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
        super().__init__(
            num_layers=num_layers,
            name=name,
            symbols=symbols,
            initial_state=initial_state,
            params=params,
            coordinates=coordinates,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            frozen=frozen,
        )

    def __call__(self, x: pnp.ndarray) -> pnp.ndarray:
        def _circ(params: pnp.ndarray):
            qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
            # applies the single excitations
            qml.SingleExcitation(params[0], wires=[0, 2])
            qml.SingleExcitation(params[1], wires=[1, 3])

        return _circ(x)


def _hf_params_h2(param_shape):
    params = pnp.zeros(param_shape)

    for i in range(param_shape[0]):
        params[i, 1, 1] = -pnp.pi
        params[i, 2, 1] = pnp.pi

    return params


def h2_simple_vqe_circuit(
    num_layers: int = 1,
    distance: float = 1.32,
    hf_params: bool = False,
    perturb_hf_params: bool = True,
    seed: int = 0,
) -> MolecularHamiltonianCircuit:

    name = "h2"
    symbols = ["H", "H"]
    frozen = 0
    coordinates = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    initial_state = pnp.array([1, 1, 0, 0])

    circuit = H2SimpleCircuit(
        num_layers=num_layers,
        name=name,
        symbols=symbols,
        coordinates=coordinates,
        initial_state=initial_state,
        active_electrons=2,
        active_orbitals=2,
        frozen=frozen,
    )

    if hf_params:
        params = _hf_params_h2(circuit.params_shape).flatten()
        circuit.params = params

        if perturb_hf_params:
            pnp.random.seed(seed)
            circuit.params += 0.05 * pnp.random.normal(size=circuit.params.shape)

    circuit.params = pnp.random.rand(2)

    return circuit
