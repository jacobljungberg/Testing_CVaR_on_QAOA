from typing import List, Tuple

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from pennylane.operation import Operation
from pennylane.templates.layers import StronglyEntanglingLayers
from vqa.hamiltonian import molecular_hamiltonian
from vqa.templates.circuits import AbstractCircuit


class MolecularHamiltonianCircuit(AbstractCircuit):
    """A hardware efficient circuit ansatz for molecular Hamiltonians.

    This class is a subclass of `AbstractCircuit` and is used to implement a
    hardware efficient ansatz for molecular Hamiltonians. The class provides
    functions to define the structure of the circuit, the underlying parameters

    Args:
        num_layers (int, optional): The number of layers in the circuit. Defaults to 1.
        H (qml.Hamiltonian): The molecular Hamiltonian. Defaults to None.
        initial_state (pnp.ndarray): The initial state of the circuit. Defaults to None.

    Attributes:
        num_layers (int): The number of layers in the circuit.
        H (qml.Hamiltonian): The molecular Hamiltonian.
        initial_state (pnp.ndarray): The initial state of the circuit.
        num_qubits (int): The number of qubits in the circuit.
        params_shape (List[int]): The shape of the parameters in the circuit.
    """

    def __init__(
        self,
        num_layers: int = 1,
        H: qml.Hamiltonian = None,
        initial_state: pnp.ndarray = None,
    ):
        self.num_layers: int = num_layers
        self.H = H
        self.initial_state: pnp.ndarray = initial_state
        self.num_qubits = len(self.H.wires)
        self.params_shape: Tuple = (self.num_layers, self.num_qubits, 3)

    @property
    def wires(self):
        """Get the wires in the circuit. It is assumed that the wires of the circuit
        are the same as the wires of the Hamiltonian.

        Returns:
            List[int]: The list of wires in the circuit.
        """
        return self.H.wires

    @wires.setter
    def wires(self, wires):
        print("update wires: ", wires)
        self._wires = wires

    def init(self, seed: int = None):
        """Initialize the parameters of the circuit.

        Args:
            seed (int, optional): The random number generator seed.
                Defaults to None.

        Returns:
            np.ndarray: The initialized parameters of the circuit.
        """
        return self._params(seed)

    def _params(self, seed: int = None):
        """Initialize the parameters for the circuit evaluation.

        Args:
            seed (int, optional): Seed for the random number generator.

        Returns:
            np.ndarray: The parameters for the circuit evaluation.

        Raises:
            AssertionError: If `params` does not have the expected shape.
        """
        # if self.params is None:
        if seed is not None:
            pnp.random.seed(seed)
        self.params = pnp.random.rand(self.num_layers * self.num_qubits * 3)
        # self.params = pnp.reshape(params, newshape=self.params_shape)
        # Use Hartree-Fock parameters
        return self.params

    def __repr__(self) -> str:
        """Get a string representation of the circuit.

        Returns:
            str: The string representation of the circuit.
        """
        return f"{self.name}"

    def __call__(self, params: pnp.ndarray) -> Operation:
        """Evaluate the circuit with given parameters.

        Args:
            params (np.ndarray): The parameters to be used for evaluating the circuit.

        Returns:
            Operation: The evaluation of the circuit.
        """
        return self._circuit_ansatz(params)

    def _circuit_ansatz(self, params: pnp.ndarray) -> Operation:
        """Perform the actual circuit evaluation with given parameters.

        Args:
            params (np.ndarray): The parameters to be used for evaluating the circuit.

        Returns:
            Operation: The evaluation of the circuit.

        Raises:
            AssertionError: If `params` does not have the expected shape or if
                `num_qubits` is not an integer.
        """
        params = np.reshape(params, newshape=self.params_shape)
        assert isinstance(self.num_qubits, int), "num_qubits is not an int"
        assert self.params_shape == params.shape, "params shape is wrong"
        wires = range(self.num_qubits)
        qml.BasisState(self.initial_state, wires=wires)
        StronglyEntanglingLayers(
            weights=params, wires=wires, ranges=[1 for _ in range(self.num_layers)]
        )
