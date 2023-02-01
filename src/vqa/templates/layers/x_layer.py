import numpy as np
from pennylane import qaoa


def Pauli_X(wires):
    """Generate a quantum layer that applies the Pauli X operator to each qubit.

    Args:
        n_qubits (int): The number of qubits in the circuit.

    Returns:
        qml.Hamiltonian: the Hamiltonian of the layer.
    """
    mixer_h = qaoa.x_mixer(wires)
    return mixer_h
