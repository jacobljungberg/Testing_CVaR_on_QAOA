import numpy as np
import pennylane as qml


def transverse_field_ising_hamiltonian(
    n_spins: int, transverse_field: float
) -> qml.Hamiltonian:
    """Calculates the Hamiltonian of the transverse field Ising model.

    Args:
        n_spins (int): The number of spins in the model.
        transverse_field (float): The strength of the transverse field.

    Returns:
        qml.Hamiltonian: The Hamiltonian of the transverse field Ising model.
    """
    # transverse field
    coeffs = [-transverse_field for _ in range(n_spins)]
    obs = [qml.PauliX(np.pi / 2) for _ in range(n_spins)]
    h_x = qml.Hamiltonian(coeffs, obs)

    # spin chain
    couplings = [(i, (i + 1) % n_spins) for i in range(n_spins)]
    coeffs = [-1 for _ in range(n_spins)]
    obs = [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in couplings]
    h_zz = qml.Hamiltonian(coeffs, obs)

    return h_zz + h_x
