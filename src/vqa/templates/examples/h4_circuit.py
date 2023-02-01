from typing import Tuple

import pennylane.numpy as pnp
from vqa.hamiltonian import molecular_hamiltonian
from vqa.templates.circuits import MolecularHamiltonianCircuit


def _hf_params_h4(param_shape: Tuple):
    """Helper function to get the Hartree-Fock parameters for the H4 molecule.

    Args:
        param_shape (tuple): The shape of the parameters.

    Returns:
        numpy.ndarray: The Hartree-Fock parameters for the H4 molecule.

    """
    params = pnp.zeros(param_shape)

    for i in range(param_shape[0]):
        params[i, 1, 1] = pnp.pi
        params[i, 2, 1] = pnp.pi
        params[i, 3, 1] = pnp.pi
        params[i, 4, 1] = -pnp.pi

    return params


def h4_vqe_circuit(
    num_layers: int = 1,
    distance: float = 1.0,
    angle: float = 90.0,
    hf_params: bool = False,
    perturb_hf_params: bool = True,
    seed: int = 0,
) -> MolecularHamiltonianCircuit:
    """
    Build a quantum circuit for the Variational Quantum Eigensolver (VQE) simulation of the H4 molecule.

    Args:
        num_layers (int, optional): Number of layers in the circuit. Defaults to 1.
        distance (float, optional): Distance between atoms in the molecule. Defaults to 1.0.
        angle (float, optional): Angle between atoms in the molecule. Defaults to 90.0.
        hf_params (bool, optional): Flag to specify whether to use Hartree-Fock parameters. Defaults to False.
        perturb_hf_params (bool, optional): Flag to specify whether to perturb Hartree-Fock parameters. Defaults to True.
        seed (int, optional): Seed for random number generation. Defaults to 0.

    Returns:
        MolecularHamiltonianCircuit: A quantum circuit object representing the H4 molecule.


    Example:
    >>> circuit = h4_vqe_circuit(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """

    # I would suggest values for dist between 0.5 and 3
    # and angles between 75 and 105
    name = "H4"
    symbols = ["H", "H", "H", "H"]
    active_electrons = 4
    active_orbitals = 4
    frozen = 0

    x = distance * pnp.cos(pnp.deg2rad(angle) / 2.0)
    y = distance * pnp.sin(pnp.deg2rad(angle) / 2.0)

    coordinates = pnp.array([[x, y, 0.0], [x, -y, 0.0], [-x, -y, 0.0], [-x, y, 0.0]])
    initial_state = pnp.array([1, 1, 1, 1, 0, 0, 0, 0])

    (H, num_qubits, ground_state_energy, hf_occ, no_occ,) = molecular_hamiltonian(
        symbols=symbols,
        coordinates=coordinates,
        name=name,
        frozen=frozen,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
    circuit = MolecularHamiltonianCircuit(
        num_layers=num_layers,
        H=H,
        initial_state=initial_state,
    )

    if hf_params:
        params = _hf_params_h4(circuit.params_shape).flatten()
        circuit.params = params

        if perturb_hf_params:
            pnp.random.seed(seed)
            circuit.params += 0.1 * pnp.random.normal(size=circuit.params.shape)

    return circuit
