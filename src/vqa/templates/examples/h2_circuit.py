import pennylane.numpy as pnp
from vqa.hamiltonian import molecular_hamiltonian
from vqa.templates.circuits import MolecularHamiltonianCircuit


def _hf_params_h2(param_shape):
    """Helper function to get the Hartree-Fock parameters for the H2 molecule.

    Args:
        param_shape (tuple): The shape of the parameters.

    Returns:
        numpy.ndarray: The Hartree-Fock parameters for the H4 molecule.

    """

    params = pnp.zeros(param_shape)

    for i in range(param_shape[0]):
        params[i, 1, 1] = -pnp.pi
        params[i, 2, 1] = pnp.pi

    return params


# TODO should the function always return circuit and min energy?
def h2_vqe_circuit(
    num_layers: int = 1,
    distance: float = 1.32,
    hf_params: bool = False,
    perturb_hf_params: bool = True,
    seed: int = 0,
) -> MolecularHamiltonianCircuit:
    """
    Build a quantum circuit for the Variational Quantum Eigensolver (VQE) simulation of the H2 molecule.

    Args:
        num_layers (int, optional): Number of layers in the circuit. Defaults to 1.
        distance (float, optional): Distance between atoms in the molecule. Defaults to 1.0.
        hf_params (bool, optional): Flag to specify whether to use Hartree-Fock parameters. Defaults to False.
        perturb_hf_params (bool, optional): Flag to specify whether to perturb Hartree-Fock parameters. Defaults to True.
        seed (int, optional): Seed for random number generation. Defaults to 0.

    Returns:
        MolecularHamiltonianCircuit: A quantum circuit object representing the H2 molecule.


    Example:
    >>> circuit = h2_vqe_circuit(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """

    name = "h2"
    symbols = ["H", "H"]
    frozen = 0
    coordinates = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    initial_state = pnp.array([1, 1, 0, 0])
    active_electrons = 2
    active_orbitals = 2

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
        params = _hf_params_h2(circuit.params_shape).flatten()
        circuit.params = params

        if perturb_hf_params:
            pnp.random.seed(seed)
            circuit.params += 0.05 * pnp.random.normal(size=circuit.params.shape)

    return circuit
