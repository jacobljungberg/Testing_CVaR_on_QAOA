import pennylane.numpy as pnp
from vqa.hamiltonian import molecular_hamiltonian
from vqa.templates.circuits import MolecularHamiltonianCircuit


# currently unused
def _hf_params_h2o(param_shape):
    """Helper function to get the Hartree-Fock parameters for the H2O molecule.

    Args:
        param_shape (tuple): The shape of the parameters.

    Returns:
        numpy.ndarray: The Hartree-Fock parameters for the H2O molecule.

    """
    params = pnp.zeros(param_shape)

    for i in range(param_shape[0]):
        for j in range(1, 10):
            params[i, j, 1] = pnp.pi

        params[i, 10, 1] = -pnp.pi

    return params


def _hf_params_h2o_frozen(param_shape):
    """Helper function to get the Hartree-Fock parameters for the H2O molecule
        with frozen orbitals.

    Args:
        param_shape (tuple): The shape of the parameters.

    Returns:
        numpy.ndarray: The Hartree-Fock parameters for the H2O molecule.

    """
    params = pnp.zeros(param_shape)

    for i in range(param_shape[0]):
        for j in range(1, 8):
            params[i, j, 1] = pnp.pi

        params[i, 8, 1] = -pnp.pi

    return params


def h2o_vqe_circuit(
    num_layers: int = 1,
    distance: float = 1.32,
    angle: float = 104.5,
    hf_params: bool = False,
    perturb_hf_params: bool = True,
    seed: int = 0,
) -> MolecularHamiltonianCircuit:
    """
    Build a quantum circuit for the Variational Quantum Eigensolver (VQE)
        simulation of the H2O molecule.

    Args:
        num_layers (int, optional): Number of layers in the circuit. Defaults to 1.
        distance (float, optional): Distance between atoms in the molecule.
            Defaults to 1.0.
        angle (float, optional): Angle between atoms in the molecule.
            Defaults to 90.0.
        hf_params (bool, optional): Flag to specify whether to use Hartree-Fock parameters.
            Defaults to False.
        perturb_hf_params (bool, optional): Flag to specify whether to perturb Hartree-Fock parameters.
            Defaults to True.
        seed (int, optional): Seed for random number generation. Defaults to 0.

    Returns:
        MolecularHamiltonianCircuit: A quantum circuit object representing the H2O molecule.


    Example:
    >>> circuit = h2o_vqe_circuit(num_layers=2)
    >>> print(circuit.num_layers)
    2
    >>> params = circuit.init()

    """

    # Vary distances between 0.5 and 3
    # and angles between 90 and 120.
    name = "h2o"
    symbols = ["O", "H", "H"]
    frozen = 1
    active_electrons = 8
    active_orbitals = 6

    x = distance * pnp.cos(pnp.deg2rad(angle) / 2.0)
    y = distance * pnp.sin(pnp.deg2rad(angle) / 2.0)

    coordinates = pnp.array([[0.0, 0.0, 0.0], [x, y, 0.0], [x, -y, 0.0]])
    initial_state = pnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

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
        params = _hf_params_h2o_frozen(circuit.params_shape).flatten()
        circuit.params = params

        if perturb_hf_params:
            pnp.random.seed(seed)
            circuit.params += 0.1 * pnp.random.normal(size=circuit.params.shape)

    return circuit
