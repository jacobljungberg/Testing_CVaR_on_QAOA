import pennylane.numpy as pnp
from vqa.templates.circuits import MolecularHamiltonianCircuit


# currently unused
def _hf_params_h2o(param_shape):
    params = pnp.zeros(param_shape)

    for i in range(param_shape[0]):
        for j in range(1, 10):
            params[i, j, 1] = pnp.pi

        params[i, 10, 1] = -pnp.pi

    return params


def _hf_params_h2o_frozen(param_shape):
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
    # Vary distances between 0.5 and 3
    # and angles between 90 and 120.

    name = "h2o"
    symbols = ["O", "H", "H"]
    frozen = 1
    active_electrons = 8
    active_orbitals = 6

    x = distance * pnp.cos(angle / 2.0)
    y = distance * pnp.sin(angle / 2.0)

    coordinates = pnp.array([[0.0, 0.0, 0.0], [x, y, 0.0], [x, -y, 0.0]])
    initial_state = pnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

    circuit = MolecularHamiltonianCircuit(
        num_layers=num_layers,
        name=name,
        symbols=symbols,
        coordinates=coordinates,
        initial_state=initial_state,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        frozen=frozen,
    )

    if hf_params:
        params = _hf_params_h2o_frozen(circuit.params_shape).flatten()
        circuit.params = params

        if perturb_hf_params:
            pnp.random.seed(seed)
            circuit.params += 0.1 * pnp.random.normal(size=circuit.params.shape)

    return circuit
