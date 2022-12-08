import pennylane.numpy as pnp
from vqa.templates.circuits import MolecularHamiltonianCircuit


def _hf_params_h4(param_shape):
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
    # I would suggest values for dist between 0.5 and 3
    # and angles between 75 and 105
    name = "H4"
    symbols = ["H", "H", "H", "H"]
    active_electrons = 4
    active_orbitals = 4
    frozen = 0

    x = distance * pnp.cos(angle / 2.0)
    y = distance * pnp.sin(angle / 2.0)

    coordinates = pnp.array([[x, y, 0.0], [x, -y, 0.0], [-x, -y, 0.0], [-x, y, 0.0]])
    initial_state = pnp.array([1, 1, 1, 1, 0, 0, 0, 0])

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
        params = _hf_params_h4(circuit.params_shape).flatten()
        circuit.params = params

        if perturb_hf_params:
            pnp.random.seed(seed)
            circuit.params += 0.1 * pnp.random.normal(size=circuit.params.shape)

    return circuit
