import pennylane.numpy as pnp
from vqa.templates.circuits import MolecularHamiltonianCircuit


def _hf_params_h2(param_shape):
    params = pnp.zeros(param_shape)

    for i in range(param_shape[0]):
        params[i, 1, 1] = -pnp.pi
        params[i, 2, 1] = pnp.pi

    return params


def h2_vqe_circuit(
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

    circuit = MolecularHamiltonianCircuit(
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

    return circuit
