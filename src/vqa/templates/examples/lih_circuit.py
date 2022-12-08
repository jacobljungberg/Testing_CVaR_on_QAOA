import pennylane.numpy as pnp
from vqa.templates.circuits import MolecularHamiltonianCircuit


# currently unused
def _hf_params_lih(param_shape):
    params = pnp.zeros(param_shape)
    for i in range(param_shape[0]):
        params[i, 1, 1] = pnp.pi
        params[i, 2, 1] = pnp.pi
        params[i, 3, 1] = pnp.pi
        params[i, 4, 1] = -pnp.pi
    return params


def _hf_params_lih_frozen(param_shape):
    params = pnp.zeros(param_shape)
    for i in range(param_shape[0]):
        params[i, 1, 1] = -pnp.pi
        params[i, 2, 1] = pnp.pi
    return params


# TODO split up into two functions one with and one without HF params.
def lih_vqe_circuit(
    num_layers: int = 1,
    distance: float = 3.0,
    hf_params: bool = False,
    perturb_hf_params: bool = True,
    seed: int = 0,
) -> MolecularHamiltonianCircuit:

    name = "LiH"
    symbols = ["Li", "H"]
    frozen = 1
    coordinates = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    initial_state = pnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    circuit = MolecularHamiltonianCircuit(
        num_layers=num_layers,
        name=name,
        frozen=frozen,
        symbols=symbols,
        coordinates=coordinates,
        initial_state=initial_state,
        active_electrons=2,
        active_orbitals=5,
    )

    if hf_params:
        params = _hf_params_lih_frozen(circuit.params_shape).flatten()
        circuit.params = params

        if perturb_hf_params:
            pnp.random.seed(seed)
            circuit.params += 0.1 * pnp.random.normal(size=circuit.params.shape)

    return circuit
