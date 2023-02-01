import numpy as np
import optax
import pytest
from jax import jit
from jaxopt import OptaxSolver
from vqa import transforms
from vqa.hamiltonian import get_eigenvalues_hamiltonian
from vqa.templates.circuits import BarrenPlateauCircuit
from vqa.utils.utils import get_approximation_ratio


@pytest.mark.parametrize(
    "num_layers, num_qubits",
    [
        (5, 7),
        (7, 4),
        (9, 4),
    ],
)
def test_barren_plateau_circuit(num_layers, num_qubits):
    circuit = BarrenPlateauCircuit(num_layers, num_qubits)
    params = circuit.init()

    eigenvalues = get_eigenvalues_hamiltonian(circuit.H)
    min_cost, max_cost = np.min(eigenvalues), np.max(eigenvalues)

    loss_fun = jit(transforms.exp_val(circuit, circuit.H, interface="jax"))

    opt = optax.adam(0.05)
    solver = OptaxSolver(opt=opt, fun=loss_fun, maxiter=100)
    state = solver.init_state(params)
    for _ in range(100):
        params, state = solver.update(
            params=params,
            state=state,
        )

    min_cost_optimized = loss_fun(params)
    r = get_approximation_ratio(min_cost_optimized, min_cost, max_cost)

    assert r > 0.9, "Approximation ratio is too low. Optimization failed."


if __name__ == "__main__":
    test_barren_plateau_circuit(5, 7)
