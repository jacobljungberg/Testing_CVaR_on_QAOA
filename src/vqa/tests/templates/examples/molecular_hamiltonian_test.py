from typing import Callable

import numpy as np
import optax
import pennylane as qml
import pytest
from jax import jit
from jaxopt import OptaxSolver
from tqdm import tqdm
from vqa import transforms
from vqa.hamiltonian import molecular_hamiltonian
from vqa.templates.examples import (
    h2_vqe_circuit,
    h2o_vqe_circuit,
    h4_vqe_circuit,
    lih_vqe_circuit,
)
from vqa.utils.utils import get_approximation_ratio


@pytest.mark.parametrize(
    "get_circuit, num_layers, hf_params, seed",
    [
        (h2_vqe_circuit, 5, True, 0),
        (h2_vqe_circuit, 5, False, 0),
        (h4_vqe_circuit, 3, True, 0),
        (h4_vqe_circuit, 3, False, 0),
        (lih_vqe_circuit, 3, True, 0),
        (lih_vqe_circuit, 3, False, 0),
    ],
)
def molecular_hamiltonain_test(get_circuit: Callable, num_layers: int, seed: int):
    circuit = get_circuit(num_layers, hf_params=True)
    params = circuit.init()
    min_cost = circuit.ground_state_energy

    loss_fun = jit(transforms.exp_val(circuit, circuit.H, interface="jax"))

    opt = optax.adam(0.1)
    solver = OptaxSolver(opt=opt, fun=loss_fun, maxiter=100)
    state = solver.init_state(params)
    for _ in tqdm(range(100)):
        params, state = solver.update(
            params=params,
            state=state,
        )

    min_cost_optimized = loss_fun(params)
    r = get_approximation_ratio(min_cost_optimized, min_cost, 0)

    assert r > 0.8, "Optimization failed"

    print(f"ground state: {np.round(min_cost, 1)}")
    print(f"exp val: {np.round(min_cost_optimized, 2)}")
    print(f"approx. ratio: {np.round(r,2).real}")


@pytest.mark.parametrize(
    "get_circuit, seed",
    [
        (h2_vqe_circuit),
        (h4_vqe_circuit),
        (h2o_vqe_circuit),
        (lih_vqe_circuit),
    ],
)
def hartree_fock_params_test(get_circuit):
    # the energy must be always the same regardless of the number of layers
    list_layers = [1, 2, 3]
    list_exp_val = []
    for i, layer in enumerate(list_layers):
        circuit = get_circuit(layer, hf_params=True, perturb_hf_params=False)
        params = circuit.init()
        fun = transforms.exp_val(circuit, circuit.H)
        exp_val = fun(params)
        list_exp_val.append(exp_val)

    for i in range(len(list_exp_val) - 1):
        assert np.isclose(list_exp_val[i], list_exp_val[i + 1])


@pytest.mark.parametrize(
    "get_circuit",
    [
        (h2_vqe_circuit),
        (h4_vqe_circuit),
        (h2o_vqe_circuit),
        (lih_vqe_circuit),
    ],
)
def test_perturb_hf_params(get_circuit):
    # the energy must be always the same regardless of the number of layers
    list_layers = [1, 2, 3]
    list_exp_val = []
    for i, layer in enumerate(list_layers):
        circuit = h2_vqe_circuit(layer, hf_params=True, perturb_hf_params=True)
        params = circuit.init()
        fun = transforms.exp_val(circuit, circuit.H)
        exp_val = fun(params)
        list_exp_val.append(exp_val)

    for i in range(len(list_exp_val) - 1):
        assert not np.isclose(list_exp_val[i], list_exp_val[i + 1])


if __name__ == "__main__":
    # pytest libs/vqa/src/vqa/tests/molecular_hamiltonian_test.py

    molecular_hamiltonain_test(h2_vqe_circuit, 2, 0)
    molecular_hamiltonain_test(h4_vqe_circuit, 2, 0)
    molecular_hamiltonain_test(h2o_vqe_circuit, 2, 0)
    molecular_hamiltonain_test(lih_vqe_circuit, 2, 0)

    hartree_fock_params_test(h2_vqe_circuit)
    hartree_fock_params_test(h4_vqe_circuit)
    hartree_fock_params_test(h2o_vqe_circuit)
    hartree_fock_params_test(lih_vqe_circuit)

    test_perturb_hf_params(h2_vqe_circuit)
    test_perturb_hf_params(h4_vqe_circuit)
    test_perturb_hf_params(h2o_vqe_circuit)
    test_perturb_hf_params(lih_vqe_circuit)

    print("All tests passed!")
