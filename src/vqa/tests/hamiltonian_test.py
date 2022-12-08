import numpy as np
import pennylane as qml
import pytest
from vqa.hamiltonian import get_eigenvalues_hamiltonian, tsp_hamiltonian
from vqa.hamiltonian.utils_hamiltonian import (
    compress_interactions,
    get_coeffs_monomials,
    get_ising_hamiltonian,
)
from vqa.instance import get_tsp_instance
from vqa.tests.utils_test import sympy_test_hamiltonian
from vqa.utils.maxcut_utils import get_maxcut_costs, get_maxcut_graph
from vqa.utils.tsp_utils import classify_bitstrings_tsp
from vqa.utils.utils import get_all_bitstrings


def test_ground_state_energy():
    n_qubits = 4
    wires = range(n_qubits)
    # get sympy expression
    H = sympy_test_hamiltonian()
    # translate into pennylane hamiltonian
    coeffs, interactions = get_coeffs_monomials(H)
    interactions, coeffs = compress_interactions(interactions, coeffs)
    cost_h = get_ising_hamiltonian(interactions, coeffs)

    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1 + 0j
    qml.QubitStateVector(state, wires=wires)


@pytest.mark.parametrize(
    "n",
    [
        (3),
        (4),
    ],
)
def test_maxcut_hamiltonian(n):
    graph = get_maxcut_graph(n, seed=123)
    costs = get_maxcut_costs(graph)
    cost_h, _ = qml.qaoa.maxcut(graph)
    costs_penny = get_eigenvalues_hamiltonian(cost_h)

    assert np.array_equal(costs, costs_penny), "Arrays are not equal"


@pytest.mark.parametrize(
    "n",
    [
        (3),
        (4),
    ],
)
def test_tsp_hamiltonian(n):
    ins = get_tsp_instance(n)

    cost_h = tsp_hamiltonian(ins, [1, 0])

    # get costs from Hamiltonian
    costs = get_eigenvalues_hamiltonian(cost_h)
    min_cost = np.min(costs)
    index_cost_min = np.where(costs == min_cost)[0]

    # verify that each optimal solution contains a single 1 in each row and col.
    bitstrings = get_all_bitstrings(len(cost_h.wires))
    best_bitstrings = bitstrings[index_cost_min]
    n_qubits = int(np.log2(len(costs)))
    all_bitstrings = get_all_bitstrings(n_qubits)  # .tolist()
    dim = int(np.sqrt(n_qubits))
    optimal_b = []
    for i, b in enumerate(all_bitstrings):
        sol = np.reshape(b, (dim, dim))
        valid_col = not len(np.where(np.sum(sol, axis=0) - 1 != 0)[0])
        valid_row = not len(np.where(np.sum(sol, axis=1) - 1 != 0)[0])
        if valid_row and valid_col:
            optimal_b.append(b)

    optimal_b = np.array(optimal_b)
    assert np.array_equal(best_bitstrings, optimal_b), "Arrays are not equal"
