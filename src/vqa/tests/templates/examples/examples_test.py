import numpy as np
import pennylane as qml
import pytest
from vqa import transforms
from vqa.hamiltonian import get_eigenvalues_hamiltonian
from vqa.templates.examples import (
    h2_vqe_circuit,
    h2o_vqe_circuit,
    h4_vqe_circuit,
    lih_vqe_circuit,
    maxcut_qaoa_circuit,
    tsp_qaoa_circuit,
)


@pytest.mark.parametrize(
    "num_layers, seed",
    [
        (2, 0),
        (1, 0),
    ],
)
def h2_vqe_circuit_test(num_layers, seed):
    # circuit
    circuit = h2_vqe_circuit(num_layers)
    params = circuit.init(seed)
    fun = transforms.exp_val(circuit, circuit.H)

    exp_val = fun(params)


@pytest.mark.parametrize(
    "num_layers, seed",
    [
        (2, 0),
        (1, 0),
    ],
)
def h2o_vqe_circuit_test(num_layers, seed):
    # circuit
    circuit = h2o_vqe_circuit(num_layers=num_layers)
    params = circuit.init(seed)
    fun = transforms.exp_val(circuit, circuit.H)

    exp_val = fun(params)


@pytest.mark.parametrize(
    "num_layers, seed",
    [
        (2, 0),
        (1, 0),
    ],
)
def h4_vqe_circuit_test(num_layers, seed):
    # circuit
    circuit = h4_vqe_circuit(num_layers=num_layers)
    params = circuit.init(seed)
    fun = transforms.exp_val(circuit, circuit.H)

    exp_val = fun(params)


@pytest.mark.parametrize(
    "num_layers, seed",
    [
        (2, 0),
        (1, 0),
    ],
)
def lih_vqe_circuit_test(num_layers, seed):
    # circuit
    circuit = lih_vqe_circuit(num_layers)
    params = circuit.init(seed)
    fun = transforms.exp_val(circuit, circuit.H)

    exp_val = fun(params)


@pytest.mark.parametrize(
    "num_layers, num_nodes, seed",
    [
        (2, 5, 0),
        (1, 4, 0),
    ],
)
def maxcut_qaoa_circuit_test(num_layers, num_nodes, seed):
    # circuit
    circuit = maxcut_qaoa_circuit(num_layers, num_nodes, seed=seed)
    params = circuit.init(seed)
    # Hamiltonian
    eigenvalues = get_eigenvalues_hamiltonian(circuit.H)
    min_cost, max_cost = np.min(eigenvalues), np.max(eigenvalues)

    fun = transforms.exp_val(circuit, circuit.H)

    exp_val = fun(params)


@pytest.mark.parametrize(
    "num_layers, num_cities, seed",
    [
        (2, 2, 0),
        (1, 3, 0),
    ],
)
def tsp_qaoa_circuit_test(num_layers, num_qubits, seed):
    # circuit
    circuit = tsp_qaoa_circuit(num_layers, num_qubits, seed=seed)
    params = circuit.init(seed)
    # Hamiltonian
    eigenvalues = get_eigenvalues_hamiltonian(circuit.H)
    min_cost, max_cost = np.min(eigenvalues), np.max(eigenvalues)

    fun = transforms.exp_val(circuit, circuit.H)

    exp_val = fun(params)


if __name__ == "__main__":
    h2_vqe_circuit_test(num_layers=2, seed=0)
    h2o_vqe_circuit_test(num_layers=1, seed=0)
    h4_vqe_circuit_test(num_layers=1, seed=0)
    lih_vqe_circuit_test(num_layers=2, seed=0)
    maxcut_qaoa_circuit_test(2, 4, 0)
    tsp_qaoa_circuit_test(2, 2, 0)
    print("All tests passed!")
