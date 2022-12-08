import numpy as np
import pennylane as qml
import pytest
from vqa import transforms
from vqa.hamiltonian import get_eigenvalues_hamiltonian
from vqa.templates.circuits import BarrenPlateauCircuit


@pytest.mark.parametrize(
    "num_layers, num_qubits, seed",
    [
        (5, 7, 0),
    ],
)
def diff_method_test(num_layers, num_qubits, seed):
    # circuit
    circuit = BarrenPlateauCircuit(num_layers, num_qubits)
    params = circuit.init(seed)
    # Hamiltonian
    eigenvalues = get_eigenvalues_hamiltonian(circuit.H)
    min_cost, max_cost = np.min(eigenvalues), np.max(eigenvalues)

    # loss function
    fun_parameter_shit = transforms.exp_val(
        circuit, circuit.H, diff_method="parameter-shift"
    )
    fun = transforms.exp_val(circuit, circuit.H)

    for _ in range(100):
        params = np.random.rand(35,) * np.random.rand(
            35,
        )
        gradients = qml.grad(fun)(params)
        gradients_parameter_shift = qml.grad(fun_parameter_shit)(params)
        np.testing.assert_almost_equal(gradients, gradients_parameter_shift, decimal=7)


if __name__ == "__main__":
    diff_method_test(5, 7, 0)
