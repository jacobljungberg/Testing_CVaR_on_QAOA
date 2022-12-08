import numpy as np
import pennylane as qml
import pytest
from vqa import transforms
from vqa.hamiltonian import get_eigenvalues_hamiltonian
from vqa.templates.circuits import BarrenPlateauCircuit
from vqa.templates.examples import (
    h2_vqe_circuit,
    h2o_vqe_circuit,
    h4_vqe_circuit,
    lih_vqe_circuit,
    maxcut_qaoa_circuit,
)
from vqa.utils.utils import get_approximation_ratio


@pytest.mark.parametrize(
    "get_circuit, num_layers, seed",
    [
        (h2_vqe_circuit, 1, 0),
        (h2_vqe_circuit, 5, 0),
        (h2o_vqe_circuit, 4, 0),
        (h4_vqe_circuit, 2, 0),
        (lih_vqe_circuit, 3, 0),
        (maxcut_qaoa_circuit, 3, 0),
        (BarrenPlateauCircuit, 3, 0),
    ],
)
def test_metric_tensor_scaling_with_num_layers(get_circuit, num_layers, seed):
    circuit = get_circuit(num_layers)
    params = circuit.init(seed)
    fun = transforms.exp_val(circuit, circuit.H)
    F = qml.metric_tensor(fun)(params)

    assert (
        len(params) ** 2 == F.shape[0] * F.shape[1]
    ), f"Metric tensor has wrong shape, expected {len(params)**2} elements but got {F.shape[0]*F.shape[1]} elements. "


if __name__ == "__main__":
    test_metric_tensor_scaling_with_num_layers(h2_vqe_circuit, 5, 1)
    test_metric_tensor_scaling_with_num_layers(BarrenPlateauCircuit, 5, 1)
    print("All tests passed")
