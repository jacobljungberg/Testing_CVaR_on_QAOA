import numpy as np
import pennylane as qml
import pytest
from vqa import transforms
from vqa.templates.examples import (
    BarrenPlateauCircuit,
    h2_simple_vqe_circuit,
    h2_vqe_circuit,
    h2o_vqe_circuit,
    h4_vqe_circuit,
    lih_vqe_circuit,
    maxcut_qaoa_circuit,
    tsp_qaoa_circuit,
)


def test_circuit_1(x: float = None):
    qml.RX(x, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])


@pytest.mark.parametrize(
    "params, circuit, H",
    [
        (0.5, test_circuit_1, qml.PauliY(0)),
        (1.0, test_circuit_1, qml.PauliY(0)),
    ],
)
def exp_val_test(params, circuit, H):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def qml_exp_val(x):
        circuit(x)
        return qml.expval(H)

    fun = transforms.exp_val(circuit, H, wires=range(2))
    exp_val = fun(params)

    exp_val_qml = qml_exp_val(params)

    assert np.isclose(exp_val_qml, exp_val)


# @pytest.mark.parametrize(
#     "params, circuit",
#     [
#         (0.5, test_circuit_1),
#         (1.0, test_circuit_1),
#         (1.0, test_circuit_1),
#     ],
# )
# def vn_entropy_simple_test(params, circuit):
#     dev = qml.device("default.qubit", wires=2)

#     @qml.qnode(dev)
#     def qml_vn_entropy(x):
#         circuit(x)
#         return qml.vn_entropy(wires=[0])

#     fun = transforms.vn_entropy(circuit, wires=range(2))
#     vn_entropy = fun(params)

#     vn_entropy_qml = qml_vn_entropy(params)

#     assert np.isclose(vn_entropy, vn_entropy_qml)


# @pytest.mark.parametrize(
#     "circuit",
#     [
#         (h2_vqe_circuit),
#         (BarrenPlateauCircuit),
#         (lih_vqe_circuit),
#         (h4_vqe_circuit),
#     ],
# )
# def vn_entropy_test(circuit):

#     circ = circuit()
#     params = circ.init()

#     dev = qml.device("default.qubit", wires=circ.wires)

#     @qml.qnode(dev)
#     def qml_vn_entropy(x):
#         circ(x)
#         return qml.vn_entropy(wires=[0])

#     fun = transforms.vn_entropy(circ, wires=circ.wires)

#     # randomly perturb params
#     for i in range(5):
#         params = params + np.random.normal(0, 0.1, params.shape)
#         vn_entropy = fun(params)
#         vn_entropy_qml = qml_vn_entropy(params)
#         assert np.isclose(vn_entropy, vn_entropy_qml)


if __name__ == "__main__":
    # pytest libs/vqa/src/vqa/tests/transforms_test.py
    exp_val_test(0.5, test_circuit_1, qml.PauliY(0))
    exp_val_test(1.5, test_circuit_1, qml.PauliY(0))

    # vn_entropy_simple_test(0.5, test_circuit_1)

    # vn_entropy_test(h2_vqe_circuit)
    # vn_entropy_test(h4_vqe_circuit)
