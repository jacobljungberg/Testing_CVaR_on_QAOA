import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest
import qiskit
from vqa import transforms
from vqa.templates.examples import BarrenPlateauCircuit, h2_vqe_circuit, lih_vqe_circuit
from vqa.transpiler import transpile_circuit
from vqa.transpiler.architectures import FakeChalmers9
from vqa.transpiler.transpile import (
    _from_pennylane_to_qiskit,
    _from_qiskit_to_pennylane,
)


@pytest.mark.parametrize(
    "circuit_ansatz, num_layers, seed, Backend",
    [
        (BarrenPlateauCircuit, 1, 1, FakeChalmers9),
        (h2_vqe_circuit, 1, 1, FakeChalmers9),
    ],
)
def test_transpile(circuit_ansatz, num_layers, seed, Backend):

    backend = Backend()
    template_circuit = circuit_ansatz(num_layers=1)
    params = template_circuit.init(rng_key=seed)

    dev = qml.device("default.qubit", wires=template_circuit.wires)

    @qml.qnode(dev)
    def circuit(params):
        template_circuit(params)
        return qml.expval(template_circuit.H)

    exp_val_before = circuit(params)
    # print(exp_val_before, "exp_val_before")

    dev = qml.device("qiskit.aer", wires=template_circuit.wires)

    @qml.qnode(dev)
    def circuit(params):
        template_circuit(params)
        return qml.expval(template_circuit.H)

    transpiled_circuit = transpile_circuit(
        template_circuit,
        backend=backend,
        params=params,
        wires=template_circuit.wires,
    )

    dev = qml.device("default.qubit", wires=range(backend.n_qubits))

    @qml.qnode(dev)
    def circuit(params):
        transpiled_circuit(params)
        return qml.expval(template_circuit.H)

    exp_val_after = circuit(params)
    # print(circuit(params), "transpiled circuit")
    # print(qml.draw(circuit)(params))
    # print(params)


@pytest.mark.parametrize(
    "circuit_ansatz, num_layers, seed",
    [
        (BarrenPlateauCircuit, 1, 1),
        # (h2_vqe_circuit, 2, 1),
    ],
)
def test_pennylane_to_qiskit(circuit_ansatz, num_layers, seed):
    ansatz = circuit_ansatz(num_layers=num_layers, num_qubits=4)
    params = ansatz.init(seed)

    # calculate exp val of pennylane circuit
    dev = qml.device("default.qubit", wires=ansatz.wires)

    @qml.qnode(dev)
    def circuit(params):
        ansatz(params)
        return qml.expval(ansatz.H)

    exp_val_before = circuit(params)

    parametrized_circuit = _from_pennylane_to_qiskit(ansatz, params, ansatz.wires)

    return parametrized_circuit, params, exp_val_before


@pytest.mark.parametrize(
    "circuit_ansatz, num_layers, seed",
    [
        (BarrenPlateauCircuit, 1, 1),
        # (h2_vqe_circuit, 2, 1),
    ],
)
def test_qiskit_to_pennylane(circuit_ansatz, num_layers, seed):
    num_qubits = 4
    qiskit_circuit, params, exp_val_before = test_pennylane_to_qiskit(
        circuit_ansatz, num_layers, seed
    )

    ansatz = circuit_ansatz(num_layers=num_layers, num_qubits=num_qubits)
    params = ansatz.init(seed)

    new_ansatz = _from_qiskit_to_pennylane(qiskit_circuit)

    # currently ugly
    ansatz.update_circuit_ansatz(new_ansatz, num_qubits)
    ansatz._circuit_ansatz = new_ansatz

    circuit = transforms.exp_val(ansatz)
    exp_val_after = circuit(params)

    # print(qml.draw(circuit)(params))

    assert np.isclose(
        exp_val_before, exp_val_after
    ), "exp val before and after not the same"


if __name__ == "__main__":
    # pytest libs/vqa/src/vqa/tests/transpiler_test.py

    test_transpile(h2_vqe_circuit, 1, 1, FakeChalmers9)
    test_transpile(BarrenPlateauCircuit, 1, 1, FakeChalmers9)

    test_pennylane_to_qiskit(BarrenPlateauCircuit, 1, 1)
    # test_pennylane_to_qiskit(h2_vqe_circuit, 1, 1)

    test_qiskit_to_pennylane(BarrenPlateauCircuit, 1, 1)
    # test_qiskit_to_pennylane(h2_vqe_circuit, 1, 1)
    print("All tests passed!")
