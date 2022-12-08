import numpy as np
import pennylane as qml
import pytest
from vqa import transforms
from vqa.qaoa.mixer_h import x_mixer
from vqa.templates.circuits import QAOACircuit, Sequential
from vqa.templates.layers import Pauli_X
from vqa.templates.state_preparation import Plus
from vqa.utils.maxcut_utils import get_maxcut_graph


@pytest.mark.parametrize(
    "params",
    [
        ([[4.37604495], [1.79786647], [0.71267486], [1.73200643]]),
    ],
)
def test_sequential(params):
    # Get Hamiltonian of optimization problem, in this case maxcut.
    graph = get_maxcut_graph(5, seed=123)
    H, _ = qml.qaoa.maxcut(graph)
    n_qubits = len(H.wires)
    wires = H.wires

    # Initialize the circuit with Sequential.
    circ = Sequential(
        [
            # state preparation is always the first layer and unparametrized!
            Plus(n_qubits),
            H,
            Pauli_X(wires),  # should this be a Hamiltonian or a circuit? Can be both
            H,
            Pauli_X(wires),
        ]
    )
    loss_fun_sequential = transforms.exp_val(circ, H, interface="jax")

    # initialize the circuit with QAOACircuit.
    mixer_h = x_mixer(n_qubits)
    circ_qaoa = QAOACircuit(
        H=H, initial_state=Plus(n_qubits), mixer_h=mixer_h, num_layers=2
    )
    # loss_fun = jit(transforms.exp_val(circuit_qaoa, H, interface="jax"))
    loss_fun_qaoa = transforms.exp_val(circ_qaoa, H, interface="jax")
    params = np.reshape(params, (-1,))

    # exp_val_sequential = loss_fun_sequential(params)
    # exp_val_qaoa = loss_fun_qaoa(params)
    # assert loss_fun_sequential(params) == loss_fun_qaoa(
    # params
    # ), "loss_fun_sequential and loss_fun_qaoa should be equal"


if __name__ == "__main__":
    test_sequential([[4.37604495], [1.79786647], [0.71267486], [1.73200643]])
