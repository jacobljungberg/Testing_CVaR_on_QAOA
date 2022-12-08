import numpy as np
import pennylane as qml
import pytest
from jax import jit
from sympy import Idx, IndexedBase, Sum, symbols
from vqa.hamiltonian.utils_hamiltonian import simplify_H
from vqa.qaoa.mixer_h import circular_xy_mixer, row_mixer, row_mixer_2, x_mixer
from vqa.templates.circuits import QAOACircuit
from vqa.templates.state_preparation import Column, Identity, Plus


def sympy_test_hamiltonian():
    """A Hamiltonian enforcing a matrix to contain a single excitation per row and column.
        An allowed solution with zero energy.
        [[1,0,0],
         [0,1,0],
         [0,0,1]]

        A violating solution with positive energy.
        [[0,1,0],
         [0,1,0],
         [0,0,1]]

    Returns:
        _type_: the Hamiltonian in the form of a sympy expression.
    """
    # define variables
    i, j, v = symbols("i j v", integer=True)
    i, j, v = Idx(i), Idx(j), Idx(v)
    X = IndexedBase("x")
    Z = IndexedBase("z")
    # problem specific variables
    n_cities = 3
    N = n_cities - 1
    # define Hamiltonian
    penalty_1 = Sum((1 - Sum(X[v, j], (j, 0, N - 1))) ** 2, (v, 0, N - 1))
    penalty_2 = Sum((1 - Sum(X[v, j], (v, 0, N - 1))) ** 2, (j, 0, N - 1))
    # optimization = (1 - Sum(X[v,j], (j,1,n_cities)) )**2
    H = penalty_1 + penalty_2  # + optimization
    # Substitute X with 1/2 (1+Z)
    H = H.doit().subs(
        [(X[v, j], 1 / 2 * (1 - Z[v, j])) for v in range(N) for j in range(N)]
    )
    # simplify
    H = simplify_H(H)
    return H


tsp_9_qubits_interactions = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

tsp_9_qubits_coeffs = np.array(
    [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        -1.0,
        0.5,
        0.5,
        0.5,
        0.5,
        -1.0,
        0.5,
        0.5,
        0.5,
        -1.0,
        0.5,
        0.5,
        0.5,
        0.5,
        -1.0,
        0.5,
        0.5,
        0.5,
        -1.0,
        0.5,
        0.5,
        -1.0,
        0.5,
        0.5,
        0.5,
        -1.0,
        0.5,
        0.5,
        -1.0,
        0.5,
        -1.0,
        1.5,
    ]
)


def H(N):
    def get_qubit_index(i: int, j: int, N: int):
        return i * N + j

    coeffs = [(2 - N) for v in range(N) for j in range(N)]
    obs = [qml.PauliZ(get_qubit_index(v, j, N)) for v in range(N) for j in range(N)]
    h_z = qml.Hamiltonian(coeffs, obs)

    coeffs = [1 for v in range(N) for j in range(N) for j_ in range(N) if j < j_]
    obs = [
        qml.PauliZ(get_qubit_index(v, j, N)) @ qml.PauliZ(get_qubit_index(v, j_, N))
        for v in range(N)
        for j in range(N)
        for j_ in range(N)
        if j < j_
    ]
    h_zz = qml.Hamiltonian(coeffs, obs)

    return h_zz + h_z


@pytest.mark.parametrize(
    "H, initial_state, mixer_h, num_layers",
    [
        (H(2), Identity, circular_xy_mixer, 1),
        (H(2), Column, row_mixer, 1),
        # (H(2), FeasibleSolution, row_mixer_2, 1),
        (H(2), Column, row_mixer, 1),  # special treatment due to list of mixers
        (H(2), Plus, x_mixer, 1),
    ],
)
def test_circuit(
    H: qml.qnode, initial_state: qml.qnode, mixer_h: qml.qnode, num_layers: int
):
    seed = 0
    wires = H.wires
    num_qubits = len(wires)
    mixer_h = mixer_h(num_qubits)
    initial_state = Plus(num_qubits)
    assert mixer_h.wires == H.wires, f"mixer: {mixer_h.wires}, cost: {H.wires}"

    circ = QAOACircuit(
        H=H, initial_state=initial_state, mixer_h=mixer_h, num_layers=num_layers
    )
    init_params = circ.init(seed)
    # print(mixer)

    assert circ.num_qubits == num_qubits
    assert circ.num_layers == num_layers
    # loss_fun = jit(exp_val(circ, H, num_qubits, interface="jax"))
    # variational_parameter = np.zeros((2 * p))
    # loss = loss_fun(variational_parameter)


if __name__ == "__main__":
    test_circuit(H(2), Identity, circular_xy_mixer, 1)
