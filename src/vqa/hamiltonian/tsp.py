import pennylane as qml

# TODO add constant terms to the Hamiltonian


def get_qubit_index(i: int, j: int, N: int):
    return i * N + j
    # return np.ravel_multi_index(np.array([[i],[j]]), (N,N))[0]


def tsp_hamiltonian(ins, weights):
    N = ins.shape[0] - 1
    A, B = weights

    penalty_h_1 = _get_penalty_hamiltonian_1(N, A)
    penalty_h_2 = _get_penalty_hamiltonian_2(N, A)
    opt_h = _get_optimization_hamiltonian(ins, N, B)
    return penalty_h_1 + penalty_h_2 + opt_h


def _get_penalty_hamiltonian_1(N, A):

    coeffs = [A * (2 - N) for v in range(N) for j in range(N)]
    obs = [qml.PauliZ(get_qubit_index(v, j, N)) for v in range(N) for j in range(N)]
    h_z = qml.Hamiltonian(coeffs, obs)

    coeffs = [A for v in range(N) for j in range(N) for j_ in range(N) if j < j_]
    obs = [
        qml.PauliZ(get_qubit_index(v, j, N)) @ qml.PauliZ(get_qubit_index(v, j_, N))
        for v in range(N)
        for j in range(N)
        for j_ in range(N)
        if j < j_
    ]
    h_zz = qml.Hamiltonian(coeffs, obs)

    return h_zz + h_z


def _get_penalty_hamiltonian_2(N, A):

    coeffs = [A * (2 - N) for v in range(N) for j in range(N)]
    obs = [qml.PauliZ(get_qubit_index(v, j, N)) for v in range(N) for j in range(N)]
    h_z = qml.Hamiltonian(coeffs, obs)

    coeffs = [A for _ in range(N) for v in range(N) for v_ in range(N) if v < v_]
    obs = [
        qml.PauliZ(get_qubit_index(v, j, N)) @ qml.PauliZ(get_qubit_index(v_, j, N))
        for j in range(N)
        for v in range(N)
        for v_ in range(N)
        if v < v_
    ]
    h_zz = qml.Hamiltonian(coeffs, obs)

    return h_zz + h_z


def _get_optimization_hamiltonian(ins, N, B):

    coeffs = [
        B * ins[u, v] for u in range(N) for v in range(N) if u < v for j in range(N - 1)
    ]
    obs = [
        qml.PauliZ(get_qubit_index(u, j, N))
        for u in range(N)
        for v in range(N)
        if u < v
        for j in range(N - 1)
    ]
    h_z_1 = qml.Hamiltonian(coeffs, obs)

    obs = [
        qml.PauliZ(get_qubit_index(u, j + 1, N))
        for u in range(N)
        for v in range(N)
        if u < v
        for j in range(N - 1)
    ]
    h_z_2 = qml.Hamiltonian(coeffs, obs)

    obs = [
        qml.PauliZ(get_qubit_index(u, j, N)) @ qml.PauliZ(get_qubit_index(v, j + 1, N))
        for u in range(N)
        for v in range(N)
        if u < v
        for j in range(N - 1)
    ]
    h_zz = qml.Hamiltonian(coeffs, obs)

    return h_z_1 + h_z_2 + h_zz
