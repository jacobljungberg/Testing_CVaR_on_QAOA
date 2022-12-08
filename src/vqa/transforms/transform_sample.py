import pennylane as qml


def sample(
    circ, shots: int = 8192, interface: str = "autograd", diff_method: str = "best"
):
    """Sample from a quantum circuit.

    Args:
        circ (): A quantum circuit.
        hamiltonian (qml.Hamiltonian): A Hamiltonian.
        n_qubits (int): Number of qubits in the quantum circuit.
        interface (str): Interface.

    Returns:
        func: A callabale function.
    """

    def _transform(x):
        def _fun(x):
            circ(x)
            return qml.sample()

        return _fun(x)

    dev_qubit = qml.device("default.qubit", shots=shots, wires=circ.wires)
    return qml.QNode(_transform, dev_qubit, interface, diff_method=diff_method)
