import pennylane as qml


def state_vector(circ, n_qubits: int, interface: str = "autograd"):
    """Transform a circuit into a function returning the statevector.

    Args:
        circ (): A quantum circuit.
        n_qubits (int): Number of qubits in the quantum circuit.
        interface (str): Interface.

    Returns:
        func: A callabale function.
    """

    def _transform(x):
        def _fun(x):
            circ(x)
            return qml.state()

        return _fun(x)

    dev_qubit = qml.device("default.qubit", wires=circ.wires)
    return qml.QNode(_transform, dev_qubit, interface)
