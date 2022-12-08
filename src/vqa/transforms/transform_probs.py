from typing import Callable, List

import pennylane as qml


def probabilities(
    circ, interface: str = "autograd", diff_method: str = "best", wires: List = None
) -> Callable:
    """Transform a circuit into a callable function returning the expecation value.

    Args:
        circ (): A quantum circuit.
        hamiltonian (qml.Hamiltonian): A Hamiltonian.
        n_qubits (int): Number of qubits in the quantum circuit.
        interface (str): Interface.

    Returns:
        func: A callabale function.
    """

    if wires is None:
        if hasattr(circ, "wires"):
            wires = circ.wires
        else:
            raise ValueError("No wires specified.")

    def _transform(x):
        def _fun(x):
            circ(x)
            return qml.probs(wires=wires)

        return _fun(x)

    dev_qubit = qml.device("default.qubit", wires=wires)
    return qml.QNode(
        _transform,
        dev_qubit,
        gradient_fn=qml.gradients.param_shift,
        interface=interface,
        diff_method=diff_method,
    )
