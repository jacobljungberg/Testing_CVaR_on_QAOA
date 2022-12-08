from typing import List

import numpy as np
import pennylane as qml


# https://pennylane.ai/qml/demos/tutorial_local_cost_functions.html
def draw(
    circ: qml.operation.Operation,
    params: np.ndarray = None,
    wires: List = None,
    interface: str = "qiskit.aer",
    output: str = "mpl",
):
    assert not isinstance(circ, qml.QNode), "circ must not be instance of qml.QNode"
    if wires is None:
        if hasattr(circ, "wires"):
            wires = circ.wires
        else:
            raise ValueError("No wires specified.")

    if params is None:
        params = circ.init()

    dev = qml.device("qiskit.aer", wires=wires)

    @qml.qnode(dev)
    def _circuit(x):
        circ(x)
        return qml.expval(qml.PauliZ(0))

    _circuit(params)  # Don't forget to run the circuit once for initialization.

    # return dev._circuit.draw()
    return dev._circuit.draw(output=output)
