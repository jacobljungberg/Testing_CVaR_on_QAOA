from typing import List

import pennylane as qml
from vqa.templates.circuits import AbstractCircuit


def exp_val(
    circ: AbstractCircuit,
    H: qml.Hamiltonian = None,
    wires: List[int] = None,
    interface: str = "autograd",
    diff_method: str = "best",
    device: str = "default.qubit",
) -> qml.QNode:
    """Transform a circuit into a callable function returning the expecation value.


    Args:
        circ: A quantum circuit.
        H: A Hamiltonian.
        wires: The wires of the circuit.
        interface (str, optional): The interface for calculations. Defaults to "autograd".
        diff_method (str, optional): The method to calculate the gradients. Defaults to "best".
        device (str, optional): The device to use. Defaults to default.qubit.

    Returns:
        fun (qml.QNode): A callabale function computing the expectation value.
    """

    if wires is None:
        if hasattr(circ, "wires"):
            wires = circ.wires
        else:
            raise ValueError("No wires specified.")

    if H is None:
        if hasattr(circ, "H"):
            H = circ.H
        else:
            raise ValueError("No Hamiltonian specified.")

    dev = qml.device(device, wires=wires)

    @qml.qnode(dev, diff_method=diff_method, interface=interface)
    def circuit(params):
        circ(params)
        return qml.expval(H)

    return circuit
