import pennylane as qml


def vn_entropy(
    circ,
    interface: str = "autograd",
    diff_method: str = "best",
    wires: list = None,
    subsystem: list = [0],
):
    """
    Transform a circuit into a callable function returning the von Neumann entropy.

    Args:
        circ (AbstractCircuit): A quantum circuit.
        interface (str, optional): The interface for calculations.
            Defaults to "autograd".
        diff_method (str, optional): _description_. Defaults to "best".
        wires (list, optional): The subsytem to use. Defaults is [0], meaning
            we consider the subsystem of the qubit located at register 0.
        subsystem (list, optional): The subsytem to use. Defaults is [0].

    Returns:
        fun (Callable): A callabale function computing the von Neumann entropy
            for a given subsystem.
    """

    if wires is None:
        if hasattr(circ, "wires"):
            wires = circ.wires
        else:
            raise ValueError("No wires specified.")

    def _transform(x):
        def _fun(x):
            circ(x)
            return qml.vn_entropy(wires=subsystem)

        return _fun(x)

    dev_qubit = qml.device("default.qubit", wires=wires)
    return qml.QNode(_transform, dev_qubit, interface, diff_method=diff_method)
