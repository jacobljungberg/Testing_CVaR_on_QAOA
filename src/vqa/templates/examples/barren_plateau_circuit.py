from vqa.templates.circuits import BarrenPlateauCircuit


def barren_plateau_circuit(num_qubits: int, num_layers: int) -> BarrenPlateauCircuit:
    """Return a BarrenPlateauCircuit instance with the specified number of qubits and layers.

    Args:
    - num_qubits (int): The number of qubits to use in the circuit.
    - num_layers (int): The number of layers in the circuit.

    Returns:
    - BarrenPlateauCircuit: An instance of the BarrenPlateauCircuit class with the specified number of qubits and layers.

    Example:
    >>> circuit = barren_plateau_circuit(num_qubits=2, num_layers=3)
    >>> print(circuit.num_qubits)
    2
    >>> print(circuit.num_layers)
    3
    >>> params = circuit.init()
    >>> print(params.shape)
    (6,)

    """
    return BarrenPlateauCircuit(num_layers=num_layers, num_qubits=num_qubits)
