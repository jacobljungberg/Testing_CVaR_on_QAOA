import pennylane as qml
from vqa.qaoa.mixer_h import x_mixer
from vqa.templates.circuits import QAOACircuit
from vqa.templates.state_preparation import Plus
from vqa.transforms import exp_val
from vqa.utils.maxcut_utils import get_maxcut_graph


def maxcut_qaoa_circuit(
    num_layers: int = 1, num_nodes: int = 4, seed: int = 0
) -> QAOACircuit:
    """Generate a QAOA circuit for the MaxCut problem.

    Args:
        num_layers (int, optional): The number of layers of the QAOA circuit.
            Defaults to 1.
        num_nodes (int, optional): The number of nodes in the MaxCut graph.
            Defaults to 4.
        seed (int, optional): Random seed. Defaults to 1.

    Returns:
        circuit (QAOACircuit): The quantum circuit class.
    """
    graph = get_maxcut_graph(num_nodes, seed=seed)
    cost_h, _ = qml.qaoa.maxcut(graph)
    num_qubits = len(cost_h.wires)
    mixer_h = x_mixer(num_qubits)
    initial_state = Plus(num_qubits)
    circuit = QAOACircuit(
        H=cost_h, initial_state=initial_state, mixer_h=mixer_h, num_layers=num_layers
    )

    return circuit
