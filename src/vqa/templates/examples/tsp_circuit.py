from typing import List

from vqa.hamiltonian import tsp_hamiltonian
from vqa.instance import get_tsp_instance
from vqa.qaoa.mixer_h import x_mixer
from vqa.templates.circuits import QAOACircuit
from vqa.templates.state_preparation import Plus
from vqa.transforms import exp_val


def tsp_qaoa_circuit(
    num_layers: int = 1,
    num_cities: int = 3,
    list_weights: List = [1.0, 1.2],
    seed: int = 0,
) -> QAOACircuit:
    """Generate a QAOA circuit for the TSP problem.

    Args:
        num_layers (int, optional): The number of layers of the QAOA circuit.
            Defaults to 1.
        num_cities (int, optional): The number of cities for the TSP problem.
            Defaults to 3.
        list_weights (List, optional): The weights balancing the optimization
            and constraint terms in the cost hamiltonian. Defaults to [1.0, 1.2].
        seed (int, optional): The random seed. Defaults to 0.

    Returns:
        QAOACircuit: The quantum circuit class.
    """

    ins = get_tsp_instance(num_cities, seed=seed)
    cost_h = tsp_hamiltonian(ins, list_weights)
    n_qubits = len(cost_h.wires)
    mixer_h = x_mixer(n_qubits)
    initial_state = Plus(n_qubits)
    circuit = QAOACircuit(
        H=cost_h, initial_state=initial_state, mixer_h=mixer_h, num_layers=num_layers
    )

    return circuit
