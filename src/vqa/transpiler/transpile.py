from dataclasses import astuple, dataclass
from typing import List, Tuple

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import qiskit
from pennylane.operation import AnyWires, Operation
from pennylane_qiskit import load, load_qasm
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.fake_provider import FakeBackend
from qiskit.visualization import plot_circuit_layout, plot_gate_map, plot_histogram
from vqa import transforms
from vqa.transpiler.architectures import (
    FakeChalmers4,
    FakeChalmers6,
    FakeChalmers9,
    FakeChalmers20,
)

# qiskit and pennylane do not always use the same name convention for quantum gates
qml_to_qiskit = {"CNOT": "CX"}
qiskit_to_qml = {"CX": "CNOT"}


@dataclass
class Gate:
    """
    A class to represent a gate in a quantum circuit.

    Attributes
    ----------
    gate_id : str
        The name of the gate.
    wires : list
        The wires the gate acts on.
    param : list
        The parameter of the gate.
    trainable: bool
        Whether the gate is trainable or not.
    """

    gate_id: str
    wires: List[int]
    param: float = None
    trainable: bool = False

    def __iter__(self):
        return iter(astuple(self))


def transpile_circuit(
    circuit: qml.operation.Operation,
    backend: FakeBackend,
    params: np.ndarray,
    wires: List[int] = None,
    # H: qml.Hamiltonian = None,
    basis_gates: List = ["rz", "rx", "h", "x", "z", "iswap"],
    layout_method: str = "sabre",
    routing_method: str = "sabre",
    optimization_level: int = 3,
) -> qml.operation.Operation:
    """Transpile a pennylane circuit to satisfy the architecture constraints of the
    Chalmers backends.


    Args:
        circuit (qml.QNode): The pennylane circuit to transpile.
        backend (FakeBackend): The target backend.
        params (np.ndarray, optional): The parameters of the circuit. Defaults to None.
        basis_gates (List, optional): The list of native gates. Defaults to None.

    Returns:
        CircuitTemplate: The transpiled circuit. Not wrapped in a qnode!
    """

    # check if circuit is instance of qml.QNode
    assert not isinstance(
        circuit, qml.QNode
    ), "circuit must not be a qml.QNode but a circuit template. Do not wrap it into a QNode."

    if hasattr(circuit, "wires"):
        wires = circuit.wires
    else:
        raise ValueError("Either provide the wires or use the AbstractCircuit class.")

    ######################################################################
    # 1. convert to qiskit
    ######################################################################
    circuit_qiskit = _from_pennylane_to_qiskit(circuit, params, wires)

    ######################################################################
    # 2. transpile to chalmers backend
    ######################################################################
    transpiled_circuit_qiskit = transpile(
        circuit_qiskit,
        backend,
        optimization_level=optimization_level,
        basis_gates=basis_gates,
        layout_method=layout_method,
        routing_method=routing_method,
    )

    ######################################################################
    # 3. convert back to pennylane
    ######################################################################
    circ = _from_qiskit_to_pennylane(transpiled_circuit_qiskit)

    return circ


def _from_pennylane_to_qiskit(
    ansatz: qml.operation.Operation, params: pnp.ndarray, wires: List[int] = None
) -> qiskit.QuantumCircuit:
    """
    Convert a pennylane circuit to a qiskit circuit. Converting also includes adding
    tunable parameters to the circuit. We detect if a gate is parametrized by checking
    if its parameter value is greater than 100000.

    """
    assert not isinstance(
        ansatz, qml.QNode
    ), "ansatz must not be a qml.QNode but a circuit template. Do not wrap it into a QNode."
    assert wires is not None, "wires must be specified."

    ######################################################################
    # Add high values to the parameters of the circuit to detect if a gate is parametrized.
    ######################################################################

    qiskit_circuit = _get_qiskit_circuit(ansatz, params, wires)
    dag = circuit_to_dag(qiskit_circuit)

    ######################################################################
    # Create a parametrized qiskit circuit
    ######################################################################

    parametrized_circuit = QuantumCircuit(
        qiskit_circuit.num_qubits, qiskit_circuit.num_clbits
    )

    # parametrized_circuit = _create_parametrized_circuit(dag, qiskit_circuit, params)

    # TODO split loop into two functions to make it more readable.
    param_id = [str(f"param_{i}") for i in range(len(params))]
    param_counter = 0
    for node in dag.topological_op_nodes():
        if len(node.op.params):
            if node.op.params[0] > 1000.0:  # parametrizable gate
                theta = Parameter(param_id[param_counter])
                wires = [node.qargs[i].index for i in range(len(node.qargs))][0]
                getattr(parametrized_circuit, node.name)(theta, wires)
                param_counter += 1
            else:
                parametrized_circuit.append(node.op, node.qargs, node.cargs)
        else:
            parametrized_circuit.append(node.op, node.qargs, node.cargs)

    return parametrized_circuit


def _get_qiskit_circuit(
    ansatz: qml.operation.Operation, params: np.ndarray, wires: List[int] = None
) -> QuantumCircuit:
    """Generating a qiskit circuit from a pennylane circuit template. We offset
    the variational parameters by a large scalar to detect if a gate is parametrized.

    Args:
        ansatz (qml.operation.Operation): The circuit ansatz.
        params (np.ndarray): The parameters of the circuit.
        wires (List[int], optional): The number of wires. Defaults to None.

    Returns:
        QuantumCircuit: A qiiskit circuit.
    """

    params = pnp.array(params) * 10000000.0

    dev = qml.device("qiskit.aer", wires=wires)

    @qml.qnode(dev)
    def circuit(params):
        ansatz(params)
        return qml.expval(qml.PauliZ(0))

    circuit(params)  # run circuit to get the qiskit circuit

    qiskit_circuit = circuit.device._circuit
    qiskit_circuit.remove_final_measurements()
    return qiskit_circuit


def _from_qiskit_to_pennylane(
    qiskit_circuit: qiskit.QuantumCircuit,
) -> qml.operation.Operation:
    """Convert a Qiskit circuit to a Pennylane circuit.
        Using the Open Qasm standard does not work, since Open Qasm does not support
        parametrized quantum circuits.

        To circumvent this, we iterate through the circuit and add the gates manually.
        We distinguish between variatonally parametrized and fixed parametrized gateso

    Args:
        circuit (qiskit.QuantumCircuit): The qiskit circuit to convert.

    Returns:
        qml.operation.Operation: The pennylane circuit template. Not wrapped in a qnode!
    """

    gate_list = _get_gate_list(qiskit_circuit)
    trainable_gates = [gate for gate in gate_list if gate.trainable]

    # For debugging
    # for gate in gate_list:
    # print(gate)
    # get gates from gate list for which trainable is true
    # print(len(trainable_gates), "trainable gates ")

    def circuit_ansatz(params: np.ndarray) -> qml.operation.Operation:
        assert len(params) == len(
            trainable_gates
        ), f"Number of trainable gates:{len(trainable_gates)}, but got {len(params)}."
        params_counter = 0
        for (gate_id, wires, param, trainable) in gate_list:
            # Some gates have a different name in qiskit and pennylane
            if gate_id in qiskit_to_qml:
                gate_id = qiskit_to_qml[gate_id]

            if param is None:
                if trainable:
                    getattr(qml, gate_id)(params[params_counter], wires=wires)
                    params_counter += 1
                else:
                    getattr(qml, gate_id)(wires=wires)
            else:
                getattr(qml, gate_id)(param, wires=wires)

    return circuit_ansatz


def _get_gate_list(qiskit_circuit: qiskit.QuantumCircuit) -> List[Gate]:
    """Generate a list of gates from a qiskit circuit.

    Args:
        qiskit_circuit (qiskit.QuantumCircuit): The qiskit circuit.

    Returns:
        List[Gate]: Returns a list of gates. Each gate contains information about:
            gate_id, param, wires, trainable.
    """
    qiskit_circuit.remove_final_measurements()
    dag = circuit_to_dag(qiskit_circuit)

    gate_list = []

    for node in dag.topological_op_nodes():

        gate_id = node.op.name.upper()
        gate_id = ["Hadamard" if gate_id == "H" else gate_id][0]
        wires = [node.qargs[i].index for i in range(len(node.qargs))]

        params, trainable = _inspect_gate(node.op)

        gate_list.append(Gate(gate_id, wires, params, trainable))

    return gate_list


def _inspect_gate(op) -> Tuple[float, bool]:
    """Ugly ass function to inspect the gate, figure out if parametrizable and
    clean up parameter to be a float.

    Args:
        op : Qiskit operation.

    Returns:
        Tuple[float, bool]: Returns a tuple of the parameter and a boolean.
    """
    trainable = False
    params = op.params
    if len(params) == 0:
        params = None
    if params is not None:
        params = op.params[0]
        trainable = _check_parametrized_gate(op)
        if trainable:
            params = None

    if params is not None:
        if isinstance(params, ParameterExpression):
            params = float(params._symbol_expr)
        assert isinstance(params, float), "params must be a float."

    return params, trainable


def _check_parametrized_gate(op) -> bool:
    """Check if a gate is parametrized.
    A gate is parametrized if its a qiskit.circuit.Parameter object.

    Args:
        op (qiskit.circuit.Instruction): The gate to check.

    Returns:
        bool: True if the gate is variationally parametrized, False otherwise.

    """
    _bool = False
    if len(op.params) > 0:
        if isinstance(op.params[0], Parameter):
            _bool = True
        elif isinstance(op.params[0], ParameterExpression):
            for e in op.params[0].parameters:
                if isinstance(e, Parameter):
                    _bool = True
    return _bool
