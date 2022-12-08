import random
from typing import List

import numpy as np
import pennylane as qml
from vqa.templates.state_preparation import Plus

# def Sequential(layers:List)-> Tuple[circ, params]:


# class Result(NamedTuple):
#     """Holds a pure function and the initial parameter of the pure function.
#     Attributes:
#     circuit: A pure function: ``params = circuit(params)``
#     params: The initial parameter for the function: params
#     """

#     # Args: [Optional[PRNGKey], ...]
#     circuit: Callable[..., List]

#     # Args: [Params, Optional[PRNGKey], ...]
#     params: List


class Sequential:
    """This module takes in a list of layers and returns a quantum circuit that
    executes the layers in order.

    Args:
        layers (List): the list of quantum layers to execute.

    Returns:
        qml.XXX : the quantum circuit.
    """

    def __init__(
        self,
        layers: List[
            qml.Hamiltonian,
        ],
    ):
        """Initialize the Sequential module and return a executable loss function
        and a list of lists with initial parameters.

        Args:
            layers (List): _description_
        """
        self.layers = layers
        self.state_preparation = layers[0]
        # print(self.state_preparation)

        self.n_qubits = len(layers[-1].wires)

        self.params = []
        for _ in self.layers[1:]:  # first element is state preparation
            # layer_params = [np.random.random() for _ in range(len(H.wires))]
            layer_params = [np.random.random()]
            self.params.append(layer_params)

    @property
    def wires(self):
        return self.layers[-1].wires

    def __repr__(self) -> str:
        # print(qml.draw(self, expansion_strategy='device')(np.zeros((2 * self.layer))))
        return f"Quantum circuit, n_qubits: {self.n_qubits}, {self.layers}"

    def __call__(self, x):
        """This module takes in a list of layers and returns a quantum circuit that
        executes the layers in order.

        Args:
            layers (List): the list of quantum layers to execute.

        Returns:
            qml.XXX : the quantum circuit.
        """

        def _circ(x):
            self.state_preparation()
            for H, params in zip(self.layers[1:], x):
                qml.ApproxTimeEvolution(H, *params, 1)

        # return Result(circuit=_circ(x), params=self.params)
        # return _circ(x), self.params
        return _circ(x)
