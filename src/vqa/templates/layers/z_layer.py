from typing import Iterable, Union

import pennylane as qml
from pennylane.wires import Wires


def z_mixer(wires: Union[Iterable, Wires]):
    r"""Creates a basic Pauli-X mixer Hamiltonian.

    This Hamiltonian is defined as:

    .. math:: H_M \ = \ \displaystyle\sum_{i} X_{i},

    where :math:`i` ranges over all wires, and :math:`X_i`
    denotes the Pauli-Y operator on the :math:`i`-th wire.

    Args:
        wires (Iterable or Wires): The wires on which the Hamiltonian is applied

    Returns:
        Hamiltonian: Mixer Hamiltonian

    **Example**

    The mixer Hamiltonian can be called as follows:

    >>> from pennylane import qaoa
    >>> wires = range(3)
    >>> mixer_h = qaoa.x_mixer(wires)
    >>> print(mixer_h)
      (1) [Y0]
    + (1) [Y1]
    + (1) [Y2]
    """

    wires = Wires(wires)

    coeffs = [1 for w in wires]
    obs = [qml.PauliZ(w) for w in wires]

    H = qml.Hamiltonian(coeffs, obs)
    # store the valuable information that all observables are in one commuting group
    H.grouping_indices = [list(range(len(H.ops)))]
    return H
