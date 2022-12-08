import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from beartype import beartype
import pennylane as qml
from scipy.linalg import eig, inv

from vqa.utils.utils import get_all_bitstrings
from vqa.utils.decorator import memory

from sympy import (
    Sum,
    symbols,
    Indexed,
    lambdify,
    expand,
    IndexedBase,
    Idx,
    simplify,
    Poly,
)


def simplify_H(H):
    return simplify(expand(H.doit()))


@beartype
def add_gate(index: np.ndarray):
    # def add_gate(index:np.ndarray, coeff:float):
    if len(index) == 1:
        return qml.PauliZ(index)  # * coeff
    elif len(index) == 2:
        return qml.PauliZ(index[0]) @ qml.PauliZ(index[1])  # * coeff


@beartype
def get_ising_hamiltonian(interactions: np.ndarray, coefficients: np.ndarray):
    # identity, Is this correct?
    index_identity = np.where(interactions == 0)[0]
    obs = [qml.Identity(i) for i in range(interactions.shape[0] - 1)]
    len_inter = interactions.shape[0] - 1
    H_identity = qml.Hamiltonian(np.ones(len_inter) * coefficients[0] / len_inter, obs)
    obs = []
    for inter, coeff in zip(interactions[1::], coefficients[1::]):
        index_ = np.where(inter == 1)[0]
        obs.append(add_gate(index_))
    H_z_zz = qml.Hamiltonian(coefficients[1::], obs)
    return H_z_zz + H_identity


def get_coeffs_monomials(H) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = np.array(Poly(H).coeffs(), dtype=np.float64)
    interactions = np.array(Poly(H).monoms())
    interactions[interactions == 2] = 0  # remove all Z**2 since they are Identity.
    return coeffs, interactions


@beartype
def compress_interactions(
    interactions: np.ndarray, coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    dict_inter = {}
    for i, inter in enumerate(interactions):
        if tuple(inter) in dict_inter:
            dict_inter[tuple(inter)] += coeffs[i]
        else:
            dict_inter[tuple(inter)] = coeffs[i]

    interactions = np.zeros((len(dict_inter), interactions.shape[1]))
    i = 0
    for key in dict_inter.keys():
        interactions[i, :] = key
        i += 1
    return interactions.astype(int), np.array([*dict_inter.values()])


@beartype
def get_eigenvalues_hamiltonian_Z_basis(hamiltonian: qml.Hamiltonian) -> np.ndarray:
    """Get the eigenvalues of the cost hamiltonian in Z basis.

    Args:
        hamiltonian (qml.Hamiltonian): The cost hamiltonian.

    Returns:
        eigenvalues (np.ndarray): An array with the eigenvalues of the cost hamiltonian.
    """
    n = len(hamiltonian.wires)
    dev = qml.device("default.qubit", wires=range(n), shots=10)

    @qml.qnode(dev)
    def exp_val(b):
        qml.BasisState(b, wires=range(n))
        return qml.expval(hamiltonian)

    bitstrings = get_all_bitstrings(n)

    eigenvalues = np.zeros((len(bitstrings)))
    for i, string in enumerate(bitstrings):
        eigenvalues[i] = exp_val(string)
        # print(f"{i}: {string}: {eigenvalues[i]}")

    return eigenvalues


@beartype
def get_eigenvalues_hamiltonian(H: qml.Hamiltonian) -> np.ndarray:
    """Get the eigenvalues of the cost hamiltonian.

    Args:
        hamiltonian (qml.Hamiltonian): The cost hamiltonian.

    Returns:
        eigenvalues (np.ndarray): An array with the eigenvalues of the cost hamiltonian.
    """
    # if type(H) == qml.Hamiltonian:
    Hmat = qml.utils.sparse_hamiltonian(H)
    Hmat = Hmat.toarray()
    # else:
    # Hmat = H.matrix()

    eigenvalues, eigenvectors = eig(Hmat)
    return eigenvalues


def get_eigenvalues_hermitian(H) -> np.ndarray:
    """Get the eigenvalues of an Hermitian observable.

    Args:
        hermitian (qml.Hamiltonian): The hermitian observable.

    Returns:
        eigenvalues (np.ndarray): An array with the eigenvalues of the cost hamiltonian.
    """
    Hmat = H.matrix()
    eigenvalues, eigenvectors = eig(Hmat)
    return eigenvalues
