import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
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

# from vqa.utils.utils import get_all_bitstrings, pairwise
from beartype import beartype as type_check


@type_check
def get_all_bitstrings(n_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get all bitstrings up to n_bits.

    Args:
        n_bits (int): The length of the bitstring.

    Returns:
        bitstrings_binary (np.ndarray): An array of shape (n_permutations, n_bits).
    """
    n_bitstrings = 2**n_bits
    bitstrings_binary = np.zeros(shape=(n_bitstrings, n_bits), dtype=np.int8)
    tf = np.array([False, True])
    for i in range(n_bits):
        j = n_bits - i - 1
        bitstrings_binary[np.tile(np.repeat(tf, 2**j), 2**i), i] = 1

    bitstrings = -2 * bitstrings_binary + 1
    return bitstrings, bitstrings_binary


@type_check
def reduce_H_parity(
    parity_interactions: np.ndarray,
    coeffs: np.ndarray,
    symmetries: Dict,
):
    """
    The interactions are in the parity basis. The basis might be chosen
        depending of the problem at hand.

    1. Updating the coefficients given the symmetries.
    2. Summarize the coefficients for the reduced subspace.

    Args:
        H (): The Hamiltonian in the parity basis.
        symmetries (Dict): The symmetries of the parity basis.

    Returns:
    """

    """
    The interactions are in the parity basis. The basis might be chosen 
        depending of the problem at hand. 
    # 1. transform H into parity basis
    1. Updating the coefficients given the symmetries.
    2. Summarize the coefficients for the reduced subspace.
    Args:
        H (sympy.Matrix): The Hamiltonian in the parity basis.
        symmetries (np.ndarray): The symmetries of the parity basis. 
    Returns:
    """
    n_qubits = parity_interactions.shape[1]
    reduced_registers = [int(key.split("_")[-1]) for key in symmetries.keys()]
    registers = [i for i in range(n_qubits) if i not in reduced_registers]
    measurements = np.array([symmetries[key] for key in symmetries.keys()])
    assert len(reduced_registers) == len(measurements)

    reduced_subspaces = parity_interactions[:, registers]
    eliminated_subspaces = parity_interactions[:, reduced_registers]

    # 1. Updating the coefficients given the measurement results of the symmetries.
    coeffs = update_coeffs(coeffs, measurements, eliminated_subspaces)
    # for i in range(len(parity_interactions)):
    # print(f'{i}, {parity_interactions[i]} reduced subspace {reduced_subspaces[i]}, eliminated subspace {eliminated_subspaces[i]}, old coeffs {H['coeffs'][i]}, new coeffs {np.round(coeffs[i],2)},')

    # 2. Summarize the coefficients for the reduced subspace.
    H = compress_interactions(reduced_subspaces, coeffs)
    return H


@type_check
def update_coeffs(
    coeffs: np.ndarray, measurements: np.ndarray, eliminated_subspaces: np.ndarray
) -> np.ndarray:
    new_coeffs = np.zeros(coeffs.shape)

    for i in range(len(eliminated_subspaces)):
        # np.einsum('ij->i',eliminated_subspaces[i] * symmetries )
        flip_sign = np.sum(eliminated_subspaces[i, :] * measurements) % 2
        if flip_sign:
            new_coeffs[i] = -coeffs[i]
        else:
            new_coeffs[i] = coeffs[i]
    return new_coeffs


@type_check
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


def simplify_H(H):
    return simplify(expand(H.doit()))


# @type_check
def get_coeffs_monomials(H) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = np.array(Poly(H).coeffs(), dtype=np.float64)
    interactions = np.array(Poly(H).monoms())
    interactions[interactions == 2] = 0  # remove all Z**2 since they are Identity.
    return coeffs, interactions


@type_check
def get_parity(arr: np.ndarray):
    """generate a parity bitstring for a given interaction"""
    temp = np.zeros((arr.shape))
    temp[-1] = arr[-1]
    for i in range(len(arr) - 2, -1, -1):  # loop from len(arr)-1 to 0.
        if arr[i] == 1:
            temp[i] += 1
            temp[i + 1] += 1
    return temp % 2


def get_H_example():
    n_electrons = 2
    orbitals = 2
    n_qubits = n_electrons * orbitals
    # define variables
    i, j, v = symbols("i j v", integer=True)
    i, j, v = Idx(i), Idx(j), Idx(v)
    X = IndexedBase("x")
    Z = IndexedBase("z")
    # define Hamiltonian
    # X
    # penalty_1 = (2 - Sum(X[i], (i,0,n_qubits-1)))**2
    # penalty_2 = ((1 - X[0] - X[1])**2 + (1 - X[2] - X[3])**2)
    # penalty_2 = [(1 - X[i] - X[j])**2 for i in pairwise(range(n_qubits))]
    # optimization = Sum( (0.1 * i* X[i]), (i,0,3))
    # Z
    penalty_1 = (2 - Sum(1 / 2 * (1 + Z[i]), (i, 0, n_qubits - 1))) ** 2
    penalty_2 = (1 - 1 / 2 * (1 + Z[0]) - 1 / 2 * (1 + Z[1])) ** 2 + (
        1 - 1 / 2 * (1 + Z[2]) - 1 / 2 * (1 + Z[3])
    ) ** 2
    optimization = Sum((0.1 * i * 1 / 2 * (1 + Z[i])), (i, 0, 3))

    H = penalty_1 + penalty_2 + optimization
    # Substitute X with 1/2 (1+Z)
    # H = H.doit().subs([(X[i], 1/2*(1 + Z[i])) for i in range(4)])
    # simplify
    H = simplify_H(H)
    return H


def get_tsp_hamiltonian(n_cities):
    # define variables
    i, j, v = symbols("i j v", integer=True)
    i, j, v = Idx(i), Idx(j), Idx(v)
    X = IndexedBase("x")
    Z = IndexedBase("z")
    # problem specific variables
    N = n_cities - 1
    # define Hamiltonian
    penalty_1 = Sum((1 - Sum(X[v, j], (j, 0, N - 1))) ** 2, (v, 0, N - 1))
    penalty_2 = Sum((1 - Sum(X[v, j], (v, 0, N - 1))) ** 2, (j, 0, N - 1))
    # optimization = (1 - Sum(X[v,j], (j,1,n_cities)) )**2
    H = penalty_1 + penalty_2  # + optimization
    # Substitute X with 1/2 (1+Z)
    H = H.doit().subs(
        [(X[v, j], 1 / 2 * (1 - Z[v, j])) for v in range(N) for j in range(N)]
    )
    # simplify
    H = simplify_H(H)
    return H


@type_check
def apply_mod2(arr: np.ndarray) -> np.ndarray:
    """Apply the addition mod2 operation on the bitstring.
        NOTE: The function acts on the right most register arr[-1] first.
        Therefore, the qubit with zero index arr[0] indicates the overall parity
        of the bitstring.

    Args:
        arr (np.ndarray): the bitstring to be operated on.

    Returns:
        new_array (np.ndarray): the new bitstring after the mod2 operation.
    """
    arr = arr[::-1]
    new_arr = np.zeros(arr.shape)
    for i in range(len(arr)):
        new_arr[i] = np.sum(arr[: i + 1]) % 2  # very strange that index must be i + 1
    return new_arr[::-1]


@type_check
def evaluate_H(interactions: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    bitstrings, _ = get_all_bitstrings(interactions.shape[1])
    costs = np.zeros(len(bitstrings))
    for i, b in enumerate(bitstrings):
        costs[i] = get_cost(interactions, coeffs, b)
    return costs


@type_check
def get_cost(interactions: np.ndarray, coeffs: np.ndarray, b: np.ndarray) -> np.float32:
    costs = np.zeros(len(interactions))
    # print('#######')
    # print(b,'bitstring')
    for i, inter in enumerate(interactions):
        # print(inter, 'current interaction')
        if np.count_nonzero(inter) == 0:  # constant
            costs[i] = coeffs[i]
        elif np.count_nonzero(inter) == 1:  # transverse field
            index = np.where(inter == 1)[0]
            costs[i] = coeffs[i] * b[index][0]
            # print(b[index][0])
        elif np.count_nonzero(inter) == 2:  # 2 qubit interactions
            index = np.where(inter == 1)[0]
            costs[i] = coeffs[i] * b[index[0]] * b[index[1]]
            # print(b[index[0]], b[index[1]])
        elif np.count_nonzero(inter) == 3:  # 3 qubit interactions
            index = np.where(inter == 1)[0]
            costs[i] = coeffs[i] * b[index[0]] * b[index[1]] * b[index[2]]
            # print(b[index[0]], b[index[1]], b[index[2]])
        elif np.count_nonzero(inter) == 4:  # 4 qubit interactions
            index = np.where(inter == 1)[0]
            costs[i] = coeffs[i] * b[index[0]] * b[index[1]] * b[index[2]] * b[index[3]]
        else:
            print("error")
            break
    return np.sum(costs).astype(np.float32)


@type_check
def get_original_bitstring(
    bitstring: np.ndarray, symmetries: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    N = int(len(bitstring) / len(symmetries))
    bitstring_reshaped = np.reshape(bitstring, (len(symmetries), N))
    measurements = [symmetries[key] for key in symmetries.keys()]
    bitstring_parity = np.insert(
        bitstring_reshaped, [0], np.reshape(measurements, (-1, 1)), axis=1
    )
    bitstring = get_parity(
        bitstring_parity.reshape(
            -1,
        )[::-1]
    )
    return bitstring[::-1], bitstring_parity.flatten()


@type_check
def get_h_tsp(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get the Hamiltonian of the TSP problem.

        1. Get the Hamiltonian of the TSP problem as a Sympy equation.
        2. Translate Sympy equation into matrix of interactions and array of coefficients.

    Args:
        n (int): number of cities

    Returns:
        interactions (np.ndarray): the matrix of interactions.
        coeffs (np.ndarray): the array of coefficients.
    """
    # n_qubits = (n-1)**2
    H = get_tsp_hamiltonian(n)
    coeffs, interactions = get_coeffs_monomials(H)
    return interactions, coeffs


@type_check
def get_h_tsp_parity(n: int) -> Tuple[np.ndarray, np.ndarray]:
    interactions, coeffs = get_h_tsp(n)
    parity_interactions = np.array([get_parity(i) for i in interactions])
    return parity_interactions, coeffs


@type_check
def get_h_tsp_parity_reduced(n: int) -> Tuple[np.ndarray, np.ndarray]:
    parity_interactions, coeffs = get_h_tsp_parity(n)
    if n == 3:
        symmetries = {"q_0": 0, "q_2": 1}
    elif n == 4:
        symmetries = {"q_0": 1, "q_3": 0, "q_6": 1}
    elif n == 5:
        symmetries = {"q_0": 0, "q_4": 1, "q_8": 0, "q_12": 1}
    elif n == 6:
        symmetries = {"q_0": 1, "q_5": 0, "q_10": 1, "q_15": 0, "q_20": 1}
    else:
        print("error")
    parity_interactions_reduced, coeffs_reduced = reduce_H_parity(
        parity_interactions, coeffs, symmetries
    )
    return parity_interactions_reduced, coeffs_reduced


@type_check
def count_interactons(interactions: np.ndarray) -> np.ndarray:
    interactions_count = np.zeros(4)
    for i, inter in enumerate(interactions):
        if int(np.count_nonzero(inter)) > 0:
            interactions_count[int(np.count_nonzero(inter)) - 1] += 1
    return interactions_count
