import numpy as np
from beartype import beartype
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from vqa.utils.utils import get_all_bitstrings


@beartype
def classify_bitstrings_tsp(costs: List):
    """
    Classify the bitstrings for the TSP problem into four categories:
    optimal (0), valid (1), const hamming weight (2) and violating (3) bitstrings.
    Args:
        costs (np.ndarray): The cost values of the bitstrings.

    Returns:
        (np.ndarray, np.ndarray): The classification of the bitstrings and
        the bitstrings.
    """

    # TODO ugly function, does too many things at once. Refactor.
    index_min = np.where(costs == np.min(costs))[0]
    n_qubits = int(np.log2(len(costs)))

    all_bitstrings = get_all_bitstrings(n_qubits)  # .tolist()
    dim = int(np.sqrt(n_qubits))
    class_bitstring = []
    const_hamming_weight_bitstrings = np.zeros(len(all_bitstrings))

    for i, b in enumerate(all_bitstrings):
        sol = np.reshape(b, (dim, dim))
        valid_col = not len(np.where(np.sum(sol, axis=0) - 1 != 0)[0])
        valid_row = not len(np.where(np.sum(sol, axis=1) - 1 != 0)[0])
        # const hamming weight
        if np.sum(b) == dim:
            const_hamming_weight_bitstrings[i] = 1
        # classification bitstring
        if i in index_min:
            class_bitstring.append(0)
        elif valid_row and valid_col:
            class_bitstring.append(1)
        else:
            class_bitstring.append(2)

    return np.array(class_bitstring), all_bitstrings
