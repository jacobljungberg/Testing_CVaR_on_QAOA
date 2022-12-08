import numpy as np
import pytest
from vqa.tests.utils_test import tsp_9_qubits_coeffs, tsp_9_qubits_interactions
from vqa.utils.symmetries import apply_mod2, get_cost, get_parity


@pytest.mark.parametrize(
    "interaction, solution",
    [
        (np.array([0, 1, 1, 1]), [0, 1, 0, 0]),
        (np.array([0, 1, 0, 1]), [0, 1, 1, 1]),
        (np.array([1, 0, 0, 0]), [1, 1, 0, 0]),
        (np.array([0, 1, 0, 0]), [0, 1, 1, 0]),
        (np.array([0, 0, 1, 0]), [0, 0, 1, 1]),
        (np.array([0, 0, 0, 1]), [0, 0, 0, 1]),
        (np.array([1, 0, 0, 1]), [1, 1, 0, 1]),
        (np.array([1, 0, 0, 0]), [1, 1, 0, 0]),
        (np.array([1, 1, 1, 0]), [1, 0, 0, 1]),
        (np.array([1, 0, 1, 1]), [1, 1, 1, 0]),
    ],
)
def test_parity_operation(interaction, solution):
    parity_interaction = get_parity(interaction)
    assert all(
        [a == b for a, b in zip(parity_interaction, solution)]
    ), f"parity operation is not correct, for the array, {interaction} we expect {solution} but got {parity_interaction} instead"


@pytest.mark.parametrize(
    "bitstring, solution",
    [
        (np.array([1, 1, 1, 1]), [0, 1, 0, 1]),
        (np.array([1, 0, 1, 0]), [0, 1, 1, 0]),
        (np.array([1, 0, 1, 1]), [1, 0, 0, 1]),
        (np.array([1, 1, 0, 1]), [1, 0, 1, 1]),
        (np.array([1, 0, 0, 1]), [0, 1, 1, 1]),
        (np.array([1, 0, 0, 0]), [1, 0, 0, 0]),
        (np.array([1, 0, 1, 0]), [0, 1, 1, 0]),
    ],
)
def test_mod_2(bitstring, solution):
    mod_bitstring = apply_mod2(bitstring)
    assert all(
        [a == b for a, b in zip(mod_bitstring, solution)]
    ), f"mod 2 operation is not correct, got {mod_bitstring} instead of {solution}"


@pytest.mark.parametrize(
    "interaction, coeffs, bitstring, expected_cost",
    [
        (
            np.array(
                [[0, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]
            ),
            np.array([2, 0.5, 0.5, 0.5, 0.5]),
            np.array([1, 0, 0, 1]),
            0,
        ),
        (
            np.array(
                [[0, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]
            ),
            np.array([2, 0.5, 0.5, 0.5, 0.5]),
            np.array([0, 1, 1, 0]),
            0,
        ),
        (
            np.array([[0, 1, 1, 1], [1, 0, 0, 1]]),
            np.array([1, 1]),
            np.array([1, 1, 0, 1]),
            2,
        ),
        (
            np.array(
                [[0, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]
            ),
            np.array([2, 0.5, 0.5, 0.5, 0.5]),
            np.array([1, 1, 0, 0]),
            2,
        ),
        (
            np.array(
                [[0, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]
            ),
            np.array([2, 0.5, 0.5, 0.5, 0.5]),
            np.array([1, 1, 1, 1]),
            4,
        ),
        (
            tsp_9_qubits_interactions,
            tsp_9_qubits_coeffs,
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
            0,
        ),
    ],
)
def test_get_cost(interaction, coeffs, bitstring, expected_cost):
    assert len(interaction) == len(
        coeffs
    ), f"interaction and coeffs must have the same length"
    b_parity = apply_mod2(bitstring)
    bitstring[bitstring == 1] = -1
    bitstring[bitstring == 0] = 1
    b_parity[b_parity == 1] = -1
    b_parity[b_parity == 0] = 1

    parity_interactions = np.array([get_parity(i) for i in interaction])
    cost = get_cost(np.array(interaction), coeffs, bitstring)
    costs_parity = get_cost(parity_interactions, coeffs, b_parity)

    assert (
        cost == costs_parity
    ), f"costs for orig H and parity H are not the same but must be got {cost} and {expected_cost}"
    assert (
        cost == expected_cost
    ), f"cost is not correct, got {cost} instead of {expected_cost}"
