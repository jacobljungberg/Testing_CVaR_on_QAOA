from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from vqa.utils.utils import get_all_spinstrings


class Instance(ABC):
    """
    Abstract class for instance.
    """

    @abstractmethod
    # def __init__(self, ins, weights, p, gamma_list, beta_list):
    def __init__(self):
        """
        Initialize instance.
        """
        self.problem_id = None
        # self.label = self.problem_id.capitalize()
        self.n_qubits = None
        self.costs = None
        self.ins = None

    def __repr__(self):
        return f" Problem id: {self.problem_id} \n n qubits: {self.n_qubits} "

    @abstractmethod
    def __call__(self):
        pass


class Maxcut(Instance):
    """
    Generate maxcut instance.
    """

    def __init__(self, **kwargs):
        """
        Initialize maxcut instance.
        """
        super().__init__()
        self.n_nodes = kwargs["n_nodes"] if "n_nodes" in kwargs else 4
        self.weights = kwargs["weights"] if "weights" in kwargs else (1,)
        self.problem_id = "maxcut"
        self.label = self.problem_id.capitalize()
        assert (
            len(self.weights) == 1
        ), f"number of weights must be 1 but given {len(self.weights)}"
        self.n_qubits = self.n_nodes

    def __call__(self):
        self.spinstrings = get_all_spinstrings(self.n_qubits)
        self.ins = get_maxcut_instance(self.n_nodes)
        W = nx.to_numpy_matrix(self.ins)
        self.costs = get_maxcut_cost(W)


class Tsp(Instance):
    """
    Generate maxcut instance.
    """

    def __init__(self, **kwargs):
        """
        Initialize maxcut instance.
        """
        super().__init__()
        self.weights = kwargs["weights"] if "weights" in kwargs else (100, 1)
        self.problem_id = "tsp"
        self.label = self.problem_id.capitalize()

        assert (
            len(self.weights) == 2
        ), f"number of weights must be 2 but given {len(self.weights)}"
        self.n_cities = kwargs["n_cities"] if "n_cities" in kwargs else 3
        self.n_qubits = (self.n_cities - 1) ** 2
        # self.n_qubits = (self.n_cities) ** 2

    def __call__(self):
        # self.spinstrings,_ = 1 - 2 * get_all_spinstrings(self.n_qubits)
        self.spinstrings = get_all_spinstrings(self.n_qubits)
        self.dist = get_tsp_instance(self.n_cities)
        self.dist = self.dist / self.dist.max()
        self.costs = get_tsp_cost(self.dist, self.weights)


###############################################################################
# Helper functions Maxcut
###############################################################################


def get_maxcut_instance(n: int = 4):
    np.random.seed(0)
    # G = nx.dense_gnm_random_graph(n=n, m=5)
    # for (u, v) in G.edges():
    #   G.edges[u,v]['weight'] = 1#random.randint(0,10)
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    G = nx.Graph(edges)
    return G


def get_maxcut_cost(ins: np.ndarray) -> np.ndarray:
    """Brute force the cost of a maxcut problem instance."""
    n_edges = np.count_nonzero(np.triu(ins))
    solutions = (get_all_spinstrings(ins.shape[0]) * 2) - 1
    cost = (
        1 / 2 * (np.diag(solutions @ np.triu(ins) @ solutions.T) - n_edges)
    )  # TODO verify
    return cost


###############################################################################
# Helper functions tsp
###############################################################################


def get_qubit_index(i: int, j: int, N: int):
    return i * N + j


def get_tsp_instance(n_cities: int = 3, seed: int = 0):
    np.random.seed(seed)
    coord = np.random.uniform(0, 100, (n_cities, 2))
    ins = calc_distance(coord)
    return ins


def calc_distance(coord):
    """calculate distance"""
    assert coord.shape[1] == 2
    dim = coord.shape[0]
    w = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):
            delta = coord[i] - coord[j]
            w[i, j] = np.rint(np.hypot(delta[0], delta[1]))
    w += w.T
    return w


def get_tsp_cost(ins, weights):
    N = ins.shape[0] - 1
    n_qubits = N**2
    spinstrings = get_all_spinstrings(n_qubits)
    costs = np.zeros((len(spinstrings),))
    for i, bitstring in enumerate(spinstrings):
        costs[i] = get_single_cost(bitstring, ins, N, weights)
    return costs


def get_single_cost(bitstring, ins, N, weights):
    solution = np.reshape(bitstring, (N, N))
    A, B = weights
    penalty_1 = A * get_cost_penalty_hamiltonian_1(N, bitstring)
    penalty_2 = A * get_cost_penalty_hamiltonian_2(N, bitstring)
    cost = B * get_cost_optimization_hamiltonian(ins, bitstring)
    return cost + penalty_1 + penalty_2


def get_cost_penalty_hamiltonian_1(N, bitstring):
    cost = 0
    for v in range(N):
        for j in range(N):
            cost += (2 - N) * bitstring[get_qubit_index(v, j, N)]
        for j in range(N):
            for j_ in range(N):
                if j < j_:
                    cost += (
                        bitstring[get_qubit_index(v, j, N)]
                        * bitstring[get_qubit_index(v, j_, N)]
                    )
    return cost


def get_cost_penalty_hamiltonian_2(N, bitstring):
    cost = 0
    for j in range(N):
        for v in range(N):
            cost += (2 - N) * bitstring[get_qubit_index(v, j, N)]
        for v in range(N):
            for v_ in range(N):
                if v < v_:
                    cost += (
                        bitstring[get_qubit_index(v, j, N)]
                        * bitstring[get_qubit_index(v_, j, N)]
                    )
    return cost


def get_cost_optimization_hamiltonian(ins, bitstring):
    N = ins.shape[0] - 1
    cost = 0
    for u in range(N):
        for v in range(N):
            if u < v:
                for j in range(N - 1):  # TODO is this -1 here correct?
                    cost += ins[u, v] * bitstring[get_qubit_index(u, j, N)]
                    cost += ins[u, v] * bitstring[get_qubit_index(v, j + 1, N)]
                    cost += (
                        ins[u, v]
                        * bitstring[get_qubit_index(u, j, N)]
                        * bitstring[get_qubit_index(v, j + 1, N)]
                    )
    return cost
