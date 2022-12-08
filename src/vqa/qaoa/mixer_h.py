import networkx as nx
import numpy as np
from beartype import beartype
from pennylane import qaoa
from vqa.utils.utils import pairwise


@beartype
def circular_xy_mixer(n_qubits: int):
    list_1 = [(u, v) for u, v in pairwise(range(n_qubits))]
    list_2 = [(u, v) for u, v in pairwise(np.roll(range(n_qubits), -1))]
    graph = nx.Graph(list_1 + list_2)
    mixer_h = qaoa.xy_mixer(graph)
    return mixer_h


@beartype
def row_mixer(n_qubits: int):
    N = int(np.sqrt(n_qubits))
    list_ = [(i + j * N, (i + 1) % N + j * N) for i in range(N) for j in range(N)]
    graph = nx.Graph(list_)
    mixer_h = qaoa.xy_mixer(graph)
    return mixer_h


@beartype
def row_mixer_2(n_qubits: int):
    N = int(np.sqrt(n_qubits))
    list_ = [(i + j * N, (i + 1) % N + j * N) for i in range(N - 1) for j in range(N)]
    graph = nx.Graph(list_)
    mixer_h = qaoa.xy_mixer(graph)
    return mixer_h


@beartype
def row_flex_mixer(n_qubits: int):
    N = int(np.sqrt(n_qubits))
    list_ = [[(i + j * N, (i + 1) % N + j * N) for i in range(N - 1)] for j in range(N)]
    mixer_h_list = []
    for sub_list in list_:
        graph = nx.Graph(sub_list)
        mixer_h_list.append(qaoa.xy_mixer(graph))
    return mixer_h_list


@beartype
def x_mixer(n_qubits: int):
    mixer_h = qaoa.x_mixer(range(n_qubits))
    return mixer_h
