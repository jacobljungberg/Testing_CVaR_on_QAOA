import pennylane as qml
import networkx as nx
from beartype import beartype


@beartype
def maxcut_hamiltonian(graph: nx.Graph):
    # if not isinstance(graph, nx.Graph):
    # raise TypeError(
    # "Graph is not a networkx class (found type: %s)" % type(graph).__name__
    # )

    coeffs = [-0.5 for e in graph.edges]
    obs = [qml.Identity(e[0]) @ qml.Identity(e[1]) for e in graph.edges]
    identity_h = qml.Hamiltonian(coeffs, obs)

    coeffs = [0.5 for (u, v) in graph.edges()]
    obs = [qml.PauliZ(u) @ qml.PauliZ(v) for (u, v) in graph.edges()]
    zz_h = qml.Hamiltonian(coeffs, obs)

    return zz_h + identity_h
