import numpy as np
import optax
import pennylane as qml
from absl import app, flags
from jax import jit
from jaxopt import OptaxSolver
from tqdm import tqdm
from vqa import transforms
from vqa.hamiltonian import get_eigenvalues_hamiltonian
from vqa.templates.circuits import Sequential
from vqa.templates.layers import Pauli_X
from vqa.templates.state_preparation import Plus
from vqa.utils.maxcut_utils import get_maxcut_graph
from vqa.utils.utils import get_approximation_ratio

flags.DEFINE_integer("p", 2, "Number of layers")
flags.DEFINE_integer("steps", 1000, "Number of optimization steps")
flags.DEFINE_integer("seed", 123, "Random seed.")
flags.DEFINE_float("stepsize", 0.05, "Random seed.")
FLAGS = flags.FLAGS


def main(argv):
    del argv

    seed = FLAGS.seed
    p = FLAGS.p
    num_nodes = 5

    graph = get_maxcut_graph(num_nodes, seed=seed)

    H, _ = qml.qaoa.maxcut(graph)

    eigenvalues = get_eigenvalues_hamiltonian(H)
    min_cost, max_cost = np.min(eigenvalues), np.max(eigenvalues)

    n_qubits = len(H.wires)
    wires = H.wires

    circ = Sequential(
        [
            # TODO Plus should use wires instead of n_qubits
            # state preparation is always the first layer and unparametrized!
            Plus(n_qubits),
            H,
            Pauli_X(wires),  # should this be a Hamiltonian or a circuit? Can be both
            H,
            Pauli_X(wires),
        ]
    )
    params = circ.params

    loss_fun = jit(transforms.exp_val(circ, H, interface="jax"))

    opt = optax.noisy_sgd(FLAGS.stepsize)
    solver = OptaxSolver(opt=opt, fun=loss_fun, maxiter=1)
    state = solver.init_state(params)
    for _ in tqdm(range(100)):
        params, state = solver.update(
            params=params,
            state=state,
        )

    min_cost_optimized = loss_fun(params)
    r = get_approximation_ratio(min_cost_optimized, min_cost, max_cost)

    print(f"ground state: {np.round(min_cost, 1)}")
    print(f"exp val: {np.round(min_cost_optimized, 2)}")
    print(f"approx. ratio: {np.round(r,2).real}")


if __name__ == "__main__":
    app.run(main)
