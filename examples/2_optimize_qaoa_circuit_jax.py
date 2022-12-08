import numpy as np
import optax
import pennylane as qml
from absl import app, flags
from jax import jit
from jaxopt import OptaxSolver
from tqdm import tqdm
from vqa import transforms
from vqa.hamiltonian import get_eigenvalues_hamiltonian
from vqa.qaoa.initial_parameter import x0
from vqa.qaoa.mixer_h import x_mixer
from vqa.templates.circuits import QAOACircuit
from vqa.templates.state_preparation import Plus
from vqa.utils.maxcut_utils import get_maxcut_graph
from vqa.utils.utils import get_approximation_ratio

flags.DEFINE_integer("p", 2, "Number of layers")
flags.DEFINE_integer("num_nodes", 9, "Number of nodes in the graph.")
flags.DEFINE_integer("steps", 1000, "Number of optimization steps")
flags.DEFINE_integer("seed", 123, "Random seed.")
flags.DEFINE_float("stepsize", 0.05, "Random seed.")
FLAGS = flags.FLAGS


def main(argv):
    del argv

    seed = FLAGS.seed
    p = FLAGS.p
    num_nodes = FLAGS.num_nodes

    graph = get_maxcut_graph(num_nodes, seed=seed)
    params = x0(p, seed)

    H, _ = qml.qaoa.maxcut(graph)

    eigenvalues = get_eigenvalues_hamiltonian(H)
    min_cost = np.min(eigenvalues)
    max_cost = np.max(eigenvalues)

    n_qubits = len(H.wires)
    mixer_h = x_mixer(n_qubits)

    circuit = QAOACircuit(
        H=H,
        initial_state=Plus(n_qubits),
        mixer_h=mixer_h,
        num_layers=p,
    )

    # loss_fun = jit(transforms.exp_val(circuit, H, interface="jax"))
    dev = qml.device("default.qubit", wires=circuit.wires)

    @jit
    @qml.qnode(dev, interface="jax")
    def loss_fun(params):
        circuit(params)
        return qml.expval(circuit.H)

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
