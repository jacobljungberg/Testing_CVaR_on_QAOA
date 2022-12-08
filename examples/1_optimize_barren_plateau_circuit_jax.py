import numpy as np
import optax
import pennylane as qml
from absl import app, flags
from jax import jit
from jaxopt import OptaxSolver
from tqdm import tqdm
from vqa import transforms
from vqa.hamiltonian import get_eigenvalues_hamiltonian
from vqa.templates.circuits import BarrenPlateauCircuit
from vqa.utils.utils import get_approximation_ratio

flags.DEFINE_integer("layer", 7, "Number of layers")
flags.DEFINE_integer("n_qubits", 7, "Number of qubits")
flags.DEFINE_integer("steps", 1000, "Number of optimization steps")
flags.DEFINE_integer("seed", 123, "Random seed.")
flags.DEFINE_float("stepsize", 0.1, "Random seed.")
FLAGS = flags.FLAGS


def main(argv):
    del argv

    circuit = BarrenPlateauCircuit(FLAGS.layer, FLAGS.n_qubits)
    params = circuit.init()

    eigenvalues = get_eigenvalues_hamiltonian(circuit.H)
    min_cost, max_cost = np.min(eigenvalues), np.max(eigenvalues)
    print(f"min cost: {min_cost}, max cost: {max_cost}")

    # loss_fun = jit(transforms.exp_val(circuit, interface="jax"))
    dev = qml.device("default.qubit", wires=circuit.wires)

    @jit
    @qml.qnode(dev, interface="jax")
    def loss_fun(params):
        circuit(params)
        return qml.expval(circuit.H)

    opt = optax.adam(FLAGS.stepsize)
    solver = OptaxSolver(opt=opt, fun=loss_fun, maxiter=100)
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
