from copy import copy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pennylane as qml
import seaborn as sns
from absl import app, flags
from pennylane import (
    AdagradOptimizer,
    AdamOptimizer,
    GradientDescentOptimizer,
    QNGOptimizer,
)
from pennylane import numpy as np
from tqdm import tqdm
from vqa import transforms
from vqa.hamiltonian.utils_hamiltonian import get_eigenvalues_hamiltonian
from vqa.qaoa.mixer_h import x_mixer
from vqa.templates.circuits import BarrenPlateauCircuit
from vqa.templates.examples import (
    h2_vqe_circuit,
    h2o_vqe_circuit,
    h4_vqe_circuit,
    lih_vqe_circuit,
    maxcut_qaoa_circuit,
    tsp_qaoa_circuit,
)
from vqa.utils.maxcut_utils import get_maxcut_graph
from vqa_opt.optimizer import (
    QBroydenAdamOptimizer,
    QBroydenAdanOptimizer,
    QBroydenOptimizer,
    QITEOptimizer,
    QNG2Optimizer,
)

flags.DEFINE_integer("seed", 10, "Random seed.")
flags.DEFINE_integer("num_layers", 3, "number of layers.")
flags.DEFINE_integer("averages", 1, "number of averages.")
flags.DEFINE_integer("steps", 300, "Optimization steps.")
flags.DEFINE_float("stepsize", 0.1, "Stepsize.")
flags.DEFINE_string("Circuit", "BarrenPlateauCircuit", "The circuit to optimize.")
flags.DEFINE_integer("num_qubits", 6, "number of qubits for barren plateau circuit.")
FLAGS = flags.FLAGS


@dataclass
class Datapoint:
    step: int
    exp_val: float
    particle_number: float
    optimizer: str


def main(argv):
    del argv

    # save params in config file
    config = {
        "num_qubits": FLAGS.num_qubits,
        "num_layers": FLAGS.num_layers,
        "steps": FLAGS.steps,
        "seed": FLAGS.seed,
        "averages": FLAGS.averages,
        "stepsize": FLAGS.stepsize,
    }

    list_optimizer = [
        # QNGOptimizer(stepsize=FLAGS.stepsize),
        QBroydenAdamOptimizer(stepsize=FLAGS.stepsize),
        # QBroydenOptimizer(stepsize=FLAGS.stepsize),
        # QBroydenAdanOptimizer(stepsize=FLAGS.stepsize),
        # QITEOptimizer(stepsize=FLAGS.stepsize),
        QNG2Optimizer(stepsize=FLAGS.stepsize),
        # GradientDescentOptimizer(stepsize=FLAGS.stepsize),
        AdamOptimizer(stepsize=FLAGS.stepsize),
    ]

    # circuit = BarrenPlateauCircuit(FLAGS.num_layers, FLAGS.num_qubits)
    # circuit = lih_vqe_circuit(num_layers=2)
    circuit = h2_vqe_circuit(num_layers=1)
    # circuit = maxcut_qaoa_circuit(num_layers=FLAGS.num_layers, num_nodes=4)

    init_params = circuit.init(FLAGS.seed)
    H = circuit.H

    eigenvalues = get_eigenvalues_hamiltonian(H)
    min_cost, max_cost = np.min(eigenvalues), np.max(eigenvalues)

    dataset = []
    # for k in range(FLAGS.steps):
    # dataset.append(Datapoint(k, np.real(min_cost), "Ground state"))

    # iterate over optimizer
    for i, optimizer in tqdm(enumerate(list_optimizer)):
        optimizer_id = optimizer.__class__.__name__
        for j in tqdm(range(FLAGS.averages)):

            dev = qml.device("default.qubit", wires=circuit.wires)

            @qml.qnode(dev)
            def get_particle_number(x):
                circuit(x)
                H = qml.qchem.particle_number(circuit.active_orbitals)
                # H = qml.qchem.spin2(circuit.active_electrons, circuit.active_orbitals)
                # H = qml.qchem.spinz(circuit.active_orbitals)
                return qml.expval(circuit.H)

            fun = transforms.exp_val(circuit, circuit.H)

            init_params = circuit.init(FLAGS.seed + j)

            # initial datapoint
            exp_val = fun(init_params)
            particle_number = get_particle_number(init_params)

            dataset.append(
                Datapoint(
                    step=0,
                    exp_val=np.real(exp_val),
                    particle_number=particle_number,
                    optimizer=optimizer_id,
                )
            )

            params = copy(init_params)
            for k in tqdm(range(1, FLAGS.steps + 1)):  # skip index zero.
                params = optimizer.step(fun, params)

                exp_val = fun(params)
                particle_number = get_particle_number(params)

                dataset.append(
                    Datapoint(
                        step=k,
                        exp_val=np.real(exp_val),
                        particle_number=particle_number,
                        optimizer=optimizer_id,
                    )
                )

    df = pd.DataFrame(data=dataset)
    df["exp_val"] = df["exp_val"].astype(float)
    df["particle_number"] = df["particle_number"].astype(float)

    df = df[df["step"] == FLAGS.steps]
    print(df.groupby("optimizer")["particle_number"].min())

    # df["entropy"] = df["entropy"].astype(float)  # why is this necessary?
    # create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("white")

    g = sns.lineplot(x="step", y="particle_number", hue="optimizer", data=df, ax=ax)
    ax.legend().set_title("")
    ax.set_xscale("symlog")
    # ax.set_yscale("log")

    title = f"entropy_plot_average_{FLAGS.num_qubits}_qubits_circuit_{circuit}_num_layer_{FLAGS.num_layers}_stepsize_{FLAGS.stepsize}_n_steps_{FLAGS.steps}_seed_{FLAGS.seed}.pdf"
    print(title)
    fig.savefig(title)


if __name__ == "__main__":
    app.run(main)


# import numpy as np
# import pennylane as qml
# from vqa import transforms
# from vqa.templates.examples import h2_vqe_circuit

# # dev = qml.device("default.qubit", wires=2)

# # @qml.qnode(dev)
# # def circuit_entropy(x):
# #     qml.IsingXX(x, wires=[0, 1])
# #     return qml.vn_entropy(wires=[0, 1])


# # print(circuit_entropy(np.pi / 2), "entropy")

# # @qml.qnode(dev)
# # def circuit_entropy(x):
# #     def _circ(x):
# #         qml.IsingXX(x, wires=[0, 1])
# #         # return qml.vn_entropy(wires=[0])
# #         return _circ(x)

# # fun = transforms.exp_val(circuit_entropy, H)
# # print(fun(np.pi / 2))

# # fun = transforms.von_neumann_entropy(circuit_entropy, wires=2)

# # print(fun(np.pi / 2))


# # dev = qml.device("default.qubit", wires=circuit.wires)

# # @qml.qnode(dev)
# # def h2_entropy(x):
# # circuit(x)
# # return qml.vn_entropy(wires=[0])

# circuit = h2_vqe_circuit()
# params = circuit.init()

# print(circuit.num_qubits, "num qubits")
# print(np.log2(circuit.num_qubits), "max entropy")
# fun = transforms.vn_entropy(circuit, wires=circuit.wires)

# print(fun(params), "my entropy 2")
