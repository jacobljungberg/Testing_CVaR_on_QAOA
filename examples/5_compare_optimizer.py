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
        QBroydenAdamOptimizer(stepsize=FLAGS.stepsize),
        QBroydenOptimizer(stepsize=FLAGS.stepsize),
        # QBroydenAdanOptimizer(stepsize=FLAGS.stepsize),
        QITEOptimizer(stepsize=FLAGS.stepsize),
        # QNGOptimizer(stepsize=FLAGS.stepsize),
        QNG2Optimizer(stepsize=FLAGS.stepsize),
        GradientDescentOptimizer(stepsize=FLAGS.stepsize),
        AdamOptimizer(stepsize=FLAGS.stepsize),
    ]

    # circuit = BarrenPlateauCircuit(FLAGS.num_layers, FLAGS.num_qubits)
    # circuit = lih_vqe_circuit(num_layers=2)
    circuit = h2_vqe_circuit(num_layers=2)
    # circuit = maxcut_qaoa_circuit(num_layers=FLAGS.num_layers, num_nodes=4)

    init_params = circuit.init(FLAGS.seed)
    H = circuit.H

    eigenvalues = get_eigenvalues_hamiltonian(H)
    min_cost, max_cost = np.min(eigenvalues), np.max(eigenvalues)

    dataset = []
    for k in range(FLAGS.steps):
        dataset.append(Datapoint(k, np.real(min_cost), "Ground state"))

    # iterate over optimizer
    for i, optimizer in tqdm(enumerate(list_optimizer)):
        optimizer_id = optimizer.__class__.__name__
        for j in tqdm(range(FLAGS.averages)):
            fun = transforms.exp_val(
                # circuit, H, wires=range(len(circuit.wires) + 1)
                circuit,
                H,
                wires=range(len(circuit.wires)),
            )
            init_params = circuit.init(FLAGS.seed + j)
            # initial datapoint
            exp_val = fun(init_params)
            dataset.append(Datapoint(0, np.real(exp_val), optimizer_id))

            params = copy(init_params)
            for k in tqdm(range(1, FLAGS.steps + 1)):  # skip index zero.
                params = optimizer.step(fun, params)
                exp_val = fun(params)
                dataset.append(Datapoint(k, np.real(exp_val), optimizer_id))

    df = pd.DataFrame(data=dataset)
    df["exp_val"] = df["exp_val"].astype(float)  # why is this necessary?
    # create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("white")

    g = sns.lineplot(x="step", y="exp_val", hue="optimizer", data=df, ax=ax)
    ax.legend().set_title("")
    ax.set_xscale("symlog")

    title = f"loss_plot_average_{FLAGS.num_qubits}_qubits_circuit_{circuit}_num_layer_{FLAGS.num_layers}_stepsize_{FLAGS.stepsize}_n_steps_{FLAGS.steps}_seed_{FLAGS.seed}.pdf"
    print(title)
    fig.savefig(title)


if __name__ == "__main__":
    app.run(main)
