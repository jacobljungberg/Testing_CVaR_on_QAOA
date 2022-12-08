from vqa import transforms
from vqa.templates.circuits import BarrenPlateauCircuit
from vqa.templates.examples import h2_vqe_circuit

circuit = BarrenPlateauCircuit(layer=5, n_qubits=7)
circuit = h2_vqe_circuit(num_layers=1)
params = circuit.init()
loss_function = transforms.exp_val(circuit, circuit.H)
transforms.draw(circuit)
# print(qml.draw(loss_function, expansion_strategy="device")(params))
