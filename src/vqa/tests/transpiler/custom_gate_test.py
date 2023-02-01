# three qubit gate
# https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040348
# https://arxiv.org/pdf/2104.14722.pdf
# Define the quantum register and quantum circuit


# import numpy as np
# from qiskit import QuantumCircuit, QuantumRegister
# from qiskit.quantum_info import Operator

# theta = np.pi / 2
# phi = np.pi / 2
# gamma = np.pi / 2

# # Define the matrix for the custom gate
# matrix = [
#     [1, 0, 0, 0],
#     [
#         0,
#         -np.exp(1j * gamma) * np.sin(theta / 2) ** 2 + np.cos(theta / 2) ** 2,
#         0.5 * (1 + np.exp(1j * gamma)) * np.exp(-1j * phi) * np.sin(theta),
#         0,
#     ],
#     [
#         0,
#         0.5 * (1 + np.exp(1j * gamma)) * np.exp(1j * phi) * np.sin(theta),
#         -np.exp(1j * gamma) * np.cos(theta / 2) ** 2 + np.sin(theta / 2) ** 2,
#         0,
#     ],
#     [0, 0, 0, -np.exp(1j * gamma)],
# ]

# # Create the matrix operator for the custom gate
# czs_op = Operator(matrix)

# # Create the quantum circuit that implements the custom gate
# q = QuantumRegister(4)
# circuit = QuantumCircuit(q, name="test circuit")
# circuit.unitary(czs_op, [q[0], q[1]], label="CZS")
# print(circuit)

# # cczs = czs_circuit.control(1)
# cczs = circuit.control(1)
# print(cczs)

# circuit.unitary(cczs, [q[0], q[1], q[2]], label="CCZS")


# print(circuit)
