from typing import List, Tuple

import pennylane as qml
import pennylane.numpy as pnp
from openfermion import MolecularData
from openfermionpyscf import prepare_pyscf_molecule
from pennylane import Hamiltonian, qchem
from pennylane.qchem.convert import import_operator
from pyscf import cc, fci, scf


# Change to this function to also compute the exact energy here directly, also includes the option of freezing orbitals (like we will do in the H2O case)
def molecular_hamiltonian(
    symbols: List = ["H", "H"],
    coordinates: pnp.ndarray = pnp.array([[0, 0, 0], [0, 0, 1.0]]),
    name: str = "molecule",
    charge: int = 0,
    mult: int = 1,
    basis: str = "sto-6g",
    method: str = "pyscf",
    active_electrons: int = None,
    active_orbitals: int = None,
    mapping: str = "jordan_wigner",
    wires: List = None,
    frozen: int = 0,
) -> Tuple[qml.Hamiltonian, int, float]:

    if len(coordinates) == len(symbols) * 3:
        geometry_hf = coordinates
    elif len(coordinates) == len(symbols):
        geometry_hf = coordinates.flatten()

    hf_file = qchem.meanfield(
        symbols, geometry_hf, name, charge, mult, basis, method, "."
    )

    molecule = MolecularData(filename=hf_file)

    # addition to get the FCI energy directly
    pyscf_molecule = prepare_pyscf_molecule(molecule)

    if pyscf_molecule.spin:
        pyscf_scf = scf.ROHF(pyscf_molecule).run()
    else:
        pyscf_scf = scf.RHF(pyscf_molecule).run()
    # print("HF energy: ", pyscf_scf.e_tot)

    if frozen > 0:
        mycc = cc.CCSD(pyscf_scf, frozen=frozen).run()
        et = mycc.ccsd_t()

        fci_energy = mycc.e_tot + et
    else:
        pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
        fci_energy = pyscf_fci.kernel()[0]

    core, active = qchem.active_space(
        molecule.n_electrons,
        molecule.n_orbitals,
        mult,
        active_electrons,
        active_orbitals,
    )

    # openfermion version of hamiltonian
    h_of, qubits = (
        qchem.decompose(hf_file, mapping, core, active),
        2 * len(active),
    )

    # pennylane version of Hamiltonian (maybe you need that!)
    h_pl = import_operator(h_of, wires=wires)

    H = Hamiltonian(h_pl.coeffs, h_pl.ops)

    # return H, qubits, fci_energy
    return H, qubits, pyscf_scf.e_tot
