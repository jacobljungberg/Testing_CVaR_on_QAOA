# TODO it is a bit odd that only BarrenPlateau is a class but the others are functions.
# maybe make everything a function?
from vqa.templates.circuits import BarrenPlateauCircuit

from .barren_plateau_circuit import barren_plateau_circuit
from .h2_circuit import h2_vqe_circuit
from .h2_simple_circuit import h2_simple_vqe_circuit
from .h2o_circuit import h2o_vqe_circuit
from .h4_circuit import h4_vqe_circuit
from .lih_circuit import lih_vqe_circuit
from .maxcut_circuit import maxcut_qaoa_circuit
from .tsp_circuit import tsp_qaoa_circuit
