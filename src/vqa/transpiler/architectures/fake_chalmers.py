# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from qiskit.providers.fake_provider import FakeBackend
from qiskit.providers.models import GateConfig, QasmBackendConfiguration
from qiskit.transpiler import CouplingMap

# LIST OF EDGES MUST BE GROUPED BUT NOT ORDERED.
# FOR EACH EDGE THE REVERSE EDGE MUST FOLLOW.

"""
Device (4 qubit).
"""
chalmers_native_gate_set = ["rz", "rx", "h", "x", "z", "iswap"]


class FakeChalmers4(FakeBackend):
    """A fake 4 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01
         ↕   ↕
        02 ↔ 03
        """
        self.n_row = 2
        self.n_column = 2
        self.n_qubits = 4
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_device",
            backend_version="0.0.0",
            n_qubits=4,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Device (6 qubit).
"""


class FakeChalmers6(FakeBackend):
    """A fake 6 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02
         ↕   ↕    ↕
        03 ↔ 04 ↔ 05
        """
        self.n_row = 2
        self.n_column = 3
        self.n_qubits = 6
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_device",
            backend_version="0.0.0",
            n_qubits=6,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Device (9 qubit).
"""


class FakeChalmers9(FakeBackend):
    """A fake 9 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02
         ↕   ↕    ↕
        03 ↔ 04 ↔ 05
         ↕   ↕    ↕
        06 ↔ 07 ↔ 08
        """
        self.n_row = 3
        self.n_column = 3
        self.n_qubits = 9
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_device",
            backend_version="0.0.0",
            n_qubits=9,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (12 qubit).
"""


class FakeChalmers12(FakeBackend):
    """A fake 12 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03
         ↕   ↕    ↕    ↕
        04 ↔ 05 ↔ 06 ↔ 07
         ↕   ↕    ↕    ↕
        08 ↔ 09 ↔ 10 ↔ 11
        """
        self.n_row = 3
        self.n_column = 4
        self.n_qubits = 12
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=12,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (16 qubit).
"""


class FakeChalmers16(FakeBackend):
    """A fake 16 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03
         ↕   ↕    ↕    ↕
        04 ↔ 05 ↔ 06 ↔ 07
         ↕   ↕    ↕    ↕
        08 ↔ 09 ↔ 10 ↔ 11
         ↕   ↕    ↕    ↕
        12 ↔ 13 ↔ 14 ↔ 15
        """
        self.n_row = 4
        self.n_column = 4
        self.n_qubits = 16
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=16,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (20 qubit).
"""


class FakeChalmers20(FakeBackend):
    """A fake 20 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
         ↕   ↕    ↕    ↕    ↕
        05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕   ↕    ↕    ↕    ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
         ↕   ↕    ↕    ↕    ↕
        15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
        """
        self.n_row = 4
        self.n_column = 5
        self.n_qubits = 20
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=20,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (25 qubit).
"""


class FakeChalmers25(FakeBackend):
    """A fake 25 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
         ↕   ↕    ↕    ↕    ↕
        05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕   ↕    ↕    ↕    ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
         ↕   ↕    ↕    ↕    ↕
        15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
         ↕   ↕    ↕    ↕    ↕
        20 ↔ 21 ↔ 22 ↔ 23 ↔ 24
        """
        self.n_row = 5
        self.n_column = 5
        self.n_qubits = 25
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=25,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (36 qubit).
"""


class FakeChalmers36(FakeBackend):
    """A fake 36 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
         ↕   ↕    ↕    ↕    ↕
        05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕   ↕    ↕    ↕    ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
         ↕   ↕    ↕    ↕    ↕
        15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
         ↕   ↕    ↕    ↕    ↕
        20 ↔ 21 ↔ 22 ↔ 23 ↔ 24
        """
        self.n_row = 6
        self.n_column = 6
        self.n_qubits = 36
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=36,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (49 qubit).
"""


class FakeChalmers49(FakeBackend):
    """A fake 49 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
         ↕   ↕    ↕    ↕    ↕
        05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕   ↕    ↕    ↕    ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
         ↕   ↕    ↕    ↕    ↕
        15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
         ↕   ↕    ↕    ↕    ↕
        20 ↔ 21 ↔ 22 ↔ 23 ↔ 24
        """
        self.n_row = 7
        self.n_column = 7
        self.n_qubits = 49
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=49,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (50 qubit).
"""


class FakeChalmers50(FakeBackend):
    """A fake 50 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04 ↔ 05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14 ↔ 15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        20 ↔ 21 ↔ 22 ↔ 23 ↔ 24 ↔ 25 ↔ 26 ↔ 27 ↔ 28 ↔ 29
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        30 ↔ 31 ↔ 32 ↔ 33 ↔ 34 ↔ 35 ↔ 36 ↔ 37 ↔ 38 ↔ 39
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        40 ↔ 41 ↔ 42 ↔ 43 ↔ 44 ↔ 45 ↔ 46 ↔ 47 ↔ 48 ↔ 49
        """
        self.n_row = 5
        self.n_column = 10
        self.n_qubits = 50
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=50,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (64 qubit).
"""


class FakeChalmers64(FakeBackend):
    """A fake 64 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
         ↕   ↕    ↕    ↕    ↕
        05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕   ↕    ↕    ↕    ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
         ↕   ↕    ↕    ↕    ↕
        15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
         ↕   ↕    ↕    ↕    ↕
        20 ↔ 21 ↔ 22 ↔ 23 ↔ 24
        """
        self.n_row = 8
        self.n_column = 8
        self.n_qubits = 64
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=64,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (81 qubit).
"""


class FakeChalmers81(FakeBackend):
    """A fake 81 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
         ↕   ↕    ↕    ↕    ↕
        05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕   ↕    ↕    ↕    ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
         ↕   ↕    ↕    ↕    ↕
        15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
         ↕   ↕    ↕    ↕    ↕
        20 ↔ 21 ↔ 22 ↔ 23 ↔ 24
        """
        self.n_row = 9
        self.n_column = 9
        self.n_qubits = 81
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=81,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None


"""
Chalmers device (100 qubit).
"""


class FakeChalmers100(FakeBackend):
    """A fake 100 qubit backend."""

    def __init__(self):
        """
        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04 ↔ 05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14 ↔ 15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        20 ↔ 21 ↔ 22 ↔ 23 ↔ 24 ↔ 25 ↔ 26 ↔ 27 ↔ 28 ↔ 29
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        30 ↔ 31 ↔ 32 ↔ 33 ↔ 34 ↔ 35 ↔ 36 ↔ 37 ↔ 38 ↔ 39
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        40 ↔ 41 ↔ 42 ↔ 43 ↔ 44 ↔ 45 ↔ 46 ↔ 47 ↔ 48 ↔ 49
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        50 ↔ 51 ↔ 52 ↔ 53 ↔ 54 ↔ 55 ↔ 56 ↔ 57 ↔ 58 ↔ 59
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        60 ↔ 61 ↔ 62 ↔ 63 ↔ 64 ↔ 65 ↔ 66 ↔ 67 ↔ 68 ↔ 69
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        70 ↔ 71 ↔ 72 ↔ 73 ↔ 74 ↔ 75 ↔ 76 ↔ 77 ↔ 78 ↔ 79
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        80 ↔ 81 ↔ 82 ↔ 83 ↔ 84 ↔ 85 ↔ 86 ↔ 87 ↔ 88 ↔ 89
         ↕   ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕    ↕
        90 ↔ 91 ↔ 92 ↔ 93 ↔ 94 ↔ 95 ↔ 96 ↔ 97 ↔ 98 ↔ 99
        """
        self.n_row = 10
        self.n_column = 10
        self.n_qubits = 100
        coupling = CouplingMap()
        cmap = coupling.from_grid(self.n_row, self.n_column).get_edges()

        configuration = QasmBackendConfiguration(
            backend_name="fake_chalmers",
            backend_version="0.0.0",
            n_qubits=100,
            basis_gates=None,  # ['rx', 'rz','iswap','cz','id'], # check native gate set
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            max_shots=100000,
            memory=False,
            gates=chalmers_native_gate_set,
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 03/05/21."""
        return None
