"""Interface for OpenDSS distribution feeder simulations using OpenDSSDirect.py."""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import opendssdirect as dss

from gridguard.utils.logging import get_logger

logger = get_logger("gridguard.simulation.opendss")


class OpenDSSInterface:
    """Wrapper around OpenDSSDirect.py providing safe execution and data access."""

    def __init__(self, dss_file: str):
        """Initialize and compile OpenDSS model.

        Args:
            dss_file: Path to master .dss file.
        """
        if not os.path.exists(dss_file):
            # Check relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            alt_path = os.path.join(base_dir, dss_file)
            if os.path.exists(alt_path):
                dss_file = alt_path
            else:
                raise FileNotFoundError(f"OpenDSS file not found: {dss_file}")
        self.dss_file = os.path.abspath(dss_file)
        self.compiled = False
        self.compile_circuit()

    def compile_circuit(self) -> None:
        """Clear and compile the circuit preserving python process CWD."""
        orig_cwd = os.getcwd()
        dss.Command("Clear")
        cmd = f'Compile "{self.dss_file}"'
        dss.Command(cmd)
        try:
            os.chdir(orig_cwd)
        except Exception:
            pass
        self.compiled = True
        logger.info("Compiled OpenDSS circuit: %s", self.dss_file)

    def solve(self, max_iter: int = 100) -> bool:
        """Solve power flow and verify convergence.

        Args:
            max_iter: Maximum power flow iterations.

        Returns:
            True if converged, False otherwise.
        """
        dss.Solution.MaxControlIterations(max_iter)
        dss.Solution.MaxIterations(max_iter)
        dss.Solution.Solve()
        converged = bool(dss.Solution.Converged())
        if not converged:
            logger.warning("OpenDSS power flow did not converge!")
        return converged

    def get_total_losses(self) -> Tuple[float, float]:
        """Get total circuit active and reactive losses in kW and kvar."""
        losses = dss.Circuit.Losses()
        return float(losses[0]) / 1000.0, float(losses[1]) / 1000.0

    def get_all_node_names(self) -> List[str]:
        """Get canonical list of all bus-phase nodes in the circuit (e.g. '150.1')."""
        return [str(n).lower() for n in dss.Circuit.AllNodeNames()]

    def get_all_bus_names(self) -> List[str]:
        """Get list of all bus names."""
        return [str(b).lower() for b in dss.Circuit.AllBusNames()]

    def get_complex_bus_voltages(self) -> np.ndarray:
        """Get complex nodal voltages V aligned with Circuit.AllNodeNames().

        Returns:
            1D numpy array of complex voltages in Volts.
        """
        v_raw = dss.Circuit.AllBusVolts()
        v_real = np.array(v_raw[0::2], dtype=np.float64)
        v_imag = np.array(v_raw[1::2], dtype=np.float64)
        return v_real + 1j * v_imag

    def get_pu_voltage_magnitudes(self) -> np.ndarray:
        """Get per-unit voltage magnitudes aligned with Circuit.AllNodeNames()."""
        return np.array(dss.Circuit.AllBusMagPu(), dtype=np.float64)

    def get_system_y_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Get System Y admittance matrix as a dense complex matrix.

        Returns:
            Tuple of (Y_matrix (N, N), node_names).
        """
        node_names = self.get_all_node_names()
        N = len(node_names)
        y_raw = dss.Circuit.SystemY()
        y_real = np.array(y_raw[0::2], dtype=np.float64)
        y_imag = np.array(y_raw[1::2], dtype=np.float64)
        y_complex = y_real + 1j * y_imag
        Y = y_complex.reshape((N, N))
        return Y, node_names

    def set_load_multiplier(self, multiplier: float) -> None:
        """Scale all loads uniformly by a global multiplier."""
        dss.Solution.LoadMult(float(multiplier))

    def get_load_names(self) -> List[str]:
        """Get names of all loads."""
        return [str(name).lower() for name in dss.Loads.AllNames()]

    def set_individual_load(
        self, load_name: str, kw: Optional[float] = None, kvar: Optional[float] = None
    ) -> None:
        """Set active and reactive power for a specific load."""
        dss.Loads.Name(load_name)
        if kw is not None:
            dss.Loads.kW(float(kw))
        if kvar is not None:
            dss.Loads.kvar(float(kvar))

    def get_load_data(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of all loads with nominal kW, kvar, bus connection, and phases."""
        load_data = {}
        for name in dss.Loads.AllNames():
            dss.Loads.Name(name)
            load_data[name.lower()] = {
                "bus": dss.CktElement.BusNames()[0].lower(),
                "kw": float(dss.Loads.kW()),
                "kvar": float(dss.Loads.kvar()),
                "phases": int(dss.Loads.Phases()),
                "kv": float(dss.Loads.kV()),
            }
        return load_data
