"""Feature extraction and measurement formulation for distribution bus-phases."""

from typing import Any, Dict, List, Tuple
import numpy as np
import opendssdirect as dss

from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.simulation.feeder import FeederModel
from gridguard.utils.logging import get_logger

logger = get_logger("gridguard.simulation.measurements")


def extract_measurements(
    dss_interface: OpenDSSInterface,
    feeder_model: FeederModel,
    base_mva: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Extract full vector of electrical measurements aligned with feeder graph nodes.

    Measurements per bus-phase node:
      1. v_mag_pu: Voltage magnitude in per unit.
      2. v_ang_rad: Voltage angle in radians.
      3. sin_theta: Sine of voltage angle.
      4. cos_theta: Cosine of voltage angle.
      5. v_real_volts: Real voltage in physical Volts.
      6. v_imag_volts: Imaginary voltage in physical Volts.
      7. p_inj_pu: Net active power injection into the bus in per unit (+gen, -load).
      8. q_inj_pu: Net reactive power injection in per unit.
      9. mask: Binary mask (1 if node physically exists and converged, 0 otherwise).

    Args:
        dss_interface: OpenDSS session.
        feeder_model: Feeder model with node indexing.
        base_mva: Base MVA for per-unit power normalizations.

    Returns:
        Dictionary of measurement arrays (shape: (N,)).
    """
    node_names = feeder_model.node_names
    num_nodes = feeder_model.num_nodes
    node_to_idx = feeder_model.node_to_idx

    # 1. Complex voltages
    v_complex = dss_interface.get_complex_bus_voltages()
    v_real_volts = np.real(v_complex)
    v_imag_volts = np.imag(v_complex)

    # 2. Voltage magnitude (pu) and angle (rad)
    v_mag_pu = dss_interface.get_pu_voltage_magnitudes()
    v_ang_rad = np.angle(v_complex)
    sin_theta = np.sin(v_ang_rad)
    cos_theta = np.cos(v_ang_rad)

    # 3. Power injections: Sum over connected loads and generators
    base_kw = base_mva * 1000.0
    base_kvar = base_mva * 1000.0

    p_inj_kw = np.zeros(num_nodes, dtype=np.float64)
    q_inj_kvar = np.zeros(num_nodes, dtype=np.float64)

    # Loads: load power is consumed (negative injection)
    for load_name in dss.Loads.AllNames():
        dss.Loads.Name(load_name)
        buses = dss.CktElement.BusNames()
        powers = dss.CktElement.Powers()  # [P1, Q1, P2, Q2, ...] in kW, kvar
        if not buses or not powers:
            continue
        b_name = buses[0].lower()
        b_parts = b_name.split(".")
        b_base = b_parts[0]
        phases = dss.Loads.Phases()
        phase_list = b_parts[1:] if len(b_parts) > 1 else [str(p) for p in range(1, phases + 1)]

        for i, p_str in enumerate(phase_list):
            node_key = f"{b_base}.{p_str}"
            if node_key in node_to_idx and (i * 2 + 1) < len(powers):
                idx = node_to_idx[node_key]
                p_inj_kw[idx] -= powers[i * 2]
                q_inj_kvar[idx] -= powers[i * 2 + 1]

    # Generators / PV (if present): generation is positive injection
    for gen_name in dss.Generators.AllNames():
        dss.Generators.Name(gen_name)
        buses = dss.CktElement.BusNames()
        powers = dss.CktElement.Powers()
        if not buses or not powers:
            continue
        b_name = buses[0].lower()
        b_parts = b_name.split(".")
        b_base = b_parts[0]
        phases = dss.Generators.Phases()
        phase_list = b_parts[1:] if len(b_parts) > 1 else [str(p) for p in range(1, phases + 1)]

        for i, p_str in enumerate(phase_list):
            node_key = f"{b_base}.{p_str}"
            if node_key in node_to_idx and (i * 2 + 1) < len(powers):
                idx = node_to_idx[node_key]
                p_inj_kw[idx] += powers[i * 2]
                q_inj_kvar[idx] += powers[i * 2 + 1]

    p_inj_pu = p_inj_kw / base_kw
    q_inj_pu = q_inj_kvar / base_kvar
    mask = np.ones(num_nodes, dtype=np.float32)

    return {
        "v_mag_pu": v_mag_pu,
        "v_ang_rad": v_ang_rad,
        "sin_theta": sin_theta,
        "cos_theta": cos_theta,
        "v_real_volts": v_real_volts,
        "v_imag_volts": v_imag_volts,
        "p_inj_pu": p_inj_pu,
        "q_inj_pu": q_inj_pu,
        "mask": mask,
    }
