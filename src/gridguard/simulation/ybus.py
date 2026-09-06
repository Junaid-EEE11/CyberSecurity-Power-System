"""Admittance matrix (Ybus) extraction and physics consistency calculations."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.utils.logging import get_logger

logger = get_logger("gridguard.simulation.ybus")


class YBusModel:
    """Manages the network admittance matrix and power-flow physics consistency calculations."""

    def __init__(self, dss_interface: OpenDSSInterface, base_mva: float = 1.0):
        """Extract and align admittance matrix with circuit bus-phase nodes.

        Args:
            dss_interface: Active OpenDSS interface.
            base_mva: Base MVA for per-unit power normalizations (default 1.0 MVA).
        """
        self.dss = dss_interface
        self.base_mva = base_mva
        self.base_va = base_mva * 1e6
        self.node_names: List[str] = self.dss.get_all_node_names()
        self.num_nodes: int = len(self.node_names)

        # Extract system Y matrix
        self.Y_dense, _ = self.dss.get_system_y_matrix()
        self.G = np.real(self.Y_dense)
        self.B = np.imag(self.Y_dense)

        logger.info("Initialized YBusModel with %d nodes, base_mva=%.2f", self.num_nodes, self.base_mva)

    def get_dense_ybus(self) -> np.ndarray:
        """Get dense complex Ybus admittance matrix."""
        return self.Y_dense

    def calculate_currents_and_powers(
        self, V_complex: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate complex current injections and complex power from voltage vector.

        Args:
            V_complex: 1D numpy array of complex voltages in Volts (shape: N,).

        Returns:
            Tuple of (I_complex [Amperes], P [Watts], Q [VARs]).
        """
        I_complex = self.Y_dense @ V_complex
        S_complex = V_complex * np.conj(I_complex)
        P = np.real(S_complex)
        Q = np.imag(S_complex)
        return I_complex, P, Q

    def calculate_physics_residual_numpy(
        self,
        V_mag: np.ndarray,
        V_ang_rad: np.ndarray,
        P_inj_pu: np.ndarray,
        Q_inj_pu: np.ndarray,
        base_kv_dict: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute normalized nodal physics consistency residual in NumPy.

        Args:
            V_mag: Voltage magnitude (pu or Volts).
            V_ang_rad: Voltage angle in radians.
            P_inj_pu: Measured active injection in per unit.
            Q_inj_pu: Measured reactive injection in per unit.
            base_kv_dict: Optional bus base kV mapping.

        Returns:
            1D array of per-node residual magnitudes.
        """
        # Form complex voltage
        V_complex = V_mag * np.exp(1j * V_ang_rad)
        _, P_calc_watts, Q_calc_watts = self.calculate_currents_and_powers(V_complex)

        P_calc_pu = P_calc_watts / self.base_va
        Q_calc_pu = Q_calc_watts / self.base_va

        res_p = np.abs(P_calc_pu - P_inj_pu)
        res_q = np.abs(Q_calc_pu - Q_inj_pu)
        res_total = np.sqrt(res_p**2 + res_q**2)
        return res_total

    def get_torch_physics_layer(self, device: torch.device) -> "TorchPhysicsLayer":
        """Get differentiable PyTorch layer for calculating physics loss."""
        return TorchPhysicsLayer(self.G, self.B, self.base_va, device)


class TorchPhysicsLayer(nn.Module):
    """Differentiable PyTorch module computing Kirchhoff power-flow consistency residuals."""

    def __init__(
        self, G_np: np.ndarray, B_np: np.ndarray, base_va: float, device: torch.device
    ):
        super().__init__()
        self.num_nodes = G_np.shape[0]
        self.base_va = float(base_va)
        self.register_buffer("G", torch.tensor(G_np, dtype=torch.float32, device=device))
        self.register_buffer("B", torch.tensor(B_np, dtype=torch.float32, device=device))

    def forward(
        self,
        v_real: torch.Tensor,
        v_imag: torch.Tensor,
        p_inj_pu: torch.Tensor,
        q_inj_pu: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing physics consistency residual loss.

        Args:
            v_real: Real voltage component in Volts [Batch, N]
            v_imag: Imaginary voltage component in Volts [Batch, N]
            p_inj_pu: Active power injection in per unit [Batch, N]
            q_inj_pu: Reactive power injection in per unit [Batch, N]
            mask: Optional boolean mask [Batch, N] of active nodes.

        Returns:
            Scalar mean physics loss.
        """
        # I_real = G @ v_real - B @ v_imag
        # I_imag = B @ v_real + G @ v_imag
        i_real = torch.matmul(v_real, self.G.t()) - torch.matmul(v_imag, self.B.t())
        i_imag = torch.matmul(v_real, self.B.t()) + torch.matmul(v_imag, self.G.t())

        # S = V * conj(I) = (v_r + j v_i) * (i_r - j i_i)
        # P = v_r * i_r + v_i * i_i
        # Q = v_i * i_r - v_r * i_i
        p_calc_w = v_real * i_real + v_imag * i_imag
        q_calc_var = v_imag * i_real - v_real * i_imag

        p_calc_pu = p_calc_w / self.base_va
        q_calc_pu = q_calc_var / self.base_va

        res_p = (p_calc_pu - p_inj_pu) ** 2
        res_q = (q_calc_pu - q_inj_pu) ** 2
        res_node = res_p + res_q

        if mask is not None:
            res_node = res_node * mask.float()
            denom = torch.clamp(mask.float().sum(), min=1.0)
            return res_node.sum() / denom
        return res_node.mean()
