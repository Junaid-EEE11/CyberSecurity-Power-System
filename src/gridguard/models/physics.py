"""B5: Physics-Residual-Only Detector baseline."""

from typing import Optional, Tuple
import numpy as np

from gridguard.simulation.ybus import YBusModel


class PhysicsResidualDetector:
    """Baseline B5: Detects attacks using pure physical admittance/power-flow consistency equations."""

    def __init__(self, ybus_model: YBusModel, name: str = "B5_PhysicsResidualOnly"):
        self.name = name
        self.ybus = ybus_model
        self.num_nodes = ybus_model.num_nodes

    def fit(self, X_train: Optional[np.ndarray] = None) -> "PhysicsResidualDetector":
        """Physics model requires no training parameters."""
        return self

    def score(
        self,
        V_mag: np.ndarray,
        V_ang_rad: np.ndarray,
        P_inj_pu: np.ndarray,
        Q_inj_pu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate node-level and global physics consistency anomaly scores.

        Args:
            V_mag: [N_samples, N_nodes] (pu or volts)
            V_ang_rad: [N_samples, N_nodes] (radians)
            P_inj_pu: [N_samples, N_nodes] (pu)
            Q_inj_pu: [N_samples, N_nodes] (pu)

        Returns:
            Tuple of (global_scores [N_samples], node_residuals [N_samples, N_nodes]).
        """
        N_samples = V_mag.shape[0]
        node_residuals = np.zeros((N_samples, self.num_nodes), dtype=np.float64)

        for i in range(N_samples):
            res_node = self.ybus.calculate_physics_residual_numpy(
                V_mag=V_mag[i],
                V_ang_rad=V_ang_rad[i],
                P_inj_pu=P_inj_pu[i],
                Q_inj_pu=Q_inj_pu[i],
            )
            node_residuals[i] = res_node

        global_scores = np.max(node_residuals, axis=-1)
        return global_scores, node_residuals
