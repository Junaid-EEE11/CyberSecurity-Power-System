"""Proposed Method (B7): Physics-Guided Graph-Temporal Anomaly Detector."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gridguard.data.scaling import FeatureScaler
from gridguard.models.graph_temporal import GraphSAGELayer, GraphTemporalNet
from gridguard.simulation.ybus import YBusModel, TorchPhysicsLayer
from gridguard.utils.logging import get_logger

logger = get_logger("gridguard.models.proposed")


class ProposedPhysicsGraphTemporalDetector:
    """Proposed Model (B7): Physics-Guided Graph-Temporal Reconstruction and Anomaly Detector."""

    def __init__(
        self,
        adj_matrix: np.ndarray,
        ybus_model: YBusModel,
        scaler: FeatureScaler,
        graph_hidden_dim: int = 64,
        graph_out_dim: int = 64,
        gru_hidden_dim: int = 64,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        epochs: int = 30,
        lambda_physics: float = 0.1,
        w_rec: float = 0.5,
        w_phys: float = 0.5,
        top_k: int = 3,
        device: str = "cpu",
        name: str = "B7_Proposed_Physics_GTAE",
    ):
        self.name = name
        self.adj_matrix = adj_matrix
        self.ybus = ybus_model
        self.scaler = scaler
        self.num_nodes = adj_matrix.shape[0]
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_out_dim = graph_out_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_physics = lambda_physics
        self.w_rec = w_rec
        self.w_phys = w_phys
        self.top_k = top_k
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")

        # Row-normalized adjacency
        deg = np.sum(adj_matrix, axis=1)
        deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
        adj_norm = deg_inv[:, np.newaxis] * adj_matrix
        self.adj_norm = torch.tensor(adj_norm, dtype=torch.float32, device=self.device)

        # Physics loss layer
        self.physics_layer = self.ybus.get_torch_physics_layer(self.device)

        self.model: Optional[GraphTemporalNet] = None
        self.fitted: bool = False

        # Calibration score normalization parameters
        self.calib_mean_rec: float = 0.0
        self.calib_std_rec: float = 1.0
        self.calib_mean_phys: float = 0.0
        self.calib_std_phys: float = 1.0

    def fit(self, X_windows: np.ndarray, Y_targets: np.ndarray) -> "ProposedPhysicsGraphTemporalDetector":
        """Train proposed model on clean training data using joint reconstruction and physics loss.

        Args:
            X_windows: [N_samples, T_win, N_nodes, N_feat]
            Y_targets: [N_samples, N_nodes, N_feat]
        """
        N_samples, T_win, N_nodes, N_feat = X_windows.shape

        self.model = GraphTemporalNet(
            num_nodes=self.num_nodes,
            in_features=N_feat,
            graph_hidden_dim=self.graph_hidden_dim,
            graph_out_dim=self.graph_out_dim,
            gru_hidden_dim=self.gru_hidden_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        X_t = torch.tensor(X_windows, dtype=torch.float32)
        Y_t = torch.tensor(Y_targets, dtype=torch.float32)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        recon_criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss_epoch = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()

                recon = self.model(bx, self.adj_norm)
                l_rec = recon_criterion(recon, by)

                # Physics loss on reconstructed physical values
                if self.lambda_physics > 0.0:
                    # Inverse scale reconstructed features to physical units
                    # Features: [0: v_mag_pu, 1: v_ang_rad, 2: p_inj_pu, 3: q_inj_pu]
                    v_mag_pu = self.scaler.inverse_transform_tensor(recon[:, :, 0], 0)
                    v_ang_rad = self.scaler.inverse_transform_tensor(recon[:, :, 1], 1)
                    p_inj_pu = self.scaler.inverse_transform_tensor(recon[:, :, 2], 2)
                    q_inj_pu = self.scaler.inverse_transform_tensor(recon[:, :, 3], 3)

                    # Approximate nominal voltage in volts (e.g. 2400V or based on node base)
                    # Use pu voltage directly scaled by base
                    v_real = v_mag_pu * 2401.77 * torch.cos(v_ang_rad)
                    v_imag = v_mag_pu * 2401.77 * torch.sin(v_ang_rad)

                    l_phys = self.physics_layer(v_real, v_imag, p_inj_pu, q_inj_pu)
                    loss = l_rec + self.lambda_physics * l_phys
                else:
                    loss = l_rec

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss_epoch += loss.item()

            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                logger.info("Epoch %d/%d - Loss: %.5f", epoch + 1, self.epochs, total_loss_epoch / len(loader))

        self.fitted = True
        return self

    def fit_calibration_stats(
        self,
        X_calib: np.ndarray,
        Y_calib: np.ndarray,
        raw_V_mag: np.ndarray,
        raw_V_ang: np.ndarray,
        raw_P_inj: np.ndarray,
        raw_Q_inj: np.ndarray,
    ) -> None:
        """Fit normalization parameters (mean, std) for score components strictly on clean calibration data."""
        rec_scores, phys_scores = self._compute_raw_score_components(
            X_calib, Y_calib, raw_V_mag, raw_V_ang, raw_P_inj, raw_Q_inj
        )
        self.calib_mean_rec = float(np.mean(rec_scores))
        self.calib_std_rec = float(np.std(rec_scores)) or 1.0
        self.calib_mean_phys = float(np.mean(phys_scores))
        self.calib_std_phys = float(np.std(phys_scores)) or 1.0

    def _compute_raw_score_components(
        self,
        X_windows: np.ndarray,
        Y_targets: np.ndarray,
        raw_V_mag: np.ndarray,
        raw_V_ang: np.ndarray,
        raw_P_inj: np.ndarray,
        raw_Q_inj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute unnormalized node reconstruction error and node physics residual."""
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be fitted before scoring.")

        self.model.eval()
        N_samples = X_windows.shape[0]
        X_t = torch.tensor(X_windows, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            recon = self.model(X_t, self.adj_norm).cpu().numpy()

        # Node reconstruction MSE [N_samples, N_nodes]
        node_rec_err = np.mean((Y_targets - recon) ** 2, axis=-1)

        # Node physics residual [N_samples, N_nodes]
        node_phys_res = np.zeros((N_samples, self.num_nodes), dtype=np.float64)
        for i in range(N_samples):
            node_phys_res[i] = self.ybus.calculate_physics_residual_numpy(
                V_mag=raw_V_mag[i],
                V_ang_rad=raw_V_ang[i],
                P_inj_pu=raw_P_inj[i],
                Q_inj_pu=raw_Q_inj[i],
            )

        return node_rec_err, node_phys_res

    def score(
        self,
        X_windows: np.ndarray,
        Y_targets: np.ndarray,
        raw_V_mag: np.ndarray,
        raw_V_ang: np.ndarray,
        raw_P_inj: np.ndarray,
        raw_Q_inj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate calibrated composite anomaly scores.

        Returns:
            Tuple of:
              - global_composite_scores: [N_samples]
              - node_composite_scores: [N_samples, N_nodes]
              - node_rec_scores: [N_samples, N_nodes]
              - node_phys_scores: [N_samples, N_nodes]
        """
        rec_err, phys_res = self._compute_raw_score_components(
            X_windows, Y_targets, raw_V_mag, raw_V_ang, raw_P_inj, raw_Q_inj
        )

        # Standardize using calibration statistics
        std_rec = (rec_err - self.calib_mean_rec) / self.calib_std_rec
        std_phys = (phys_res - self.calib_mean_phys) / self.calib_std_phys

        # Fused node score: w_rec * std_rec + w_phys * std_phys
        node_composite = self.w_rec * std_rec + self.w_phys * std_phys

        # Global score: top-k mean across nodes
        N_samples = node_composite.shape[0]
        k = min(self.top_k, self.num_nodes)
        top_k_vals = np.partition(node_composite, -k, axis=-1)[:, -k:]
        global_composite = np.mean(top_k_vals, axis=-1)

        return global_composite, node_composite, std_rec, std_phys
