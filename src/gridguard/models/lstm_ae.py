"""B4: LSTM Autoencoder baseline (temporal sequence modeling, no graph topology)."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAENet(nn.Module):
    """Sequence-to-sequence LSTM autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Window_Length, Input_Dim]
        _, (h_n, _) = self.encoder(x)
        # h_n[-1]: [Batch, Hidden_Dim]
        x_recon_last = self.decoder_mlp(h_n[-1])
        return x_recon_last


class LSTMAutoencoderDetector:
    """Baseline B4: Temporal LSTM Autoencoder without graph message passing."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 20,
        device: str = "cpu",
        name: str = "B4_LSTM_AE",
    ):
        self.name = name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")

        self.model: Optional[LSTMAENet] = None
        self.num_nodes: Optional[int] = None
        self.num_features: Optional[int] = None
        self.window_length: Optional[int] = None
        self.fitted: bool = False

    def fit(self, X_windows: np.ndarray, Y_targets: np.ndarray) -> "LSTMAutoencoderDetector":
        """Fit LSTM AE on temporal sequences.

        Args:
            X_windows: [N_samples, T_window, N_nodes, N_features]
            Y_targets: [N_samples, N_nodes, N_features]
        """
        N_samples, self.window_length, self.num_nodes, self.num_features = X_windows.shape
        input_dim = self.num_nodes * self.num_features

        X_flat = torch.tensor(X_windows.reshape(N_samples, self.window_length, input_dim), dtype=torch.float32)
        Y_flat = torch.tensor(Y_targets.reshape(N_samples, input_dim), dtype=torch.float32)

        self.model = LSTMAENet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        dataset = TensorDataset(X_flat, Y_flat)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                recon = self.model(bx)
                loss = criterion(recon, by)
                loss.backward()
                optimizer.step()

        self.fitted = True
        return self

    def score(self, X_windows: np.ndarray, Y_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute reconstruction error anomaly scores.

        Args:
            X_windows: [N_samples, T_window, N_nodes, N_features]
            Y_targets: [N_samples, N_nodes, N_features]

        Returns:
            Tuple of (global_scores [N_samples], node_scores [N_samples, N_nodes]).
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be fitted before scoring.")

        self.model.eval()
        N_samples = X_windows.shape[0]
        input_dim = self.num_nodes * self.num_features
        X_flat = torch.tensor(X_windows.reshape(N_samples, self.window_length, input_dim), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            recon_flat = self.model(X_flat).cpu().numpy()

        recon = recon_flat.reshape(N_samples, self.num_nodes, self.num_features)
        node_scores = np.mean((Y_targets - recon) ** 2, axis=-1)  # [N_samples, N_nodes]
        global_scores = np.max(node_scores, axis=-1)

        return global_scores, node_scores
