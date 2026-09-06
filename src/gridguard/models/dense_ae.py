"""B3: Dense Autoencoder baseline (non-temporal, no graph structure)."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DenseAENet(nn.Module):
    """Dense bottleneck autoencoder architecture."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, bottleneck_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class DenseAutoencoderDetector:
    """Baseline B3: Non-temporal Dense Autoencoder detector."""

    def __init__(
        self,
        hidden_dim: int = 128,
        bottleneck_dim: int = 32,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 20,
        device: str = "cpu",
        name: str = "B3_Dense_AE",
    ):
        self.name = name
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")

        self.model: Optional[DenseAENet] = None
        self.num_nodes: Optional[int] = None
        self.num_features: Optional[int] = None
        self.fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "DenseAutoencoderDetector":
        """Fit Dense AE on clean training samples.

        Args:
            X_train: [N_samples, N_nodes, N_features]
        """
        N_samples, self.num_nodes, self.num_features = X_train.shape
        input_dim = self.num_nodes * self.num_features
        X_flat = torch.tensor(X_train.reshape(N_samples, input_dim), dtype=torch.float32)

        self.model = DenseAENet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout,
        ).to(self.device)

        dataset = TensorDataset(X_flat)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                bx = batch[0].to(self.device)
                optimizer.zero_grad()
                recon = self.model(bx)
                loss = criterion(recon, bx)
                loss.backward()
                optimizer.step()

        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate reconstruction error anomaly scores.

        Args:
            X: [N_samples, N_nodes, N_features]

        Returns:
            Tuple of (global_scores [N_samples], node_scores [N_samples, N_nodes]).
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be fitted before scoring.")

        self.model.eval()
        N_samples = X.shape[0]
        input_dim = self.num_nodes * self.num_features
        X_flat = torch.tensor(X.reshape(N_samples, input_dim), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            recon_flat = self.model(X_flat).cpu().numpy()

        recon = recon_flat.reshape(N_samples, self.num_nodes, self.num_features)
        # Node reconstruction MSE
        node_scores = np.mean((X - recon) ** 2, axis=-1)  # [N_samples, N_nodes]
        global_scores = np.max(node_scores, axis=-1)  # [N_samples]

        return global_scores, node_scores
