"""B6: Graph-Temporal Autoencoder baseline (topology + temporal, no physics loss)."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class GraphSAGELayer(nn.Module):
    """GraphSAGE convolutional message passing layer with mean aggregation."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features, bias=False)
        self.linear_neigh = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [Batch, Num_Nodes, In_Features]
            adj_norm: [Num_Nodes, Num_Nodes] normalized row-stochastic adjacency matrix.

        Returns:
            [Batch, Num_Nodes, Out_Features]
        """
        # Self transform
        h_self = self.linear_self(x)
        # Neighbor aggregation: matmul on node dimension
        # x: [B, N, F] -> adj_norm @ x -> [B, N, F]
        neigh = torch.matmul(adj_norm, x)
        h_neigh = self.linear_neigh(neigh)

        out = self.activation(h_self + h_neigh)
        return self.dropout(out)


class GraphTemporalNet(nn.Module):
    """Spatial GraphSAGE followed by Per-Node Temporal GRU and Feature Decoder."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        graph_hidden_dim: int = 64,
        graph_out_dim: int = 64,
        gru_hidden_dim: int = 64,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features

        # Spatial GraphSAGE Encoder (2 layers)
        self.sage1 = GraphSAGELayer(in_features, graph_hidden_dim, dropout=dropout)
        self.sage2 = GraphSAGELayer(graph_hidden_dim, graph_out_dim, dropout=dropout)

        # Temporal Recurrent GRU (applied per node across time window)
        self.gru = nn.GRU(
            input_size=graph_out_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )

        # Decoder MLP mapping per-node final hidden state to reconstructed features
        self.decoder = nn.Sequential(
            nn.Linear(gru_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, in_features),
        )

    def forward(self, x_window: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass over sliding temporal window.

        Args:
            x_window: [Batch, Window_Length, Num_Nodes, In_Features]
            adj_norm: [Num_Nodes, Num_Nodes] normalized adjacency

        Returns:
            reconstructed_final: [Batch, Num_Nodes, In_Features]
        """
        B, T, N, F = x_window.shape

        # 1. Apply spatial GraphSAGE at each timestep
        # Reshape [B * T, N, F]
        x_flat = x_window.view(B * T, N, F)
        h1 = self.sage1(x_flat, adj_norm)  # [B * T, N, graph_hidden]
        h2 = self.sage2(h1, adj_norm)      # [B * T, N, graph_out]

        # 2. Reshape for per-node temporal GRU: [B, T, N, graph_out] -> [(B * N), T, graph_out]
        h_spatial = h2.view(B, T, N, -1).permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)

        # 3. GRU forward pass
        _, h_n = self.gru(h_spatial)  # h_n: [1, B * N, gru_hidden]
        h_last = h_n[-1]              # [B * N, gru_hidden]

        # 4. Decode per-node features
        recon_flat = self.decoder(h_last)  # [B * N, In_Features]
        recon = recon_flat.view(B, N, F)   # [B, N, In_Features]

        return recon


class GraphTemporalDetector:
    """Baseline B6: Graph-Temporal Autoencoder without electrical physics loss."""

    def __init__(
        self,
        adj_matrix: np.ndarray,
        graph_hidden_dim: int = 64,
        graph_out_dim: int = 64,
        gru_hidden_dim: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 25,
        device: str = "cpu",
        name: str = "B6_GraphTemporal_AE",
    ):
        self.name = name
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_out_dim = graph_out_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")

        # Compute degree-normalized adjacency matrix D^{-1} A
        deg = np.sum(adj_matrix, axis=1)
        deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
        adj_norm = deg_inv[:, np.newaxis] * adj_matrix
        self.adj_norm = torch.tensor(adj_norm, dtype=torch.float32, device=self.device)

        self.model: Optional[GraphTemporalNet] = None
        self.fitted: bool = False

    def fit(self, X_windows: np.ndarray, Y_targets: np.ndarray) -> "GraphTemporalDetector":
        N_samples, T_win, N_nodes, N_feat = X_windows.shape

        self.model = GraphTemporalNet(
            num_nodes=self.num_nodes,
            in_features=N_feat,
            graph_hidden_dim=self.graph_hidden_dim,
            graph_out_dim=self.graph_out_dim,
            gru_hidden_dim=self.gru_hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        X_t = torch.tensor(X_windows, dtype=torch.float32)
        Y_t = torch.tensor(Y_targets, dtype=torch.float32)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                recon = self.model(bx, self.adj_norm)
                loss = criterion(recon, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

        self.fitted = True
        return self

    def score(self, X_windows: np.ndarray, Y_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be fitted before scoring.")

        self.model.eval()
        N_samples = X_windows.shape[0]
        X_t = torch.tensor(X_windows, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            recon = self.model(X_t, self.adj_norm).cpu().numpy()

        node_scores = np.mean((Y_targets - recon) ** 2, axis=-1)  # [N_samples, N_nodes]
        global_scores = np.max(node_scores, axis=-1)

        return global_scores, node_scores
