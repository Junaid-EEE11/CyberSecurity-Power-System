"""B1: Principal Component Analysis (PCA) reconstruction and SPE baseline."""

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


class PCADetector:
    """Baseline B1: Subspace reconstruction detector using PCA Squared Prediction Error (SPE)."""

    def __init__(self, n_components: float = 0.95, name: str = "B1_PCA_SPE", seed: int = 42):
        self.name = name
        self.n_components = n_components
        self.seed = seed
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.num_nodes: Optional[int] = None
        self.num_features: Optional[int] = None
        self.fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "PCADetector":
        """Fit PCA on flattened clean training data.

        Args:
            X_train: [N_samples, N_nodes, N_features]
        """
        N_samples, self.num_nodes, self.num_features = X_train.shape
        X_flat = X_train.reshape(N_samples, -1)
        self.pca.fit(X_flat)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute reconstruction SPE anomaly scores.

        Args:
            X: [N_samples, N_nodes, N_features]

        Returns:
            Tuple of (global_scores [N_samples], node_scores [N_samples, N_nodes]).
        """
        if not self.fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        N_samples = X.shape[0]
        X_flat = X.reshape(N_samples, -1)
        X_recon_flat = self.pca.inverse_transform(self.pca.transform(X_flat))
        X_recon = X_recon_flat.reshape(N_samples, self.num_nodes, self.num_features)

        # Node-level reconstruction error: squared error averaged over features
        node_res = np.mean((X - X_recon) ** 2, axis=-1)  # [N_samples, N_nodes]
        global_scores = np.max(node_res, axis=-1)  # [N_samples]

        return global_scores, node_res
