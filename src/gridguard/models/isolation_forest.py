"""B2: Isolation Forest anomaly detector baseline."""

from typing import Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """Baseline B2: Tree-based space-partitioning anomaly detector."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.01,
        name: str = "B2_IsolationForest",
        seed: int = 42,
    ):
        self.name = name
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.seed = seed
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )
        self.num_nodes: Optional[int] = None
        self.num_features: Optional[int] = None
        self.fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        """Fit Isolation Forest on flattened clean training data.

        Args:
            X_train: [N_samples, N_nodes, N_features]
        """
        N_samples, self.num_nodes, self.num_features = X_train.shape
        X_flat = X_train.reshape(N_samples, -1)
        self.model.fit(X_flat)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute anomaly scores (higher = more anomalous).

        Args:
            X: [N_samples, N_nodes, N_features]

        Returns:
            Tuple of (global_scores [N_samples], node_scores [N_samples, N_nodes]).
        """
        if not self.fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        N_samples = X.shape[0]
        X_flat = X.reshape(N_samples, -1)
        # score_samples returns opposite of anomaly score (lower is more anomalous)
        raw_scores = -self.model.score_samples(X_flat)  # [N_samples]

        # For node-level localization, calculate feature contribution heuristic (deviation from training mean)
        # Broadcast global score modulated by node magnitude
        node_dev = np.linalg.norm(X, axis=-1)  # [N_samples, N_nodes]
        node_weights = node_dev / (np.sum(node_dev, axis=-1, keepdims=True) + 1e-8)
        node_scores = raw_scores[:, np.newaxis] * node_weights * self.num_nodes

        return raw_scores, node_scores
