"""PyTorch Dataset classes for graph-temporal power system telemetry."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from gridguard.data.windows import create_sliding_windows


class GridTimeDataset(Dataset):
    """PyTorch Dataset serving sliding-window graph-temporal power flow samples."""

    def __init__(
        self,
        X_windows: np.ndarray,
        Y_targets: np.ndarray,
        is_attack: np.ndarray,
        compromised_nodes: np.ndarray,
        target_timesteps: np.ndarray,
    ):
        """Initialize dataset from pre-computed numpy arrays.

        Args:
            X_windows: [N_samples, T_window, N_nodes, N_features]
            Y_targets: [N_samples, N_nodes, N_features]
            is_attack: [N_samples]
            compromised_nodes: [N_samples, N_nodes]
            target_timesteps: [N_samples]
        """
        self.X = torch.tensor(X_windows, dtype=torch.float32)
        self.Y = torch.tensor(Y_targets, dtype=torch.float32)
        self.is_attack = torch.tensor(is_attack, dtype=torch.long)
        self.compromised_nodes = torch.tensor(compromised_nodes, dtype=torch.long)
        self.target_timesteps = torch.tensor(target_timesteps, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.X[idx],
            "y": self.Y[idx],
            "is_attack": self.is_attack[idx],
            "compromised_nodes": self.compromised_nodes[idx],
            "timestep": self.target_timesteps[idx],
        }


def get_data_loader(
    dataset: GridTimeDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a standard PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
