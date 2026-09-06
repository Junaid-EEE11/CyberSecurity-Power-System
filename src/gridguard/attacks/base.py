"""Base attack interface and metadata structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class AttackMetadata:
    """Detailed provenance metadata for a simulated attack event."""

    attack_id: str
    attack_type: str
    start_timestep: int
    end_timestep: int
    target_nodes: List[int]
    target_node_names: List[str]
    target_features: List[str]
    attack_strength: float
    seed: int
    extra_info: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    """Abstract base class for simulated False Data Injection Attacks (FDIAs)."""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def apply(
        self,
        df_test: pd.DataFrame,
        start_step: int,
        duration: int,
        target_nodes: List[int],
        strength: float,
        target_features: Optional[List[str]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, AttackMetadata]:
        """Apply attack manipulation to telemetry DataFrame.

        Args:
            df_test: Original test DataFrame with clean values.
            start_step: Starting timestep integer.
            duration: Number of timesteps attack persists.
            target_nodes: List of target node IDs.
            strength: Attack scaling or magnitude parameter.
            target_features: List of feature names to manipulate.
            extra_context: Additional feeder or simulator context if needed.

        Returns:
            Tuple of (modified_DataFrame, AttackMetadata).
        """
        pass
