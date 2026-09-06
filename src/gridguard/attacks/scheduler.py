"""Attack scheduler to generate non-overlapping cyberattack scenarios across test split."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from gridguard.attacks.base import AttackMetadata
from gridguard.attacks.step import StepFDIA
from gridguard.attacks.ramp import RampFDIA
from gridguard.attacks.replay import ReplayFDIA
from gridguard.attacks.coordinated import CoordinatedSpatialFDIA
from gridguard.attacks.operating_point import OperatingPointSubstitutionFDIA
from gridguard.utils.logging import get_logger

logger = get_logger("gridguard.attacks.scheduler")


class AttackScheduler:
    """Schedules synthetic attack scenarios across the held-out test split."""

    def __init__(self, attack_config: Dict[str, Any], seed: int = 42):
        self.config = attack_config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def schedule_attacks(
        self,
        df_test: pd.DataFrame,
        num_nodes: int,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, List[AttackMetadata]]:
        """Inject non-overlapping sequence of all configured attacks into test DataFrame.

        Args:
            df_test: Clean test split DataFrame.
            num_nodes: Total number of bus-phase nodes.
            extra_context: Additional feeder and simulator context.

        Returns:
            Tuple of (attacked_DataFrame, list_of_AttackMetadata).
        """
        df = df_test.copy()
        # Initialize attack ground truth columns
        df["is_attack"] = 0
        df["attack_id"] = "none"
        df["attack_type"] = "none"
        df["attack_strength"] = 0.0
        df["is_compromised"] = 0

        timesteps = sorted(df["timestep"].unique())
        total_steps = len(timesteps)
        min_step = timesteps[0]
        max_step = timesteps[-1]

        metadata_list: List[AttackMetadata] = []

        # Instantiated attack generators
        generators = {
            "step": StepFDIA(seed=self.seed),
            "ramp": RampFDIA(seed=self.seed + 1),
            "replay": ReplayFDIA(seed=self.seed + 2),
            "coordinated": CoordinatedSpatialFDIA(seed=self.seed + 3),
            "operating_point": OperatingPointSubstitutionFDIA(seed=self.seed + 4),
        }

        # Plan non-overlapping intervals
        # Leave buffer periods of clean operation between attacks
        buffer_steps = max(8, total_steps // 25)
        curr_step = min_step + buffer_steps

        # Sequence of attack types and intensities
        attack_plan = [
            # Step attacks (weak, medium, strong)
            ("step", 0.03, 8),
            ("step", 0.08, 10),
            ("step", 0.15, 8),
            # Ramp attacks (weak, medium, strong)
            ("ramp", 0.03, 12),
            ("ramp", 0.08, 14),
            ("ramp", 0.15, 12),
            # Replay attack (medium lag, long lag)
            ("replay", 24.0, 10),
            ("replay", 48.0, 12),
            # Coordinated attacks (medium, strong)
            ("coordinated", 0.08, 10),
            ("coordinated", 0.15, 10),
            # Unseen attack E (operating point substitution)
            ("operating_point", 0.20, 10),
            ("operating_point", 0.30, 12),
        ]

        for att_type, strength, duration in attack_plan:
            if curr_step + duration + buffer_steps > max_step:
                logger.info("Reached end of test window; concluding attack scheduling.")
                break

            gen = generators.get(att_type)
            if gen is None:
                continue

            # Target nodes selection
            num_targets = max(2, int(num_nodes * 0.06))
            target_nodes = sorted(list(self.rng.choice(num_nodes, size=num_targets, replace=False)))

            df, meta = gen.apply(
                df_test=df,
                start_step=curr_step,
                duration=duration,
                target_nodes=target_nodes,
                strength=strength,
                extra_context=extra_context,
            )
            metadata_list.append(meta)
            logger.info("Scheduled %s from step %d to %d (strength=%.2f)", meta.attack_type, curr_step, meta.end_timestep, strength)

            curr_step = meta.end_timestep + buffer_steps + 1

        return df, metadata_list
