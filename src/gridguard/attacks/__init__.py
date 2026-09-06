"""Synthetic cyberattack generator library for defensive research."""

from gridguard.attacks.base import BaseAttack, AttackMetadata
from gridguard.attacks.step import StepFDIA
from gridguard.attacks.ramp import RampFDIA
from gridguard.attacks.replay import ReplayFDIA
from gridguard.attacks.coordinated import CoordinatedSpatialFDIA
from gridguard.attacks.operating_point import OperatingPointSubstitutionFDIA
from gridguard.attacks.scheduler import AttackScheduler

__all__ = [
    "BaseAttack",
    "AttackMetadata",
    "StepFDIA",
    "RampFDIA",
    "ReplayFDIA",
    "CoordinatedSpatialFDIA",
    "OperatingPointSubstitutionFDIA",
    "AttackScheduler",
]
