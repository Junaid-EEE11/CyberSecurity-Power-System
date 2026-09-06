"""Stochastic load profile generator with temporal autocorrelation and diurnal cycles."""

from typing import Dict, List, Optional
import numpy as np


class LoadProfileGenerator:
    """Generates synthetic multi-day stochastic load profiles for distribution feeders.

    Mathematical Formulation:
      multiplier(t, i) = mu_base(t) * w_day(t) * (1 + delta_system(t) + delta_indiv(t, i) + xi(t, i))

      where:
        mu_base(t): Normalized diurnal load shape with dual peaks (morning ~08:00, evening ~19:00).
        w_day(t): Weekday (1.00) vs Weekend (0.85) multiplier.
        delta_system(t): Low-frequency common grid-wide variation.
        delta_indiv(t, i): Load-specific static scaling factor.
        xi(t, i): AR(1) temporally autocorrelated stochastic noise:
                  xi(t, i) = phi * xi(t-1, i) + sqrt(1 - phi^2) * eps(t, i),  eps ~ N(0, sigma^2).
    """

    def __init__(
        self,
        load_names: List[str],
        resolution_minutes: int = 15,
        phi_ar1: float = 0.85,
        noise_std: float = 0.05,
        system_noise_std: float = 0.03,
        clip_min: float = 0.15,
        clip_max: float = 2.20,
        seed: int = 42,
    ):
        self.load_names = sorted(load_names)
        self.num_loads = len(self.load_names)
        self.resolution_minutes = resolution_minutes
        self.steps_per_day = (24 * 60) // resolution_minutes
        self.phi_ar1 = phi_ar1
        self.noise_std = noise_std
        self.system_noise_std = system_noise_std
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _diurnal_base_shape(self) -> np.ndarray:
        """Construct a smooth 24-hour dual-peak normalized diurnal load curve."""
        t_hours = np.linspace(0, 24, self.steps_per_day, endpoint=False)
        # Night trough (~0.5), morning peak at 08:30 (~1.1), afternoon plateau (~0.95), evening peak at 19:30 (~1.25)
        shape = (
            0.65
            + 0.25 * np.exp(-0.5 * ((t_hours - 8.5) / 2.5) ** 2)
            + 0.45 * np.exp(-0.5 * ((t_hours - 19.5) / 3.0) ** 2)
            + 0.10 * np.sin(2 * np.pi * (t_hours - 6.0) / 24.0)
        )
        return shape / np.mean(shape)

    def generate_profiles(self, num_days: int) -> np.ndarray:
        """Generate load multipliers for all loads across time.

        Args:
            num_days: Number of simulation days.

        Returns:
            2D numpy array of shape (T, num_loads) where T = num_days * steps_per_day.
        """
        T = num_days * self.steps_per_day
        diurnal_day = self._diurnal_base_shape()

        # Replicate diurnal shape for all days
        base_curve = np.tile(diurnal_day, num_days)

        # Apply weekday vs weekend pattern (day 5, 6 are weekend)
        day_indices = np.repeat(np.arange(num_days), self.steps_per_day)
        weekend_mask = (day_indices % 7 == 5) | (day_indices % 7 == 6)
        day_weights = np.where(weekend_mask, 0.85, 1.00)
        base_curve = base_curve * day_weights

        # Common system-level stochastic variation
        system_noise = np.zeros(T)
        sys_ar = 0.0
        for t in range(T):
            sys_ar = self.phi_ar1 * sys_ar + np.sqrt(1 - self.phi_ar1**2) * self.rng.normal(
                0, self.system_noise_std
            )
            system_noise[t] = sys_ar

        # Load-specific AR(1) processes
        ar_noise = np.zeros((T, self.num_loads))
        ar_state = np.zeros(self.num_loads)
        innovations = self.rng.normal(0, self.noise_std, size=(T, self.num_loads))

        for t in range(T):
            ar_state = self.phi_ar1 * ar_state + np.sqrt(1 - self.phi_ar1**2) * innovations[t]
            ar_noise[t] = ar_state

        # Combine components
        profiles = (
            base_curve[:, np.newaxis]
            * (1.0 + system_noise[:, np.newaxis])
            * (1.0 + ar_noise)
        )

        # Apply physical clipping
        profiles = np.clip(profiles, self.clip_min, self.clip_max)
        return profiles
