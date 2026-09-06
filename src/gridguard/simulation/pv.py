"""Stochastic solar photovoltaic (PV) generation profile generator."""

from typing import List, Tuple
import numpy as np


class PVProfileGenerator:
    """Generates synthetic solar generation profiles with diurnal irradiance and cloud intermittency."""

    def __init__(
        self,
        num_pv_systems: int,
        resolution_minutes: int = 15,
        cloud_variability: float = 0.20,
        seed: int = 42,
    ):
        self.num_pv_systems = num_pv_systems
        self.resolution_minutes = resolution_minutes
        self.steps_per_day = (24 * 60) // resolution_minutes
        self.cloud_variability = cloud_variability
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _clear_sky_irradiance(self) -> np.ndarray:
        """Clear-sky solar irradiance profile spanning 06:00 to 18:00."""
        t_hours = np.linspace(0, 24, self.steps_per_day, endpoint=False)
        irradiance = np.zeros_like(t_hours)

        # Sun shines between 06:00 and 18:00 (12 hours)
        day_mask = (t_hours >= 6.0) & (t_hours <= 18.0)
        # Half-sine model
        irradiance[day_mask] = np.sin(np.pi * (t_hours[day_mask] - 6.0) / 12.0)
        return irradiance

    def generate_profiles(self, num_days: int) -> np.ndarray:
        """Generate solar generation multipliers [T, num_pv_systems] normalized in [0, 1].

        Args:
            num_days: Number of days.

        Returns:
            2D numpy array of shape (T, num_pv_systems).
        """
        T = num_days * self.steps_per_day
        clear_sky_day = self._clear_sky_irradiance()
        base_irradiance = np.tile(clear_sky_day, num_days)

        # Generate stochastic cloud cover / intermittency
        # Cloud events with beta-distributed shading
        cloud_factor = np.ones((T, self.num_pv_systems))
        for pv_idx in range(self.num_pv_systems):
            # Markov-like cloud passing
            shading = 1.0
            for t in range(T):
                if base_irradiance[t] > 0.05:
                    if self.rng.uniform() < 0.10:  # Cloud transition
                        shading = self.rng.uniform(1.0 - self.cloud_variability, 1.0)
                    cloud_factor[t, pv_idx] = shading
                else:
                    cloud_factor[t, pv_idx] = 0.0

        pv_profiles = base_irradiance[:, np.newaxis] * cloud_factor
        return np.clip(pv_profiles, 0.0, 1.0)
