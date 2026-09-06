"""Detection delay calculation for simulated cyberattack events."""

from typing import Dict, List, Optional
import numpy as np


def compute_detection_delay(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    timesteps: np.ndarray,
    step_duration_minutes: float = 15.0,
) -> Dict[str, float]:
    """Compute mean and median detection delay across contiguous attack events.

    Args:
        y_true: [N_samples] binary ground truth labels.
        scores: [N_samples] continuous anomaly scores.
        threshold: Decision threshold.
        timesteps: [N_samples] integer timesteps.
        step_duration_minutes: Duration of single timestep (default: 15.0 min).

    Returns:
        Dictionary of mean_delay_steps, median_delay_steps, mean_delay_minutes, and detection_rate.
    """
    y_pred = (scores > threshold).astype(np.int64)

    # Identify contiguous attack episodes
    delays_steps = []
    detected_episodes = 0
    total_episodes = 0

    in_attack = False
    episode_start_idx = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_attack:
            in_attack = True
            episode_start_idx = i
            total_episodes += 1
        elif y_true[i] == 0 and in_attack:
            # Episode concluded
            in_attack = False
            episode_preds = y_pred[episode_start_idx:i]
            hit_indices = np.where(episode_preds == 1)[0]
            if len(hit_indices) > 0:
                first_hit_step = hit_indices[0]
                delays_steps.append(first_hit_step)
                detected_episodes += 1
            else:
                # Undetected during episode: penalty is full duration
                delays_steps.append(len(episode_preds))

    # Check if last episode reaches end
    if in_attack:
        total_episodes += 1
        episode_preds = y_pred[episode_start_idx:]
        hit_indices = np.where(episode_preds == 1)[0]
        if len(hit_indices) > 0:
            delays_steps.append(hit_indices[0])
            detected_episodes += 1
        else:
            delays_steps.append(len(episode_preds))

    if total_episodes == 0:
        return {
            "mean_delay_steps": 0.0,
            "median_delay_steps": 0.0,
            "mean_delay_minutes": 0.0,
            "detection_rate": 1.0,
        }

    mean_steps = float(np.mean(delays_steps))
    median_steps = float(np.median(delays_steps))

    return {
        "mean_delay_steps": mean_steps,
        "median_delay_steps": median_steps,
        "mean_delay_minutes": mean_steps * step_duration_minutes,
        "detection_rate": float(detected_episodes / total_episodes),
    }
