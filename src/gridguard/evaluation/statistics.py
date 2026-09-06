"""Statistical rigor utilities: bootstrap confidence intervals and paired hypothesis testing."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate empirical bootstrap confidence interval for mean of metric.

    Args:
        data: 1D array of metric values (e.g. across runs/seeds).
        n_bootstraps: Number of resampling iterations (default: 1000).
        ci_level: Confidence level (default: 0.95).
        seed: Random seed.

    Returns:
        Tuple of (point_mean, ci_lower, ci_upper).
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_bootstraps)

    for b in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha_tail = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha_tail * 100.0))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha_tail) * 100.0))
    point_mean = float(np.mean(arr))

    return point_mean, ci_lower, ci_upper


def paired_wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Compute paired Wilcoxon signed-rank test between two matched models across runs.

    Args:
        scores_a: Metric array from model A.
        scores_b: Metric array from model B.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Tuple of (statistic, p_value).
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = a - b

    # If all differences are zero
    if np.all(diff == 0):
        return 0.0, 1.0

    try:
        res = stats.wilcoxon(a, b, alternative=alternative)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni step-down procedure for multiple hypothesis testing.

    Args:
        p_values: List of unadjusted p-values.
        alpha: Family-wise significance level (default: 0.05).

    Returns:
        List of tuples: (adjusted_p_value, is_significant_boolean) in original input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_results = [None] * m

    cum_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        k = rank + 1
        adj_p = min(1.0, (m - k + 1) * p_val)
        # Ensure monotonic non-decreasing adjusted p-values
        cum_max = max(cum_max, adj_p)
        is_sig = cum_max < alpha
        adjusted_results[orig_idx] = (cum_max, is_sig)

    return adjusted_results
