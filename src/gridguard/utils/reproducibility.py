"""Reproducibility utilities for seeds, environment tracking, and hashes."""

import hashlib
import os
import platform
import random
import subprocess
import sys
from typing import Any, Dict, Optional
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_commit() -> Optional[str]:
    """Retrieve current git commit hash if in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return None


def calculate_file_hash(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for data lineage and provenance.

    Args:
        filepath: Path to file.
        chunk_size: Size of byte chunks to read.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_environment_info(seed: Optional[int] = None) -> Dict[str, Any]:
    """Capture complete execution environment for scientific auditing.

    Args:
        seed: Optional experiment random seed.

    Returns:
        Dictionary detailing OS, Python version, hardware, package versions, and git state.
    """
    import importlib.metadata

    packages = [
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "torch",
        "networkx",
        "OpenDSSDirect.py",
        "pyyaml",
        "matplotlib",
        "pytest",
    ]

    pkg_versions = {}
    for pkg in packages:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pkg_versions[pkg] = "unknown"

    info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "os_platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": get_git_commit(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "package_versions": pkg_versions,
    }
    if seed is not None:
        info["random_seed"] = seed

    return info
