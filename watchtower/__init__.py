"""
watchtower package public API.

This module exposes the commonly used functions and classes from the project
via lazy imports so that importing `watchtower` is lightweight and does not
require all optional heavy dependencies to be present.

Usage examples:
- from watchtower import get_delta_func, DeltaCalculator
- from watchtower import cartesian_to_spherical, spherical_to_cartesian
- from watchtower import CMFEnv, generate_trajs
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

# Keep this in sync with your project version
__version__ = "0.1.0"

# Public API surface
__all__ = [
    # utils.py
    "DeltaCalculator",
    "get_delta_func",
    "random_step",
    "next_steps",
    "get_neighbors_fn",
    "quantize",
    "plot_search",
    # delta_estimator.py
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "deltaDataset",
    "find_best_traj",
    # dataset_generation.py
    "generate_trajs",
    # env.py
    "CMFEnv",
]

# Map public symbols to candidate import paths.
# We try namespaced modules first (watchtower.*), then top-level modules as fallback.
_IMPORT_MAP: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # utils.py
    "DeltaCalculator": (
        ("watchtower.utils", "DeltaCalculator"),
        ("utils", "DeltaCalculator"),
    ),
    "get_delta_func": (
        ("watchtower.utils", "get_delta_func"),
        ("utils", "get_delta_func"),
    ),
    "random_step": (
        ("watchtower.utils", "random_step"),
        ("utils", "random_step"),
    ),
    "next_steps": (
        ("watchtower.utils", "next_steps"),
        ("utils", "next_steps"),
    ),
    "get_neighbors_fn": (
        ("watchtower.utils", "get_neighbors_fn"),
        ("utils", "get_neighbors_fn"),
    ),
    "quantize": (
        ("watchtower.utils", "quantize"),
        ("utils", "quantize"),
    ),
    "plot_search": (
        ("watchtower.utils", "plot_search"),
        ("utils", "plot_search"),
    ),
    # delta_estimator.py
    "cartesian_to_spherical": (
        ("watchtower.delta_estimator", "cartesian_to_spherical"),
        ("delta_estimator", "cartesian_to_spherical"),
    ),
    "spherical_to_cartesian": (
        ("watchtower.delta_estimator", "spherical_to_cartesian"),
        ("delta_estimator", "spherical_to_cartesian"),
    ),
    "deltaDataset": (
        ("watchtower.delta_estimator", "deltaDataset"),
        ("delta_estimator", "deltaDataset"),
    ),
    "find_best_traj": (
        ("watchtower.delta_estimator", "find_best_traj"),
        ("delta_estimator", "find_best_traj"),
    ),
    # dataset_generation.py
    "generate_trajs": (
        ("watchtower.dataset_generation", "generate_trajs"),
        ("dataset_generation", "generate_trajs"),
    ),
    # env.py
    "CMFEnv": (
        ("watchtower.env", "CMFEnv"),
        ("env", "CMFEnv"),
    ),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import public API symbols on first access.

    This defers importing heavy dependencies until they are actually used and
    supports both namespaced (watchtower.*) and top-level module layouts.
    """
    candidates = _IMPORT_MAP.get(name)
    if not candidates:
        raise AttributeError(f"module 'watchtower' has no attribute '{name}'")

    last_error: Exception | None = None
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            try:
                value = getattr(module, attr_name)
                globals()[name] = value  # cache for future access
                return value
            except AttributeError as e:
                last_error = e
                continue
        except Exception as e:  # ImportError or any import-time error
            last_error = e
            continue

    # If we reach here, nothing worked
    hint = (
        "The required module could not be imported. Ensure that all optional "
        "dependencies for this symbol are installed and available on PYTHONPATH."
    )
    detail = f"Failed to resolve '{name}'. Last error: {last_error!r}"
    raise AttributeError(f"{detail} {hint}")


def __dir__() -> list[str]:
    """Improve auto-completion by listing public attributes."""
    return sorted(list(globals().keys()) + __all__)
