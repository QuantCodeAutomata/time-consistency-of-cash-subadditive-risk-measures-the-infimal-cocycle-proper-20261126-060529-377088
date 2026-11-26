"""Synthetic sample-path generation for the numerical experiments.

The paper works on an abstract filtered probability space carrying a
d-dimensional Brownian motion W (Section 2; Section 4, paragraph before
Eq. (4.1)). No market data enters any of the four experiments: every
quantity verified in this repository (the discounted-expectation risk
measure, the interest-rate-ambiguity driver of Example 4.4, the penalized
driver of Example 4.5, and the BSDE of Section 4) is a theoretical
construction on a Brownian filtration. This module is the single place
that draws Brownian sample paths; every ``src`` module consumes paths
produced here instead of drawing its own randomness.
"""

from __future__ import annotations

import numpy as np


def simulate_brownian_paths(
    time_grid: np.ndarray,
    n_paths: int,
    dim: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate exact Brownian motion values on an arbitrary time grid.

    Parameters
    ----------
    time_grid:
        Strictly increasing 1-D array of times, with ``time_grid[0] >= 0``.
        If ``time_grid[0] > 0``, ``W`` is still started from ``W_0 = 0`` and
        simulated jointly with the earlier, unreported times, so that the
        increments used internally are independent and the returned values
        at every requested time are exact draws from the Brownian law
        (no discretization bias, only Monte Carlo sampling error).
    n_paths:
        Number of independent sample paths ``M``.
    dim:
        Dimension ``d`` of the Brownian motion.
    seed:
        Seed of the random generator, for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, len(time_grid), dim)`` with
        ``W[:, k, :]`` distributed as ``W_{time_grid[k]}``.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    if time_grid.ndim != 1 or len(time_grid) == 0:
        raise ValueError("time_grid must be a non-empty 1-D array")
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing")
    if time_grid[0] < 0:
        raise ValueError("time_grid must start at or after 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    rng = np.random.default_rng(seed)
    full_grid = np.concatenate(([0.0], time_grid)) if time_grid[0] > 0 else time_grid
    dt = np.diff(full_grid)
    increments = rng.standard_normal((n_paths, len(dt), dim)) * np.sqrt(dt)[None, :, None]
    path = np.concatenate(
        [np.zeros((n_paths, 1, dim)), np.cumsum(increments, axis=1)], axis=1
    )
    # path[:, k, :] holds W_{full_grid[k]}; drop the k=0 (t=0) row if it was
    # only added as scaffolding and is not among the requested times.
    if time_grid[0] > 0:
        return path[:, 1:, :]
    return path


def brownian_increments(
    time_grid: np.ndarray,
    paths: np.ndarray,
) -> np.ndarray:
    """Increments ``W_{t_{k+1}} - W_{t_k}`` implied by simulated paths.

    Parameters
    ----------
    time_grid:
        The same time grid used to simulate ``paths``.
    paths:
        Array of shape ``(n_paths, len(time_grid), dim)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, len(time_grid) - 1, dim)``.
    """
    if paths.shape[1] != len(time_grid):
        raise ValueError("paths and time_grid have inconsistent lengths")
    return np.diff(paths, axis=1)
