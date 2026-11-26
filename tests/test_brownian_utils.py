"""Unit tests for src.brownian_utils and data.generate_synthetic."""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import brownian_increments, simulate_brownian_paths
from src.brownian_utils import regress_conditional_expectation


def test_simulate_brownian_paths_starts_at_zero_and_has_right_shape():
    grid = np.array([0.0, 0.5, 1.0])
    paths = simulate_brownian_paths(grid, n_paths=2000, dim=2, seed=0)
    assert paths.shape == (2000, 3, 2)
    assert np.allclose(paths[:, 0, :], 0.0)


def test_simulate_brownian_paths_has_correct_variance():
    grid = np.array([0.0, 1.0, 2.0])
    paths = simulate_brownian_paths(grid, n_paths=200_000, dim=1, seed=1)
    var_t1 = paths[:, 1, 0].var()
    var_t2 = paths[:, 2, 0].var()
    assert abs(var_t1 - 1.0) < 0.05
    assert abs(var_t2 - 2.0) < 0.07


def test_simulate_brownian_paths_rejects_bad_grid():
    with pytest.raises(ValueError):
        simulate_brownian_paths(np.array([1.0, 0.5]), n_paths=10)
    with pytest.raises(ValueError):
        simulate_brownian_paths(np.array([0.0]), n_paths=0)


def test_brownian_increments_sum_to_total_displacement():
    grid = np.linspace(0.0, 1.0, 11)
    paths = simulate_brownian_paths(grid, n_paths=500, dim=1, seed=2)
    dw = brownian_increments(grid, paths)
    assert dw.shape == (500, 10, 1)
    assert np.allclose(np.sum(dw, axis=1)[:, 0], paths[:, -1, 0] - paths[:, 0, 0])


def test_regress_conditional_expectation_trivial_sigma_field_is_sample_mean():
    rng = np.random.default_rng(3)
    target = rng.standard_normal(10_000) + 2.0
    fitted = regress_conditional_expectation(np.zeros((10_000, 0)), target, degree=3)
    assert np.allclose(fitted, target.mean())


def test_regress_conditional_expectation_recovers_known_conditional_mean():
    """E[W_1^2 | W_s] = (1 - s) + W_s^2 for standard Brownian motion (s<1)."""
    grid = np.array([0.5, 1.0])
    rng = np.random.default_rng(4)
    paths = simulate_brownian_paths(grid, n_paths=300_000, dim=1, seed=4)
    w_s, w_1 = paths[:, 0, 0], paths[:, 1, 0]
    fitted = regress_conditional_expectation(w_s, w_1**2, degree=2)
    expected = 0.5 + w_s**2
    assert np.mean(np.abs(fitted - expected)) < 0.05
