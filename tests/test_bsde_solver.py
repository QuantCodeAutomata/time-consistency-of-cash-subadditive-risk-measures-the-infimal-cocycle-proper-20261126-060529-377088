"""Unit tests for src.bsde_solver (paper Section 4, Eq. (4.1)-(4.2), Prop. 4.1)."""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import brownian_increments, simulate_brownian_paths
from src.bsde_solver import (
    density_path_adapted,
    feedback_optimal_scenario,
    solve_bsde_corridor,
    verify_dual_representation,
    verify_time_consistency_bsde,
)


def _simulate(n_steps: int, n_paths: int, seed: int, t_end: float = 1.0):
    grid = np.linspace(0.0, t_end, n_steps + 1)
    paths = simulate_brownian_paths(grid, n_paths, dim=1, seed=seed)
    dw = brownian_increments(grid, paths)
    return grid, paths, dw


def test_solve_bsde_corridor_terminal_condition_matches_normalization():
    grid, paths, dw = _simulate(n_steps=10, n_paths=2_000, seed=0)
    payoff = paths[:, -1, 0]
    y, v = solve_bsde_corridor(payoff, grid, paths, dw, r=0.1, upper_r=0.3, nu=0.3, degree=3)
    assert np.allclose(y[:, -1], -payoff)
    assert y.shape == (2_000, 11)
    assert v.shape == (2_000, 10)


def test_solve_bsde_corridor_constant_payoff_gives_deterministic_value():
    grid, paths, dw = _simulate(n_steps=10, n_paths=2_000, seed=1)
    payoff = np.full(2_000, 2.0)
    y, v = solve_bsde_corridor(payoff, grid, paths, dw, r=0.1, upper_r=0.3, nu=0.3, degree=3)
    # y is deterministic along v; cross-path std should be tiny (regression noise only)
    assert np.std(y[:, 0]) < 0.05


def test_feedback_optimal_scenario_within_corridor_and_ball():
    r, upper_r, nu = 0.1, 0.3, 0.3
    y_values = np.array([-1.0, 0.0, 2.0])
    v_values = np.array([0.5, -0.2, 0.0])
    beta, q = feedback_optimal_scenario(y_values, v_values, r, upper_r, nu)
    assert np.all((beta == r) | (beta == upper_r))
    assert np.all(np.abs(q) <= nu + 1e-12)


def test_density_path_adapted_starts_at_one():
    beta = np.full((5, 4), 0.1)
    q = np.zeros((5, 4))
    dw = np.zeros((5, 4, 1))
    dt = np.full(4, 0.25)
    lam = density_path_adapted(beta, q, dw, dt)
    assert lam.shape == (5, 5)
    assert np.allclose(lam[:, 0], 1.0)


def test_dual_representation_matches_bsde_value_within_tolerance():
    grid, paths, dw = _simulate(n_steps=20, n_paths=20_000, seed=2)
    payoff = np.tanh(paths[:, -1, 0])
    out = verify_dual_representation(payoff, grid, paths, dw, r=0.1, upper_r=0.3, nu=0.3, degree=3)
    assert out["abs_diff"] < 0.05


def test_time_consistency_holds_for_bsde_risk_measure():
    grid, paths, dw = _simulate(n_steps=20, n_paths=10_000, seed=3)
    payoff = np.tanh(paths[:, -1, 0])
    out = verify_time_consistency_bsde(
        payoff, grid, paths, dw, r=0.1, upper_r=0.3, nu=0.3, u_index=10, degree=3
    )
    assert out["abs_diff"] < 0.05
