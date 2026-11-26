"""Unit tests for src.penalized_rates (paper Example 4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.penalized_rates import (
    admissibility_bound,
    conjugate_G,
    constant_rate_factorization_penalty,
    minimal_penalty,
    optimal_density_path,
    optimal_intermediate_factor,
    optimal_rate_path,
    restriction_factorization_penalty,
    verify_cocycle_4_2,
    verify_ic,
)


def test_conjugate_G_finite_on_corridor_and_infinite_outside():
    theta, upper_r = 1.0, 0.2
    assert conjugate_G(0.1, 0.0, theta, upper_r) == pytest.approx(0.1**2 / (2 * theta))
    assert np.isinf(conjugate_G(-0.01, 0.0, theta, upper_r))
    assert np.isinf(conjugate_G(0.3, 0.0, theta, upper_r))
    assert np.isinf(conjugate_G(0.1, 0.5, theta, upper_r))  # q != 0


def test_minimal_penalty_zero_at_z_equal_one():
    assert minimal_penalty(1.0, theta=1.0, delta_t=1.0) == 0.0


def test_minimal_penalty_positive_and_increasing_as_z_decreases():
    theta, dt = 1.0, 1.0
    c_high = minimal_penalty(0.99, theta, dt)
    c_low = minimal_penalty(0.5, theta, dt)
    assert 0 < c_high < c_low


def test_minimal_penalty_rejects_invalid_z():
    with pytest.raises(ValueError):
        minimal_penalty(0.0, theta=1.0, delta_t=1.0)
    with pytest.raises(ValueError):
        minimal_penalty(1.5, theta=1.0, delta_t=1.0)


def test_optimal_density_path_matches_boundary_conditions():
    z, s, t = 0.9, 0.0, 1.0
    v = np.array([s, t])
    lam = optimal_density_path(z, s, t, v)
    assert lam[0] == pytest.approx(1.0)
    assert lam[1] == pytest.approx(z)


def test_optimal_rate_path_is_nondecreasing():
    z, s, t = 0.9, 0.0, 1.0
    v = np.linspace(s, t, 50)
    beta = optimal_rate_path(z, s, t, v)
    assert np.all(np.diff(beta) >= -1e-12)


def test_admissibility_bound_matches_formula():
    upper_r, dt = 0.1, 1.0
    assert admissibility_bound(upper_r, dt) == pytest.approx((1 + upper_r * dt / 2) ** (-2))


def test_ic_property_holds_at_optimal_factorization():
    z, theta, s, u, t = 0.93, 1.0, 0.0, 0.5, 1.0
    out = verify_ic(z, theta, s, u, t)
    assert out["abs_gap"] < 1e-10


def test_restriction_and_constant_rate_factorizations_are_suboptimal():
    z, theta, s, u, t = 0.93, 1.0, 0.0, 0.5, 1.0
    c_st = minimal_penalty(z, theta, t - s)
    gamma_restriction = restriction_factorization_penalty(z, theta, s, u, t)
    gamma_constant = constant_rate_factorization_penalty(z, theta, s, u, t)
    assert gamma_restriction > c_st
    assert gamma_constant > c_st


def test_ordering_z_lt_constant_lt_optimal_lt_one():
    z, s, u, t = 0.93, 0.0, 0.5, 1.0
    z1_opt = optimal_intermediate_factor(z, s, u, t)
    z1_const = z ** ((u - s) / (t - s))
    assert z < z1_const < z1_opt < 1.0


def test_scenario_cocycle_property_4_2_for_arbitrary_rate_path():
    times = np.linspace(0.0, 1.0, 21)
    rng = np.random.default_rng(5)
    beta_path = rng.uniform(0.0, 0.1, size=20)
    out = verify_cocycle_4_2(beta_path, times, theta=1.0, s_idx=0, u_idx=10, t_idx=20)
    assert out["abs_gap"] < 1e-10
