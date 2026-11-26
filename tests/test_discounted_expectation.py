"""Unit tests for src.discounted_expectation (paper Remark 3.9)."""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import simulate_brownian_paths
from src.discounted_expectation import (
    discount_factor,
    minimal_penalty,
    proportional_factorization_penalty,
    restriction_factorization_penalty,
    rho,
    verify_time_consistency,
)


def test_discount_factor_decreasing_in_delta():
    assert discount_factor(0.05, 0.0) == 1.0
    assert discount_factor(0.05, 1.0) < discount_factor(0.05, 0.5) < 1.0


def test_discount_factor_rejects_negative_delta():
    with pytest.raises(ValueError):
        discount_factor(0.05, -1.0)


def test_rho_trivial_sigma_field_equals_minus_discounted_mean():
    rng = np.random.default_rng(0)
    x_values = rng.standard_normal(50_000) + 1.0
    values = rho(x_values, r=0.05, s=0.0, t=1.0, conditioning=None)
    expected = -discount_factor(0.05, 1.0) * x_values.mean()
    assert np.allclose(values, expected)


def test_rho_is_linear_in_x():
    rng = np.random.default_rng(1)
    x1 = rng.standard_normal(20_000)
    x2 = rng.standard_normal(20_000)
    r1 = rho(x1, r=0.03, s=0.0, t=0.5, conditioning=None)
    r2 = rho(x2, r=0.03, s=0.0, t=0.5, conditioning=None)
    r_sum = rho(x1 + x2, r=0.03, s=0.0, t=0.5, conditioning=None)
    assert np.allclose(r_sum, r1 + r2, atol=1e-10)


def test_minimal_penalty_zero_only_at_discount_factor():
    r, s, t = 0.05, 0.0, 1.0
    z_star = discount_factor(r, t - s)
    assert minimal_penalty(z_star, r, s, t) == 0.0
    assert np.isinf(minimal_penalty(z_star + 0.1, r, s, t))
    assert np.isinf(minimal_penalty(1.0, r, s, t))  # Z=1 != e^{-r(t-s)} when r>0


def test_restriction_factorization_is_infinite_when_r_positive_and_u_before_t():
    gamma = restriction_factorization_penalty(r=0.05, s=0.0, u=0.6, t=1.0)
    assert np.isinf(gamma)


def test_restriction_factorization_is_finite_when_r_is_zero():
    gamma = restriction_factorization_penalty(r=0.0, s=0.0, u=0.6, t=1.0)
    assert gamma == 0.0


def test_proportional_factorization_attains_the_minimal_penalty():
    r, s, u, t = 0.05, 0.0, 0.6, 1.0
    gamma = proportional_factorization_penalty(r, s, u, t)
    c_st = minimal_penalty(discount_factor(r, t - s), r, s, t)
    assert gamma == pytest.approx(c_st)
    assert gamma == 0.0


def test_restriction_strictly_worse_than_proportional_when_r_positive():
    r, s, u, t = 0.05, 0.0, 0.6, 1.0
    gamma_restriction = restriction_factorization_penalty(r, s, u, t)
    gamma_proportional = proportional_factorization_penalty(r, s, u, t)
    assert gamma_restriction > gamma_proportional


def test_time_consistency_holds_numerically_for_bounded_position():
    r, s, u, t = 0.05, 0.0, 0.6, 1.0
    grid = np.array([s, u, t])
    paths = simulate_brownian_paths(grid, n_paths=100_000, dim=1, seed=7)
    x_values = np.tanh(paths[:, -1, 0])
    out = verify_time_consistency(x_values, r, s, u, t, w_s=None, w_u=paths[:, 1, :])
    assert out["abs_diff"] < 0.01
