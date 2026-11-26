"""Unit tests for src.ambiguous_rates (paper Example 4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.ambiguous_rates import (
    conjugate_G,
    density_path,
    driver_g,
    restriction_vs_products_check,
    sample_admissible_scenario,
    worst_case_rho,
)


def test_conjugate_G_zero_on_corridor_infinite_outside():
    r, upper_r, nu = 0.1, 0.3, 0.3
    assert conjugate_G(0.2, 0.1, r, upper_r, nu) == 0.0
    assert np.isinf(conjugate_G(0.05, 0.0, r, upper_r, nu))  # beta < r
    assert np.isinf(conjugate_G(0.4, 0.0, r, upper_r, nu))  # beta > R
    assert np.isinf(conjugate_G(0.2, 0.5, r, upper_r, nu))  # |q| > nu


def test_driver_g_matches_conjugate_relationship_at_sample_points():
    r, upper_r, nu = 0.1, 0.3, 0.3
    y = np.array([-1.0, 0.0, 2.0])
    z = np.array([0.5, -0.2, 0.0])
    g = driver_g(y, z, r, upper_r, nu)
    # g(v,y,z) = R*y^- - r*y^+ + nu*|z|; check pointwise for y=-1,z=0.5
    assert g[0] == pytest.approx(upper_r * 1.0 - r * 0.0 + nu * 0.5)
    assert g[2] == pytest.approx(upper_r * 0.0 - r * 2.0 + nu * 0.0)


def test_density_path_starts_at_one_and_matches_closed_form():
    beta_path = np.array([0.1, 0.1])
    q_path = np.zeros((2, 1))
    dw = np.zeros((10, 2, 1))
    dt = np.array([0.5, 0.5])
    lam = density_path(beta_path, q_path, dw, dt)
    assert np.allclose(lam[:, 0], 1.0)
    assert np.allclose(lam[:, -1], np.exp(-0.1 * 1.0))


def test_worst_case_rho_selects_the_maximizing_scenario():
    rng = np.random.default_rng(0)
    x_values = rng.standard_normal(5_000)
    densities = np.array([
        np.full(5_000, 0.9),
        np.full(5_000, 1.1),
    ])
    values, idx = worst_case_rho(x_values, densities, conditioning=None)
    expected = [np.mean(-0.9 * x_values), np.mean(-1.1 * x_values)]
    assert idx == int(np.argmax(expected))


def test_worst_case_rho_is_pointwise_not_a_single_global_scenario():
    """ess max_Z E[-ZX|F_u] must pick, at each value of the conditioning
    variable, whichever scenario is best there -- not one scenario that is
    merely best on average (paper Example 4.4, dual formula (4.1))."""
    rng = np.random.default_rng(0)
    n = 20_000
    # Binary conditioning variable: a degree-1 polynomial fits two distinct
    # values exactly, so the regression step introduces no bias here and the
    # test isolates the pointwise-vs-global selection logic.
    sign = rng.choice([-1.0, 1.0], size=n)
    x_values = np.ones(n)  # X = 1 (deterministic), so -Z*X = -Z

    # Scenario A: Z = 1 - 0.3*sign(w_u) -> -Z is large (good) only when sign = +1.
    # Scenario B: Z = 1 + 0.3*sign(w_u) -> -Z is large (good) only when sign = -1.
    densities = np.array([1.0 - 0.3 * sign, 1.0 + 0.3 * sign])

    values, _ = worst_case_rho(x_values, densities, conditioning=sign, degree=1)
    # The true pointwise ess-max of {-Z_A, -Z_B} is -min(Z_A, Z_B) = -0.7 everywhere.
    assert np.allclose(values, -0.7, atol=0.05)
    # A single globally-best scenario could reach at best -1.0 (mean of -Z_A or -Z_B
    # is -1.0 for both, since E[sign(w_u)] = 0), far from the true pointwise optimum.
    single_scenario_curve = np.mean(-densities, axis=1)
    assert np.max(single_scenario_curve) < np.mean(values) - 0.1


def test_restriction_vs_products_check_runs_and_cocycle_identity_holds():
    rng = np.random.default_rng(1)
    out = restriction_vs_products_check(
        rng, r=0.1, upper_r=0.3, nu=0.3, dim=1, s=0.0, u=0.4, t=1.0,
        n_steps=20, n_paths=5_000, seed=1,
    )
    assert out["cocycle_identity_max_abs_gap"] < 1e-8


def test_restriction_suboptimal_when_r_positive():
    rng = np.random.default_rng(2)
    out = restriction_vs_products_check(
        rng, r=0.1, upper_r=0.3, nu=0.3, dim=1, s=0.0, u=0.4, t=1.0,
        n_steps=20, n_paths=20_000, seed=2,
    )
    assert out["restriction_conditional_expectation_mean"] > out["theoretical_bound_e_minus_r_t_minus_u"]


def test_restriction_admissible_when_r_is_zero():
    rng = np.random.default_rng(3)
    out = restriction_vs_products_check(
        rng, r=0.0, upper_r=0.3, nu=0.3, dim=1, s=0.0, u=0.4, t=1.0,
        n_steps=20, n_paths=20_000, seed=3,
    )
    assert out["theoretical_bound_e_minus_r_t_minus_u"] == pytest.approx(1.0)
    assert out["restriction_conditional_expectation_mean"] <= out["theoretical_bound_e_minus_r_t_minus_u"] + 1e-2


def test_sample_admissible_scenario_respects_corridor_and_ball():
    rng = np.random.default_rng(4)
    r, upper_r, nu = 0.1, 0.3, 0.3
    beta, q = sample_admissible_scenario(rng, n_steps=100, dim=2, r=r, upper_r=upper_r, nu=nu)
    assert np.all(beta >= r - 1e-12) and np.all(beta <= upper_r + 1e-12)
    assert np.all(np.linalg.norm(q, axis=1) <= nu + 1e-12)
