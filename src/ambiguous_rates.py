"""Example 4.4 -- ambiguous short rate in a corridor [r, R], model ambiguity nu.

Custom implementation (paper Example 4.4, p. 13). Driver

    g(v, y, z) = R * y^- - r * y^+ + nu * |z|,   y^- = max(-y, 0), y^+ = max(y, 0),

is mu-Lipschitz with mu = max(R, nu). Its convex conjugate in (y, z) is

    G(v, beta, q) = 0                 if r <= beta <= R and |q| <= nu,
    G(v, beta, q) = +infinity         otherwise,

so that the induced risk measure is the worst-case discounted expectation

    rho_{s,t}(X) = ess max_{Z in D_{s,t}} E[-Z X | F_s],
    D_{s,t} = { Lambda_{s,t}^{beta,q} : (beta, q) predictable, r<=beta<=R, |q|<=nu },

with Lambda_{s,t}^{beta,q} = exp(-int beta dv + int q.dW - 1/2 int |q|^2 dv).
The minimal penalty is c_{s,t}(Z) = 0 for Z in D_{s,t} and +infinity
otherwise, and (IC) reduces to the stability of D under products,
D_{s,t} = {Z^1 Z^2 : Z^1 in D_{s,u}, Z^2 in D_{u,t}}.
"""

from __future__ import annotations

import numpy as np

from src.brownian_utils import regress_conditional_expectation


def driver_g(y: np.ndarray, z: np.ndarray, r: float, upper_r: float, nu: float) -> np.ndarray:
    """g(v, y, z) = R * y^- - r * y^+ + nu * |z| (Example 4.4)."""
    y_minus = np.maximum(-y, 0.0)
    y_plus = np.maximum(y, 0.0)
    z_norm = np.abs(z) if np.ndim(z) <= 1 else np.linalg.norm(z, axis=-1)
    return upper_r * y_minus - r * y_plus + nu * z_norm


def conjugate_G(beta: float, q: np.ndarray, r: float, upper_r: float, nu: float) -> float:
    """G(v, beta, q) = 0 if r<=beta<=R and |q|<=nu, else +infinity (Example 4.4)."""
    q_norm = np.abs(q) if np.ndim(q) == 0 else np.linalg.norm(np.atleast_1d(q))
    admissible = (r - 1e-12 <= beta <= upper_r + 1e-12) and (q_norm <= nu + 1e-12)
    return 0.0 if admissible else np.inf


def sample_admissible_scenario(
    rng: np.random.Generator,
    n_steps: int,
    dim: int,
    r: float,
    upper_r: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one predictable, piecewise-constant admissible scenario (beta, q).

    Returns
    -------
    beta_path: array of shape (n_steps,), values in [r, upper_r].
    q_path: array of shape (n_steps, dim), each row with Euclidean norm <= nu.
    """
    beta_path = rng.uniform(r, upper_r, size=n_steps)
    directions = rng.standard_normal((n_steps, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    radii = rng.uniform(0.0, nu, size=(n_steps, 1))
    q_path = directions / norms * radii
    return beta_path, q_path


def density_path(
    beta_path: np.ndarray,
    q_path: np.ndarray,
    dw: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Cumulative Lambda_{s,v}^{beta,q} at every grid point, for M Monte Carlo paths.

    Parameters
    ----------
    beta_path:
        Array of shape (n_steps,): scenario rate, shared across MC paths.
    q_path:
        Array of shape (n_steps, dim): scenario kernel, shared across MC paths.
    dw:
        Brownian increments, array of shape (n_paths, n_steps, dim).
    dt:
        Array of shape (n_steps,) with the time-step lengths.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1); column 0 is 1 (Lambda_s = 1).
    """
    drift = -beta_path * dt  # (n_steps,)
    martingale = np.einsum("mnd,nd->mn", dw, q_path)  # (n_paths, n_steps)
    quad_var = 0.5 * np.sum(q_path**2, axis=1) * dt  # (n_steps,)
    log_increments = drift[None, :] + martingale - quad_var[None, :]
    log_lambda = np.concatenate(
        [np.zeros((dw.shape[0], 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return np.exp(log_lambda)


def worst_case_rho(
    x_values: np.ndarray,
    terminal_densities: np.ndarray,
    conditioning: np.ndarray | None,
    degree: int = 4,
) -> tuple[np.ndarray, int]:
    """Monte Carlo estimate of rho_{s,t}(X) = ess max_Z E[-Z X | F_s] over a scenario grid.

    Parameters
    ----------
    x_values:
        Array (n_paths,), the position X.
    terminal_densities:
        Array (n_scenarios, n_paths) with Lambda_{s,t}^{beta,q} for each
        sampled scenario.
    conditioning:
        F_s-conditioning variables, or None if F_s is trivial.

    Returns
    -------
    (values, argmax_index): the path-wise ess-max value and the index of the
    scenario that is best on average (reported for diagnostics only).

    Notes
    -----
    ``ess max_Z E[-ZX|F_s]`` is a pointwise (F_s-measurable) supremum: when
    F_s is non-trivial, different realizations of the conditioning variable
    can be maximized by different scenarios, so the ess-max must be computed
    as the elementwise maximum, over scenarios, of each scenario's regressed
    conditional-expectation curve -- not as the curve of a single globally
    "best on average" scenario. When F_s is trivial (``conditioning is
    None``), ``regress_conditional_expectation`` already collapses every
    curve to its scalar sample mean, so the elementwise maximum coincides
    with the scalar argmax of the means.
    """
    conditioning_arg = (
        conditioning if conditioning is not None else np.zeros((len(x_values), 0))
    )
    all_curves = np.empty((terminal_densities.shape[0], len(x_values)))
    for k, z_terminal in enumerate(terminal_densities):
        all_curves[k] = regress_conditional_expectation(
            conditioning_arg, -z_terminal * x_values, degree
        )
    best = int(np.argmax(all_curves.mean(axis=1)))
    best_value = np.max(all_curves, axis=0)
    return best_value, best


def restriction_vs_products_check(
    rng: np.random.Generator,
    r: float,
    upper_r: float,
    nu: float,
    dim: int,
    s: float,
    u: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> dict:
    """Verify stability under products (IC) and suboptimality of the restriction split.

    Simulates one admissible scenario on [s, t], builds Z = Lambda_{s,t},
    checks the cocycle identity Lambda_{s,t} = Lambda_{s,u} * Lambda_{u,t},
    and compares the restriction factorization (Z_u, Z (x) Z_u) with the
    theoretical bound sup_{Z^2 in D_{u,t}} E[Z^2 | F_u] = e^{-r(t-u)}.
    """
    from data.generate_synthetic import simulate_brownian_paths, brownian_increments

    if not (s <= u <= t):
        raise ValueError("require s <= u <= t")
    grid = np.linspace(s, t, n_steps + 1)
    paths = simulate_brownian_paths(grid, n_paths, dim=dim, seed=seed)
    dw = brownian_increments(grid, paths)
    dt = np.diff(grid)
    u_index = int(np.argmin(np.abs(grid - u)))

    beta_path, q_path = sample_admissible_scenario(rng, n_steps, dim, r, upper_r, nu)
    lam = density_path(beta_path, q_path, dw, dt)  # (n_paths, n_steps+1)
    z_full = lam[:, -1]
    z_u_true = lam[:, u_index]  # Lambda_{s,u}^{beta,q}, exact cocycle factor (same scenario)

    # Cocycle identity Lambda_{s,t} = Lambda_{s,u} * Lambda_{u,t} for this scenario.
    lam_u_to_t = density_path(
        beta_path[u_index:], q_path[u_index:], dw[:, u_index:, :], dt[u_index:]
    )[:, -1]
    cocycle_gap = float(np.max(np.abs(z_full - z_u_true * lam_u_to_t)))

    # Restriction factorization: Z_u (regression estimate of E[Z|F_u]) and Z (x) Z_u.
    w_u = paths[:, u_index, :]
    z_u_hat = regress_conditional_expectation(w_u, z_full, degree=4)
    z_u_hat = np.maximum(z_u_hat, 1e-8)
    z_odot_zu = z_full / z_u_hat
    e_z_odot_zu_given_u = regress_conditional_expectation(w_u, z_odot_zu, degree=4)

    theoretical_bound = float(np.exp(-r * (t - u)))
    # Empirical check of the bound sup_{Z^2 in D_{u,t}} E[Z^2|F_u] <= e^{-r(t-u)}
    # over a further batch of independently sampled admissible scenarios on [u, t].
    sample_bounds = []
    for _ in range(50):
        b2, q2 = sample_admissible_scenario(rng, n_steps - u_index, dim, r, upper_r, nu)
        lam2 = density_path(b2, q2, dw[:, u_index:, :], dt[u_index:])[:, -1]
        sample_bounds.append(float(np.mean(lam2)))
    empirical_max_e_z2 = float(np.max(sample_bounds))

    return {
        "cocycle_identity_max_abs_gap": cocycle_gap,
        "restriction_conditional_expectation_mean": float(np.mean(e_z_odot_zu_given_u)),
        "theoretical_bound_e_minus_r_t_minus_u": theoretical_bound,
        "empirical_max_E_Z2_given_Fu_over_scenarios": empirical_max_e_z2,
        "restriction_is_admissible": bool(
            np.mean(e_z_odot_zu_given_u) <= theoretical_bound + 2e-3
        ),
        "r": r,
    }
