"""Discounted-expectation risk measure (paper Remark 3.9).

Custom implementation (paper Remark 3.9): rho_{s,t}(X) := -e^{-r(t-s)}
E[X | F_s] is linear, local, convex, cash-subadditive, normalized and
time-consistent, but not cash-additive for r > 0. Its minimal penalty is
0 at Z = e^{-r(t-s)} and +infinity everywhere else on Z_{s,t}, because the
risk measure is linear: for any Z,

    c_{s,t}(Z) = ess sup_X { E[-Z X | F_s] - rho_{s,t}(X) }
               = ess sup_X { E[-Z X | F_s] + e^{-r(t-s)} E[X | F_s] }
               = ess sup_X  E[ (e^{-r(t-s)} - Z) X | F_s ],

which is 0 when Z = e^{-r(t-s)} (the bracket vanishes for every X) and
+infinity otherwise (choose X = M * sign(e^{-r(t-s)} - Z) and let M -> infinity).
"""

from __future__ import annotations

import numpy as np

from src.brownian_utils import regress_conditional_expectation


def discount_factor(r: float, delta: float) -> float:
    """Deterministic discount e^{-r * delta} over a horizon of length delta."""
    if delta < 0:
        raise ValueError("delta (time horizon) must be non-negative")
    return float(np.exp(-r * delta))


def rho(
    x_values: np.ndarray,
    r: float,
    s: float,
    t: float,
    conditioning: np.ndarray | None = None,
    degree: int = 4,
) -> np.ndarray:
    """Monte Carlo estimate of rho_{s,t}(X) = -e^{-r(t-s)} E[X | F_s].

    Parameters
    ----------
    x_values:
        Array of shape ``(n_paths,)`` with the simulated position ``X``
        (F_t-measurable).
    r, s, t:
        Discount rate and evaluation/terminal times, s <= t.
    conditioning:
        F_s-measurable conditioning variables (e.g. simulated ``W_s``); use
        ``None`` or a constant array when F_s is trivial (s = 0).
    degree:
        Regression degree passed to :func:`regress_conditional_expectation`.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths,)`` with the (path-wise) value of
        rho_{s,t}(X); constant across paths if F_s is trivial.
    """
    if s > t:
        raise ValueError("require s <= t")
    if conditioning is None:
        conditional_expectation = np.full_like(x_values, x_values.mean(), dtype=float)
    else:
        conditional_expectation = regress_conditional_expectation(conditioning, x_values, degree)
    return -discount_factor(r, t - s) * conditional_expectation


def minimal_penalty(z: float, r: float, s: float, t: float, tol: float = 1e-8) -> float:
    """Minimal penalty c_{s,t}(Z) for a deterministic Z (Remark 3.9).

    Returns 0.0 if Z equals the deterministic discount e^{-r(t-s)} within
    ``tol``, and ``np.inf`` otherwise.
    """
    target = discount_factor(r, t - s)
    return 0.0 if abs(z - target) <= tol else np.inf


def restriction_factorization_penalty(r: float, s: float, u: float, t: float) -> float:
    """Gamma for the restriction factorization (Z_u, Z (x) Z_u) of Z=e^{-r(t-s)}.

    Restriction allocates the whole discount to [s, u]: Z^1 = e^{-r(t-s)},
    Z^2 = 1. Since c_{u,t}(1) = 0 only if 1 = e^{-r(t-u)}, i.e. r = 0 or
    u = t, otherwise c_{u,t}(1) = +infinity by :func:`minimal_penalty`, and
    Gamma = c_{s,u}(e^{-r(t-s)}) + e^{-r(t-s)} * c_{u,t}(1).
    """
    if not (s <= u <= t):
        raise ValueError("require s <= u <= t")
    z = discount_factor(r, t - s)
    c_su = minimal_penalty(z, r, s, u)  # Z^1 = z does not equal e^{-r(u-s)} unless r=0 or u=t
    c_ut = minimal_penalty(1.0, r, u, t)  # Z^2 = 1 does not equal e^{-r(t-u)} unless r=0 or u=t
    if np.isinf(c_su) or np.isinf(c_ut):
        return np.inf
    return c_su + z * c_ut


def proportional_factorization_penalty(r: float, s: float, u: float, t: float) -> float:
    """Gamma for the proportional factorization (e^{-r(u-s)}, e^{-r(t-u)})."""
    if not (s <= u <= t):
        raise ValueError("require s <= u <= t")
    z1 = discount_factor(r, u - s)
    z2 = discount_factor(r, t - u)
    c_su = minimal_penalty(z1, r, s, u)
    c_ut = minimal_penalty(z2, r, u, t)
    if np.isinf(c_su) or np.isinf(c_ut):
        return np.inf
    return c_su + z1 * c_ut


def verify_time_consistency(
    x_values: np.ndarray,
    r: float,
    s: float,
    u: float,
    t: float,
    w_s: np.ndarray | None,
    w_u: np.ndarray | None,
    degree: int = 4,
) -> dict:
    """Check rho_{s,t}(X) = rho_{s,u}(-rho_{u,t}(X)) (paper, Introduction).

    Returns a dict with both sides (as path-averages, since both are
    F_s-measurable and in this experiment F_s is trivial) and their
    absolute/relative difference.
    """
    direct = rho(x_values, r, s, t, w_s, degree)
    inner = rho(x_values, r, u, t, w_u, degree)
    recursive = rho(-inner, r, s, u, w_s, degree)
    direct_mean = float(np.mean(direct))
    recursive_mean = float(np.mean(recursive))
    diff = abs(direct_mean - recursive_mean)
    return {
        "rho_s_t": direct_mean,
        "rho_s_u_of_minus_rho_u_t": recursive_mean,
        "abs_diff": diff,
        "rel_diff": diff / max(abs(direct_mean), 1e-12),
    }
