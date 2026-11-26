"""Example 4.5 -- penalized interest rates: explicit minimal penalty.

Custom implementation (paper Example 4.5, pp. 13-14). Driver

    g(v, y, z) = max_{0<=beta<=R} { -beta*y - beta^2/(2*theta) },

which is R-Lipschitz, does not depend on z, and has conjugate
G(v, beta, q) = beta^2/(2*theta) for 0<=beta<=R, q=0, and +infinity
otherwise. For a deterministic discount factor z in (0, 1], the minimal
penalty is obtained via a Cauchy-Schwarz argument on the path
Lambda_v = exp(-int_s^v beta_w dw):

    c_{s,t}(z) = 2 * (1 - sqrt(z))^2 / (theta * (t - s)),

attained uniquely at sqrt(Lambda_v^*) = 1 - a*(v - s), a = (1-sqrt(z))/(t-s),
with optimal rate beta_v^* = 2a / (1 - a*(v-s)), admissible
(0 <= beta_v^* <= R for all v) iff z >= (1 + R*(t-s)/2)^{-2}.
"""

from __future__ import annotations

import numpy as np


def driver_g(y: np.ndarray, theta: float, upper_r: float) -> np.ndarray:
    """g(v, y, 0..) = max_{0<=beta<=R} {-beta*y - beta^2/(2 theta)} (Example 4.5)."""
    beta_star = np.clip(-theta * y, 0.0, upper_r)
    return beta_star * y + beta_star**2 / (2.0 * theta)


def conjugate_G(beta: float, q: float, theta: float, upper_r: float) -> float:
    """G(v, beta, q) = beta^2/(2 theta) for 0<=beta<=R, q=0, else +infinity."""
    if 0.0 - 1e-12 <= beta <= upper_r + 1e-12 and abs(q) <= 1e-12:
        return beta**2 / (2.0 * theta)
    return np.inf


def admissibility_bound(upper_r: float, delta_t: float) -> float:
    """Lower bound on z below which the optimal rate would exceed R: z >= (1+R*dt/2)^{-2}."""
    return float((1.0 + upper_r * delta_t / 2.0) ** (-2))


def minimal_penalty(z: float, theta: float, delta_t: float) -> float:
    """c_{s,t}(z) = 2*(1-sqrt(z))^2 / (theta * (t-s)) (Example 4.5)."""
    if not (0.0 < z <= 1.0):
        raise ValueError("z must lie in (0, 1]")
    if delta_t <= 0:
        raise ValueError("delta_t = t - s must be positive")
    return 2.0 * (1.0 - np.sqrt(z)) ** 2 / (theta * delta_t)


def optimal_density_path(z: float, s: float, t: float, v: np.ndarray) -> np.ndarray:
    """Lambda_v^* = (1 - a*(v - s))^2, a = (1 - sqrt(z)) / (t - s)."""
    a = (1.0 - np.sqrt(z)) / (t - s)
    return (1.0 - a * (np.asarray(v) - s)) ** 2


def optimal_rate_path(z: float, s: float, t: float, v: np.ndarray) -> np.ndarray:
    """beta_v^* = 2a / (1 - a*(v - s)), a = (1 - sqrt(z)) / (t - s)."""
    a = (1.0 - np.sqrt(z)) / (t - s)
    return 2.0 * a / (1.0 - a * (np.asarray(v) - s))


def optimal_intermediate_factor(z: float, s: float, u: float, t: float) -> float:
    """Z^1 = Lambda_u^* = ((t-u)/(t-s) + (u-s)/(t-s) * sqrt(z))^2."""
    return float(((t - u) / (t - s) + (u - s) / (t - s) * np.sqrt(z)) ** 2)


def verify_ic(z: float, theta: float, s: float, u: float, t: float) -> dict:
    """Verify (IC): c_{s,t}(z) = c_{s,u}(Z^1) + Z^1 * c_{u,t}(z / Z^1) at the optimum."""
    z1 = optimal_intermediate_factor(z, s, u, t)
    z2 = z / z1
    c_su = minimal_penalty(z1, theta, u - s)
    c_ut = minimal_penalty(z2, theta, t - u)
    total = c_su + z1 * c_ut
    c_st = minimal_penalty(z, theta, t - s)
    return {
        "z1_optimal_intermediate": z1,
        "z2": z2,
        "c_su": c_su,
        "c_ut": c_ut,
        "c_su_plus_z1_c_ut": total,
        "c_st": c_st,
        "abs_gap": abs(total - c_st),
    }


def restriction_factorization_penalty(z: float, theta: float, s: float, u: float, t: float) -> float:
    """Gamma for the restriction factorization Z^1=z, Z^2=1: c_{s,u}(z) + z*c_{u,t}(1).

    c_{u,t}(1) = 0 since 1 is the trivial (undiscounted) density, attained
    at beta = 0 identically, which is admissible and has zero running cost.
    """
    c_su = minimal_penalty(z, theta, u - s)
    c_ut = 0.0  # beta=0 on [u,t] generates Lambda=1 at zero cost
    return c_su + z * c_ut


def constant_rate_factorization_penalty(z: float, theta: float, s: float, u: float, t: float) -> float:
    """Gamma for the constant-rate factorization Z^1=z^{(u-s)/(t-s)}, Z^2=z^{(t-u)/(t-s)}."""
    z1 = z ** ((u - s) / (t - s))
    z2 = z ** ((t - u) / (t - s))
    c_su = minimal_penalty(z1, theta, u - s)
    c_ut = minimal_penalty(z2, theta, t - u)
    return c_su + z1 * c_ut


def scenario_penalty_path(beta_path: np.ndarray, times: np.ndarray, theta: float) -> np.ndarray:
    """Cumulative scenario penalty c_{s,v}^{beta,0} = int_s^v Lambda_w G(beta_w) dw.

    Deterministic (q=0), used to check the scenario cocycle property (4.2)
    for an arbitrary (not necessarily optimal) admissible rate path beta,
    with running cost G(beta) = beta^2 / (2 theta) (Example 4.5, Remark 4.2).
    """
    dt = np.diff(times)
    log_lambda = np.concatenate([[0.0], -np.cumsum(beta_path * dt)])
    lam = np.exp(log_lambda)
    running_cost = lam[:-1] * (beta_path**2) / (2.0 * theta)
    increments = running_cost * dt
    return np.concatenate([[0.0], np.cumsum(increments)])


def verify_cocycle_4_2(
    beta_path: np.ndarray, times: np.ndarray, theta: float, s_idx: int, u_idx: int, t_idx: int
) -> dict:
    """Verify c_{s,t}^{beta} = c_{s,u}^{beta} + Lambda_{s,u}^{beta} * c_{u,t}^{beta} (Eq. 4.2)."""
    c_cumulative = scenario_penalty_path(beta_path, times, theta)
    dt = np.diff(times)
    lam_s_u = float(np.exp(-np.sum(beta_path[s_idx:u_idx] * dt[s_idx:u_idx])))

    c_st = c_cumulative[t_idx] - c_cumulative[s_idx]
    c_su = c_cumulative[u_idx] - c_cumulative[s_idx]
    c_ut_from_s = c_cumulative[t_idx] - c_cumulative[u_idx]
    c_ut_alone = scenario_penalty_path(beta_path[u_idx:t_idx], times[u_idx : t_idx + 1], theta)[-1]
    rhs = c_su + lam_s_u * c_ut_alone
    return {
        "c_s_t": c_st,
        "c_s_u": c_su,
        "lambda_s_u": lam_s_u,
        "c_u_t": c_ut_alone,
        "c_u_t_consistency_check": abs(c_ut_from_s / max(lam_s_u, 1e-12) - c_ut_alone),
        "rhs_c_su_plus_lambda_c_ut": rhs,
        "abs_gap": abs(c_st - rhs),
    }
