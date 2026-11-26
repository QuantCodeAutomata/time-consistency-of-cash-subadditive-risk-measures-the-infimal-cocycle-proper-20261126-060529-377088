"""Section 4 -- BSDE-induced risk measures: numerical solver and Proposition 4.1.

Custom implementation (paper Section 4, around Eq. (4.1)-(4.2) and
Proposition 4.1). No maintained Python package solves BSDEs with a general
convex driver (confirmed before implementation: no equivalent of the
regression-Monte-Carlo scheme below was found among the available
libraries), so the backward sweep is implemented from scratch, following
the explicit (Bouchard-Touzi / Longstaff-Schwartz style) scheme:

    Y_v = -X + int_v^t g(w, Y_w, V_w) dw - int_v^t V_w . dW_w,

discretized on a time grid t_0 < ... < t_N = t with

    V_{t_n} ~= E[Y_{t_{n+1}} * Delta W_n | F_{t_n}] / Delta t_n,
    Y_{t_n} ~= E[Y_{t_{n+1}} + g(t_n, Y_{t_{n+1}}, V_{t_n}) * Delta t_n | F_{t_n}],

with conditional expectations estimated by polynomial regression on
W_{t_n} (src.brownian_utils.regress_conditional_expectation).

The driver used is that of Example 4.4 (ambiguous short rate corridor),
g(v,y,z) = R*y^- - r*y^+ + nu*|z| (src.ambiguous_rates.driver_g), because
it is genuinely state- and z-dependent (unlike Example 4.5's driver, whose
optimal control is deterministic and already verified in closed form in
Example 4.5 / exp_3). For this driver, the pointwise maximizer of
g(y,z) = sup_{(beta,q)} {-beta*y + q.z - G(beta,q)} is attained at the
feedback rule beta*(y) = R if y<0 else r, q*(z) = nu*sign(z); this is the
scenario used to check the dual representation (4.1) and Proposition 4.1.
"""

from __future__ import annotations

import numpy as np

from src.ambiguous_rates import driver_g
from src.brownian_utils import regress_conditional_expectation


def solve_bsde_corridor(
    terminal_payoff: np.ndarray,
    time_grid: np.ndarray,
    paths: np.ndarray,
    dw: np.ndarray,
    r: float,
    upper_r: float,
    nu: float,
    degree: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward regression sweep for the BSDE with the Example 4.4 driver.

    Parameters
    ----------
    terminal_payoff:
        Array (n_paths,), the position X = payoff(W_t).
    time_grid:
        Array (N+1,) of times t_0=s, ..., t_N=t.
    paths:
        Array (n_paths, N+1, dim) with the simulated Brownian paths (dim=1).
    dw:
        Array (n_paths, N, dim), Brownian increments between grid points.
    r, upper_r, nu:
        Corridor driver parameters (Example 4.4).
    degree:
        Regression degree.

    Returns
    -------
    (Y, V): Y has shape (n_paths, N+1), V has shape (n_paths, N) (V_n is the
    control used on [t_n, t_{n+1}]).
    """
    n_paths, n_grid, dim = paths.shape
    n_steps = n_grid - 1
    dt = np.diff(time_grid)

    y = np.empty((n_paths, n_grid))
    v = np.empty((n_paths, n_steps))
    # Y_v = -X + int g dw - int V dW  =>  Y_t = -X (cash-subadditive
    # normalization rho_{t,t}(X) = -X, consistent with Remark 3.9/Section 2).
    y[:, -1] = -terminal_payoff

    for n in range(n_steps - 1, -1, -1):
        w_n = paths[:, n, :]
        y_next = y[:, n + 1]
        target_v = y_next * dw[:, n, 0]
        v_n = regress_conditional_expectation(w_n, target_v, degree) / dt[n]
        driver_value = driver_g(y_next, v_n, r, upper_r, nu)
        rhs = y_next + driver_value * dt[n]
        y[:, n] = regress_conditional_expectation(w_n, rhs, degree)
        v[:, n] = v_n

    return y, v


def feedback_optimal_scenario(
    y_values: np.ndarray, v_values: np.ndarray, r: float, upper_r: float, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise maximizer (beta*, q*) of g(y,z) = sup_{(beta,q)}{-beta*y+q*z-G}.

    beta*(y) = R if y < 0 else r; q*(z) = nu * sign(z) (Example 4.4, dim=1).
    """
    beta_star = np.where(y_values < 0.0, upper_r, r)
    q_star = nu * np.sign(v_values)
    return beta_star, q_star


def density_path_adapted(
    beta: np.ndarray, q: np.ndarray, dw: np.ndarray, dt: np.ndarray
) -> np.ndarray:
    """Cumulative Lambda path for a possibly path- and time-dependent scenario.

    Parameters
    ----------
    beta:
        Array (n_paths, n_steps).
    q:
        Array (n_paths, n_steps) (dim=1 kernel).
    dw:
        Array (n_paths, n_steps, 1).
    dt:
        Array (n_steps,).

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps + 1), column 0 equal to 1.
    """
    drift = -beta * dt[None, :]
    martingale = q * dw[:, :, 0]
    quad_var = 0.5 * q**2 * dt[None, :]
    log_increments = drift + martingale - quad_var
    log_lambda = np.concatenate(
        [np.zeros((beta.shape[0], 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return np.exp(log_lambda)


def verify_dual_representation(
    terminal_payoff: np.ndarray,
    time_grid: np.ndarray,
    paths: np.ndarray,
    dw: np.ndarray,
    r: float,
    upper_r: float,
    nu: float,
    degree: int = 5,
) -> dict:
    """Compare the BSDE value Y_0 with the dual value under the feedback-optimal scenario.

    This implements Proposition 4.1: the scenario (beta*, q*) built from the
    BSDE's own solution (Y, V) attains the ess max in the dual formula
    (4.1), E[-Lambda_{s,t}^{beta*,q*} X | F_s] = Y_s, since it is admissible
    (G = 0 along it) by construction.
    """
    y, v = solve_bsde_corridor(terminal_payoff, time_grid, paths, dw, r, upper_r, nu, degree)
    n_steps = len(time_grid) - 1
    dt = np.diff(time_grid)

    beta_star = np.empty((paths.shape[0], n_steps))
    q_star = np.empty((paths.shape[0], n_steps))
    for n in range(n_steps):
        b, q = feedback_optimal_scenario(y[:, n + 1], v[:, n], r, upper_r, nu)
        beta_star[:, n] = b
        q_star[:, n] = q

    lam = density_path_adapted(beta_star, q_star, dw, dt)
    z_terminal = lam[:, -1]
    dual_value = np.mean(-z_terminal * terminal_payoff)
    primal_value = float(np.mean(y[:, 0]))

    return {
        "bsde_Y0_mean": primal_value,
        "dual_scenario_value_mean": float(dual_value),
        "abs_diff": abs(primal_value - dual_value),
        "rel_diff": abs(primal_value - dual_value) / max(abs(primal_value), 1e-8),
    }


def verify_time_consistency_bsde(
    terminal_payoff: np.ndarray,
    time_grid: np.ndarray,
    paths: np.ndarray,
    dw: np.ndarray,
    r: float,
    upper_r: float,
    nu: float,
    u_index: int,
    degree: int = 5,
) -> dict:
    """Check rho_{s,t}(X) = rho_{s,u}(-rho_{u,t}(X)) for the BSDE risk measure.

    rho_{s,t}(X) is Y_0 of the full BSDE on [s,t]. rho_{u,t}(X) is Y at the
    intermediate index solved from the same terminal condition; rho_{s,u} of
    the (negated) intermediate value is then a second BSDE solve on [s,u].
    """
    y_full, _ = solve_bsde_corridor(terminal_payoff, time_grid, paths, dw, r, upper_r, nu, degree)
    direct = float(np.mean(y_full[:, 0]))
    rho_u_t = y_full[:, u_index]

    sub_grid = time_grid[: u_index + 1]
    sub_paths = paths[:, : u_index + 1, :]
    sub_dw = dw[:, :u_index, :]
    y_recursive, _ = solve_bsde_corridor(
        -rho_u_t, sub_grid, sub_paths, sub_dw, r, upper_r, nu, degree
    )
    recursive = float(np.mean(y_recursive[:, 0]))
    return {
        "rho_s_t": direct,
        "rho_s_u_of_minus_rho_u_t": recursive,
        "abs_diff": abs(direct - recursive),
        "rel_diff": abs(direct - recursive) / max(abs(direct), 1e-8),
    }
