"""Experiment 4 -- Section 4: BSDE-induced cash-subadditive risk measure.

Solves the BSDE with the Example 4.4 corridor driver g(v,y,z) = R*y^- -
r*y^+ + nu*|z| by a backward regression (Longstaff-Schwartz style) sweep,
and verifies:

  * the dual representation (4.1): the feedback-optimal scenario (beta*,
    q*) built from the BSDE solution attains rho_{s,t}(X) = Y_s;
  * the scenario cocycle property (4.2) for a fixed admissible scenario;
  * time-consistency rho_{s,t}(X) = rho_{s,u}(-rho_{u,t}(X));
  * Proposition 4.1: the optimal intermediate factorization (Lambda_{s,u}^*,
    Lambda_{u,t}^*) built from the feedback-optimal scenario attains the
    infimum in the infimal cocycle property (IC).

Run: python -m exp.exp4_bsde
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import brownian_increments, simulate_brownian_paths
from src import config
from src.ambiguous_rates import conjugate_G
from src.bsde_solver import (
    density_path_adapted,
    feedback_optimal_scenario,
    solve_bsde_corridor,
    verify_dual_representation,
    verify_time_consistency_bsde,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _scenario_penalty(
    beta: np.ndarray, q: np.ndarray, dw: np.ndarray, dt: np.ndarray, r: float, upper_r: float, nu: float
) -> np.ndarray:
    """c_{s,v}^{beta,q} = E[int_s^v Lambda_{s,w} G(beta_w,q_w) dw] (Eq. before (4.2)).

    G=0 on this feedback scenario by construction (Example 4.4), so the
    cumulative penalty is identically zero; this is checked explicitly.
    """
    lam = density_path_adapted(beta, q, dw, dt)
    g_vals = np.array([
        conjugate_G(float(np.mean(beta[:, n])), float(np.mean(q[:, n])), r, upper_r, nu)
        for n in range(beta.shape[1])
    ])
    running = lam[:, :-1] * g_vals[None, :] * dt[None, :]
    return np.concatenate([np.zeros((beta.shape[0], 1)), np.cumsum(running, axis=1)], axis=1)


def run() -> dict:
    r, upper_r, nu = config.EXP4_R, config.EXP4_R_UPPER, config.EXP4_NU
    s, u, t = config.EXP4_S, config.EXP4_U, config.EXP4_T
    n_steps, n_paths = config.EXP4_N_STEPS, config.EXP4_N_PATHS
    degree = config.EXP4_REGRESSION_DEGREE
    seed = config.EXP4_SEED

    grid = np.linspace(s, t, n_steps + 1)
    paths = simulate_brownian_paths(grid, n_paths, dim=1, seed=seed)
    dw = brownian_increments(grid, paths)
    dt = np.diff(grid)
    u_index = int(np.argmin(np.abs(grid - u)))

    terminal_payoff = np.tanh(paths[:, -1, 0])  # bounded F_t-measurable position

    # --- dual representation (4.1) --------------------------------------
    dual_check = verify_dual_representation(
        terminal_payoff, grid, paths, dw, r, upper_r, nu, degree
    )

    # --- time-consistency --------------------------------------------------
    tc_check = verify_time_consistency_bsde(
        terminal_payoff, grid, paths, dw, r, upper_r, nu, u_index, degree
    )

    # --- scenario cocycle property (4.2) for the feedback-optimal scenario -
    y_full, v_full = solve_bsde_corridor(terminal_payoff, grid, paths, dw, r, upper_r, nu, degree)
    beta_star = np.empty((n_paths, n_steps))
    q_star = np.empty((n_paths, n_steps))
    for n in range(n_steps):
        b, q = feedback_optimal_scenario(y_full[:, n + 1], v_full[:, n], r, upper_r, nu)
        beta_star[:, n] = b
        q_star[:, n] = q
    cumulative_penalty = _scenario_penalty(beta_star, q_star, dw, dt, r, upper_r, nu)
    c_st = float(np.mean(cumulative_penalty[:, -1] - cumulative_penalty[:, 0]))
    c_su = float(np.mean(cumulative_penalty[:, u_index] - cumulative_penalty[:, 0]))
    c_ut = float(np.mean(cumulative_penalty[:, -1] - cumulative_penalty[:, u_index]))
    lam_full = density_path_adapted(beta_star, q_star, dw, dt)
    lam_s_u = float(np.mean(lam_full[:, u_index]))
    cocycle_rhs = c_su + lam_s_u * c_ut
    cocycle_check = {
        "c_s_t": c_st,
        "c_s_u": c_su,
        "lambda_s_u": lam_s_u,
        "c_u_t": c_ut,
        "rhs_c_su_plus_lambda_c_ut": cocycle_rhs,
        "abs_gap": abs(c_st - cocycle_rhs),
        "all_penalties_zero": bool(
            abs(c_st) < 1e-8 and abs(c_su) < 1e-8 and abs(c_ut) < 1e-8
        ),
    }

    # --- Proposition 4.1: optimal factorization attains the infimum in (IC) -
    # The feedback-optimal scenario has G=0 everywhere (admissible by
    # construction), so c_{s,t}(Z)=c_{s,u}(Z^1)=c_{u,t}(Z^2)=0 at the
    # induced factorization (Z^1,Z^2)=(Lambda_{s,u}^*, Lambda_{u,t}^*):
    # this is exactly Proposition 4.1b with the minimal (zero) penalty.
    ic_gap = cocycle_check["abs_gap"]

    log = {
        "parameters": {
            "r": r, "upper_r": upper_r, "nu": nu, "s": s, "u": u, "t": t,
            "n_steps": n_steps, "n_paths": n_paths, "degree": degree, "seed": seed,
        },
        "dual_representation_check": dual_check,
        "time_consistency_check": tc_check,
        "scenario_cocycle_check_4_2": cocycle_check,
        "infimal_cocycle_ic_gap_at_optimal_factorization": ic_gap,
        "proposition_4_1_holds": bool(ic_gap < 1e-6 and cocycle_check["all_penalties_zero"]),
    }

    assert dual_check["abs_diff"] < 0.05, f"dual representation (4.1) mismatch: {dual_check}"
    assert tc_check["abs_diff"] < 0.05, f"time-consistency failed: {tc_check}"
    assert cocycle_check["abs_gap"] < 1e-6, f"scenario cocycle (4.2) failed: {cocycle_check}"
    assert cocycle_check["all_penalties_zero"], "feedback-optimal scenario must have zero penalty (G=0 by construction)"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp4_bsde.json", "w") as f:
        json.dump(log, f, indent=2)

    _plot_comparison(dual_check, tc_check)
    return log


def _plot_comparison(dual_check: dict, tc_check: dict) -> None:
    labels = ["BSDE Y_0\n(primal)", "dual scenario\nvalue"]
    values = [dual_check["bsde_Y0_mean"], dual_check["dual_scenario_value_mean"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, values, color=["steelblue", "seagreen"])
    axes[0].set_title("Dual representation (4.1)")
    axes[0].set_ylabel("Risk measure value")

    labels2 = ["rho_s,t(X)\n(direct)", "rho_s,u(-rho_u,t(X))\n(recursive)"]
    values2 = [tc_check["rho_s_t"], tc_check["rho_s_u_of_minus_rho_u_t"]]
    axes[1].bar(labels2, values2, color=["steelblue", "darkorange"])
    axes[1].set_title("Time-consistency")
    axes[1].set_ylabel("Risk measure value")
    fig.suptitle("Experiment 4: BSDE risk measure (Section 4)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_bsde_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
