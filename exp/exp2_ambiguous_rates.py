"""Experiment 2 -- Example 4.4: ambiguous interest-rate corridor.

Reproduces the worst-case discounted-expectation risk measure driven by
g(v,y,z) = R*y^- - r*y^+ + nu*|z|, whose convex conjugate G vanishes exactly
on the admissible corridor r<=beta<=R, |q|<=nu and is +infinity outside it.
The infimal cocycle property (IC) reduces to the stability of the
admissible density set D_{s,t} under products, D_{s,t} = D_{s,u} . D_{u,t};
when r>0 the restriction factorization (Z_u, Z (x) Z_u) is not optimal,
since E[Z (x) Z_u | F_u] = 1 is incompatible with sup_{Z^2 in D_{u,t}}
E[Z^2|F_u] = e^{-r(t-u)} < 1.

Run: python -m exp.exp2_ambiguous_rates
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
from src.ambiguous_rates import (
    conjugate_G,
    density_path,
    restriction_vs_products_check,
    sample_admissible_scenario,
    worst_case_rho,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def check_conjugate_G(r: float, upper_r: float, nu: float) -> list[dict]:
    """Table of G(v,beta,q) at points inside and outside the corridor."""
    probe_points = [
        (r, 0.0, True), (upper_r, 0.0, True), ((r + upper_r) / 2, nu / 2, True),
        (r - 0.05, 0.0, False), (upper_r + 0.05, 0.0, False), (r, nu + 0.05, False),
    ]
    rows = []
    for beta, q, expected_admissible in probe_points:
        g_val = conjugate_G(beta, q, r, upper_r, nu)
        rows.append({
            "beta": beta, "q": q,
            "G": None if np.isinf(g_val) else g_val,
            "G_is_infinite": bool(np.isinf(g_val)),
            "expected_admissible": expected_admissible,
            "matches_expectation": bool(np.isinf(g_val) != expected_admissible),
        })
    return rows


def compute_time_consistency(
    seed: int, n_paths: int, n_scenarios: int,
    r: float, upper_r: float, nu: float, dim: int,
    s: float, u: float, t: float, n_steps: int,
) -> dict:
    """rho_{s,t}(X) vs rho_{s,u}(-rho_{u,t}(X)) over a finite admissible-scenario
    sample of the worst-case risk measure of Example 4.4 (dual formula (4.1))."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(s, t, n_steps + 1)
    paths = simulate_brownian_paths(grid, n_paths, dim=dim, seed=seed)
    dw = brownian_increments(grid, paths)
    dt = np.diff(grid)
    u_index = int(np.argmin(np.abs(grid - u)))

    x_values = np.tanh(paths[:, -1, :].sum(axis=1))  # bounded F_t-measurable position

    # --- rho_{s,t}(X) over the admissible scenario family ---------------
    terminal_densities = np.empty((n_scenarios, n_paths))
    for k in range(n_scenarios):
        beta_path, q_path = sample_admissible_scenario(rng, n_steps, dim, r, upper_r, nu)
        terminal_densities[k] = density_path(beta_path, q_path, dw, dt)[:, -1]
    rho_s_t_values, _ = worst_case_rho(x_values, terminal_densities, conditioning=None)
    rho_s_t = float(np.mean(rho_s_t_values))

    # --- rho_{u,t}(X): ess max over scenarios restricted to [u, t], -----
    # conditioned on F_u (regression on W_u), as in the dual formula (4.1).
    terminal_densities_u_t = np.empty((n_scenarios, n_paths))
    for k in range(n_scenarios):
        beta_path, q_path = sample_admissible_scenario(rng, n_steps - u_index, dim, r, upper_r, nu)
        terminal_densities_u_t[k] = density_path(
            beta_path, q_path, dw[:, u_index:, :], dt[u_index:]
        )[:, -1]
    w_u = paths[:, u_index, :]
    rho_u_t_values, _ = worst_case_rho(x_values, terminal_densities_u_t, conditioning=w_u)

    # --- rho_{s,u}(-rho_{u,t}(X)): ess max over scenarios on [s, u] -----
    terminal_densities_s_u = np.empty((n_scenarios, n_paths))
    for k in range(n_scenarios):
        beta_path, q_path = sample_admissible_scenario(rng, u_index, dim, r, upper_r, nu)
        terminal_densities_s_u[k] = density_path(
            beta_path, q_path, dw[:, :u_index, :], dt[:u_index]
        )[:, -1]
    rho_s_u_of_minus_values, _ = worst_case_rho(
        -rho_u_t_values, terminal_densities_s_u, conditioning=None
    )
    rho_s_u_of_minus = float(np.mean(rho_s_u_of_minus_values))

    return {
        "n_paths": n_paths,
        "n_scenarios": n_scenarios,
        "rho_s_t": rho_s_t,
        "rho_s_u_of_minus_rho_u_t": rho_s_u_of_minus,
        "abs_diff": abs(rho_s_t - rho_s_u_of_minus),
        "rel_diff": abs(rho_s_t - rho_s_u_of_minus) / max(abs(rho_s_t), 1e-8),
    }


def run(seed: int = config.EXP2_SEED) -> dict:
    r, upper_r, nu = config.EXP2_R, config.EXP2_R_UPPER, config.EXP2_NU
    dim = config.EXP2_DIM
    s, u, t = config.EXP2_S, config.EXP2_U, config.EXP2_T
    n_steps, n_paths = config.EXP2_N_STEPS, config.EXP2_N_PATHS
    n_scenarios = config.EXP2_N_SCENARIOS
    cap = config.EXP2_LARGE_PENALTY_CAP

    time_consistency = compute_time_consistency(
        seed, n_paths, n_scenarios, r, upper_r, nu, dim, s, u, t, n_steps
    )

    # Scenario-sampling sensitivity: the finite sample of admissible scenarios
    # used to approximate the ess-max in (4.1) is the only approximation in this
    # experiment, so the abs_diff above should shrink as n_scenarios grows (at a
    # reduced path count, to keep the sweep cheap).
    sensitivity_n_paths = max(2_000, n_paths // 5)
    sensitivity_scenarios = sorted({max(50, n_scenarios // 4), n_scenarios, n_scenarios * 4})
    scenario_sensitivity = [
        compute_time_consistency(
            seed + 100 + k, sensitivity_n_paths, n_sc,
            r, upper_r, nu, dim, s, u, t, n_steps,
        )
        for k, n_sc in enumerate(sensitivity_scenarios)
    ]

    conjugate_table = check_conjugate_G(r, upper_r, nu)

    rng = np.random.default_rng(seed)
    products_check = restriction_vs_products_check(
        rng, r, upper_r, nu, dim, s, u, t, n_steps, n_paths, seed=seed + 1
    )
    products_check_r_zero = restriction_vs_products_check(
        rng, 0.0, upper_r, nu, dim, s, u, t, n_steps, n_paths, seed=seed + 2
    )

    log = {
        "parameters": {
            "r": r, "upper_r": upper_r, "nu": nu, "dim": dim,
            "s": s, "u": u, "t": t, "n_steps": n_steps, "n_paths": n_paths,
            "n_scenarios": n_scenarios, "seed": seed,
        },
        "conjugate_G_table": conjugate_table,
        "time_consistency": time_consistency,
        "scenario_sensitivity": scenario_sensitivity,
        "restriction_vs_products_r_gt_0": products_check,
        "restriction_vs_products_r_eq_0": products_check_r_zero,
        "restriction_suboptimal_when_r_positive": bool(
            products_check["restriction_conditional_expectation_mean"]
            > products_check["theoretical_bound_e_minus_r_t_minus_u"] + 1e-3
        ),
        "restriction_admissible_when_r_zero": bool(
            products_check_r_zero["restriction_conditional_expectation_mean"]
            <= products_check_r_zero["theoretical_bound_e_minus_r_t_minus_u"] + 2e-3
        ),
    }

    mismatches = [row for row in conjugate_table if not row["matches_expectation"]]
    assert not mismatches, (
        f"conjugate_G must be finite (=0) exactly on the admissible corridor: {mismatches}"
    )
    assert time_consistency["abs_diff"] < 0.1, f"time-consistency failed: {time_consistency}"
    assert products_check["cocycle_identity_max_abs_gap"] < 1e-8, "cocycle identity Lambda_st=Lambda_su*Lambda_ut failed"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp2_ambiguous_rates.json", "w") as f:
        json.dump(log, f, indent=2)

    _plot_restriction_bound(products_check, products_check_r_zero, cap)
    return log


def _plot_restriction_bound(check_r_pos: dict, check_r_zero: dict, cap: float) -> None:
    labels = ["r > 0\n(restriction split)", "r = 0\n(restriction split)"]
    e_values = [
        check_r_pos["restriction_conditional_expectation_mean"],
        check_r_zero["restriction_conditional_expectation_mean"],
    ]
    bound_values = [
        check_r_pos["theoretical_bound_e_minus_r_t_minus_u"],
        check_r_zero["theoretical_bound_e_minus_r_t_minus_u"],
    ]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, e_values, width, label="E[Z (x) Z_u | F_u] (restriction)", color="firebrick")
    ax.bar(x + width / 2, bound_values, width, label="sup_{Z2 in D_u,t} E[Z2|F_u] = e^{-r(t-u)}", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Conditional expectation")
    ax.set_title("Experiment 2: restriction factorization vs. admissible bound\n(Example 4.4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_restriction_bound.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
