"""Experiment 3 -- Example 4.5: penalized interest rates.

Reproduces the explicit minimal penalty of Example 4.5 for the driver
g(v,y,z) = max_{0<=beta<=R} {-beta*y - beta^2/(2*theta)}, computed in
closed form via a Cauchy-Schwarz argument:

    c_{s,t}(z) = 2*(1-sqrt(z))^2 / (theta*(t-s)),

attained at the optimal density path Lambda_v^* = (1-a(v-s))^2,
a = (1-sqrt(z))/(t-s). The optimal intermediate factorization Z^1 = Lambda_u^*
is compared with the restriction factorization (Z^1=z) and the constant-rate
factorization (Z^1 = z^{(u-s)/(t-s)}); both alternatives give a strictly
larger penalty Gamma than c_{s,t}(z), while the optimal split attains (IC)
exactly.

Run: python -m exp.exp3_penalized_rates
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.penalized_rates import (
    admissibility_bound,
    constant_rate_factorization_penalty,
    minimal_penalty,
    optimal_density_path,
    optimal_intermediate_factor,
    optimal_rate_path,
    restriction_factorization_penalty,
    verify_ic,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> dict:
    theta, upper_r = config.EXP3_THETA, config.EXP3_R_UPPER
    s, u, t = config.EXP3_S, config.EXP3_U, config.EXP3_T
    z = config.EXP3_Z
    z_grid = config.EXP3_Z_GRID

    bound = admissibility_bound(upper_r, t - s)
    assert z >= bound, f"z={z} violates admissibility z >= {bound:.6f} (Example 4.5)"

    ic_check = verify_ic(z, theta, s, u, t)
    gamma_restriction = restriction_factorization_penalty(z, theta, s, u, t)
    gamma_constant = constant_rate_factorization_penalty(z, theta, s, u, t)
    c_st = minimal_penalty(z, theta, t - s)
    z1_opt = ic_check["z1_optimal_intermediate"]
    z1_const = z ** ((u - s) / (t - s))

    v_grid = np.linspace(s, t, 200)
    lambda_opt = optimal_density_path(z, s, t, v_grid)
    beta_opt = optimal_rate_path(z, s, t, v_grid)
    beta_t_star = float(optimal_rate_path(z, s, t, np.array([t]))[0])

    grid_rows = []
    for zg in z_grid:
        bg = admissibility_bound(upper_r, t - s)
        if zg < bg:
            continue
        c_g = minimal_penalty(zg, theta, t - s)
        g_restr = restriction_factorization_penalty(zg, theta, s, u, t)
        g_const = constant_rate_factorization_penalty(zg, theta, s, u, t)
        z1_g = optimal_intermediate_factor(zg, s, u, t)
        z1_const_g = zg ** ((u - s) / (t - s))
        grid_rows.append({
            "z": zg,
            "c_st": c_g,
            "gamma_restriction": g_restr,
            "gamma_constant_rate": g_const,
            "z1_optimal": z1_g,
            "z1_restriction": zg,
            "z1_constant_rate": z1_const_g,
            "ordering_holds": bool(zg < z1_const_g < z1_g < 1.0 if zg < 1.0 else True),
        })

    log = {
        "parameters": {
            "theta": theta, "upper_r": upper_r, "s": s, "u": u, "t": t, "z": z,
            "admissibility_bound": bound,
        },
        "minimal_penalty_c_st": c_st,
        "ic_verification_at_optimum": ic_check,
        "restriction_factorization_gamma": gamma_restriction,
        "constant_rate_factorization_gamma": gamma_constant,
        "optimal_z1": z1_opt,
        "constant_rate_z1": z1_const,
        "restriction_z1": z,
        "ordering_z_lt_const_lt_opt_lt_1": bool(z < z1_const < z1_opt < 1.0),
        "optimal_rate_at_t": beta_t_star,
        "optimal_rate_admissible": bool(beta_t_star <= upper_r + 1e-9),
        "restriction_strictly_worse": bool(gamma_restriction > c_st + 1e-10),
        "constant_rate_strictly_worse": bool(gamma_constant > c_st + 1e-10),
        "z_grid_comparison": grid_rows,
    }

    assert ic_check["abs_gap"] < 1e-8, f"(IC) failed at the optimal factorization: {ic_check}"
    assert gamma_restriction > c_st, "restriction factorization must give a strictly larger penalty"
    assert gamma_constant > c_st, "constant-rate factorization must give a strictly larger penalty"
    assert z < z1_const < z1_opt < 1.0, "expected ordering z < z^{(u-s)/(t-s)} < Lambda_u^* < 1 failed"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp3_penalized_rates.json", "w") as f:
        json.dump(log, f, indent=2)

    _plot_paths(v_grid, lambda_opt, beta_opt, s, u, t)
    _plot_factorizations(grid_rows)
    return log


def _plot_paths(v_grid: np.ndarray, lambda_opt: np.ndarray, beta_opt: np.ndarray, s: float, u: float, t: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(v_grid, lambda_opt, color="steelblue")
    axes[0].axvline(u, color="gray", linestyle="--", label="u")
    axes[0].set_xlabel("v")
    axes[0].set_ylabel("Lambda_v* (optimal density path)")
    axes[0].set_title("Optimal density path")
    axes[0].legend()

    axes[1].plot(v_grid, beta_opt, color="firebrick")
    axes[1].axvline(u, color="gray", linestyle="--", label="u")
    axes[1].set_xlabel("v")
    axes[1].set_ylabel("beta_v* (optimal rate path)")
    axes[1].set_title("Optimal rate path (nondecreasing)")
    axes[1].legend()
    fig.suptitle("Experiment 3: optimal factorization (Example 4.5)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_optimal_paths.png", dpi=150)
    plt.close(fig)


def _plot_factorizations(grid_rows: list[dict]) -> None:
    z_vals = [row["z"] for row in grid_rows]
    c_st = [row["c_st"] for row in grid_rows]
    g_restr = [row["gamma_restriction"] for row in grid_rows]
    g_const = [row["gamma_constant_rate"] for row in grid_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_vals, c_st, "o-", label="c_{s,t}(z) = optimal Gamma", color="steelblue")
    ax.plot(z_vals, g_restr, "s--", label="restriction factorization", color="firebrick")
    ax.plot(z_vals, g_const, "^--", label="constant-rate factorization", color="darkorange")
    ax.set_xlabel("z (deterministic discount factor)")
    ax.set_ylabel("Factorization penalty Gamma")
    ax.set_title("Experiment 3: factorization comparison (Example 4.5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_factorization_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
