"""Experiment 1 -- discounted expectation: restriction fails, proportional succeeds.

Reproduces the counterexample of Remark 3.9: the risk measure
rho_{s,t}(X) = -e^{-r(t-s)} E[X | F_s] is time-consistent and its minimal
penalty c_{s,t}(Z) vanishes only at Z = e^{-r(t-s)}. Along the restriction
factorization (which allocates the whole discount to [s, u]) the penalty
functional Gamma is +infinity, while the proportional factorization
e^{-r(t-s)} = e^{-r(u-s)} e^{-r(t-u)} attains Gamma = 0 = c_{s,t}(Z), so the
infimum in the infimal cocycle property (IC) is attained there and not at
the restriction split.

Run: python -m exp.exp1_discounted_expectation
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import simulate_brownian_paths
from src import config
from src.discounted_expectation import (
    discount_factor,
    minimal_penalty,
    proportional_factorization_penalty,
    restriction_factorization_penalty,
    rho,
    verify_time_consistency,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_test_positions(w_t: np.ndarray) -> dict[str, np.ndarray]:
    """Bounded F_t-measurable test positions X in L^infty(F_t)."""
    return {
        "constant_1": np.ones_like(w_t),
        "tanh_Wt": np.tanh(w_t),
        "indicator_Wt_gt_0": (w_t > 0.0).astype(float),
    }


def run(seed: int = config.EXP1_SEED) -> dict:
    r, s, u, t = config.EXP1_R, config.EXP1_S, config.EXP1_U, config.EXP1_T
    n_paths = config.EXP1_N_PATHS
    degree = config.EXP1_REGRESSION_DEGREE
    cap = config.EXP1_LARGE_PENALTY_CAP

    grid = np.array([s, u, t]) if s > 0 else np.array([u, t]) if s == 0 else None
    # s = 0: F_s is trivial, so no conditioning variable at s is needed; W_u is
    # needed to estimate rho_{u,t}(X) by regression, consistent with Remark 3.9.
    time_grid = np.array(sorted({s, u, t}))
    paths = simulate_brownian_paths(time_grid, n_paths, dim=1, seed=seed)
    idx = {float(tv): k for k, tv in enumerate(time_grid)}
    w_s = paths[:, idx[s], 0] if s > 0 else None
    w_u = paths[:, idx[u], 0]
    w_t = paths[:, idx[t], 0]

    positions = build_test_positions(w_t)

    time_consistency = {}
    for name, x in positions.items():
        tc = verify_time_consistency(x, r, s, u, t, w_s, w_u, degree)
        time_consistency[name] = tc

    z_star = discount_factor(r, t - s)
    penalty_at_discount = minimal_penalty(z_star, r, s, t)
    penalty_elsewhere = minimal_penalty(1.0, r, s, t)  # Z = 1 != e^{-r(t-s)} since r>0

    gamma_restriction = restriction_factorization_penalty(r, s, u, t)
    gamma_proportional = proportional_factorization_penalty(r, s, u, t)

    def cap_inf(value: float) -> float:
        return cap if np.isinf(value) else value

    log = {
        "parameters": {
            "r": r, "s": s, "u": u, "t": t,
            "n_paths": n_paths, "seed": seed, "regression_degree": degree,
        },
        "discount_factor_z_star": z_star,
        "penalty_at_z_star": penalty_at_discount,
        "penalty_at_z_equal_1": cap_inf(penalty_elsewhere),
        "penalty_at_z_equal_1_is_infinite": bool(np.isinf(penalty_elsewhere)),
        "time_consistency": time_consistency,
        "restriction_factorization": {
            "gamma": cap_inf(gamma_restriction),
            "gamma_is_infinite": bool(np.isinf(gamma_restriction)),
        },
        "proportional_factorization": {
            "gamma": cap_inf(gamma_proportional),
            "gamma_is_infinite": bool(np.isinf(gamma_proportional)),
        },
        "ic_infimum_matches_c_st": bool(
            np.isclose(gamma_proportional, penalty_at_discount, atol=1e-10)
        ),
        "restriction_is_suboptimal": bool(
            np.isinf(gamma_restriction) and not np.isinf(gamma_proportional)
        ),
    }

    assert penalty_at_discount == 0.0, "c_{s,t}(e^{-r(t-s)}) must vanish (Remark 3.9)"
    assert np.isinf(penalty_elsewhere), "c_{s,t}(Z) must be +infinity for Z != e^{-r(t-s)}"
    assert np.isinf(gamma_restriction), "restriction factorization must give Gamma = +infinity for r>0"
    assert np.isclose(gamma_proportional, 0.0), "proportional factorization must give Gamma = 0"
    for name, tc in time_consistency.items():
        assert tc["abs_diff"] < 0.05, f"time-consistency failed for {name}: {tc}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp1_discounted_expectation.json", "w") as f:
        json.dump(log, f, indent=2)

    _plot_factorizations(gamma_restriction, gamma_proportional, penalty_at_discount, cap)
    return log


def _plot_factorizations(gamma_restriction: float, gamma_proportional: float, c_st: float, cap: float) -> None:
    labels = ["restriction\n(Z1=e^{-r(t-s)}, Z2=1)", "proportional\n(e^{-r(u-s)}, e^{-r(t-u)})", "c_{s,t}(Z) target"]
    values = [
        cap if np.isinf(gamma_restriction) else gamma_restriction,
        cap if np.isinf(gamma_proportional) else gamma_proportional,
        c_st,
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["firebrick", "seagreen", "steelblue"])
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylabel("Factorization penalty Gamma (symlog scale)")
    ax.set_title("Experiment 1: restriction vs. proportional factorization\n(discounted expectation, Remark 3.9)")
    for bar, value, raw in zip(bars, values, [gamma_restriction, gamma_proportional, c_st]):
        label = "+inf (capped)" if np.isinf(raw) else f"{raw:.4g}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_factorization_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
