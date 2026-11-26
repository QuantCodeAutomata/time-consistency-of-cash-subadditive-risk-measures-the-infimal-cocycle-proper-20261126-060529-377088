"""Module-level configuration for all four experiments.

Every numerical parameter used by ``exp/`` is defined here, grouped by
experiment, with a short note on its meaning, unit and source. The paper
(Ferrari, Laeven, Rosazza Gianin & Zullino, "Time-consistency of
cash-subadditive risk measures: The infimal cocycle property") states the
qualitative statements (signs of penalties, the factorization formulas) but
leaves Monte Carlo sample sizes, discretization steps and the numerical
values of the free parameters (r, R, theta, s, u, t, ...) unspecified; this
is noted per block below.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Experiment 1 -- discounted expectation (Remark 3.9)
# ---------------------------------------------------------------------------
EXP1_R = 0.05  # discount rate r > 0, unit 1/year; value not prescribed by the paper
EXP1_S = 0.0  # evaluation time s (years); F_0 trivial, as in Remark 3.9
EXP1_U = 0.6  # intermediate time u (years); 0 < s < u < t, not prescribed by the paper
EXP1_T = 1.0  # terminal time t (years); not prescribed by the paper
EXP1_N_PATHS = 200_000  # Monte Carlo paths; not specified in the paper
EXP1_SEED = 1  # RNG seed, for reproducibility only
EXP1_REGRESSION_DEGREE = 4  # polynomial degree used to estimate E[.|F_s] when s>0 (not specified in the paper)
EXP1_LARGE_PENALTY_CAP = 1.0e10  # finite numerical stand-in for the +infinity penalty of Remark 3.9

# ---------------------------------------------------------------------------
# Experiment 2 -- Example 4.4, ambiguous interest-rate corridor
# ---------------------------------------------------------------------------
EXP2_R = 0.1  # lower bound of the short-rate corridor r >= 0 (Example 4.4); value not prescribed, chosen > 0 so the suboptimality gap of Remark exceeds Monte Carlo noise
EXP2_R_UPPER = 0.3  # upper bound R >= r of the short-rate corridor (Example 4.4)
EXP2_NU = 0.3  # model-ambiguity bound nu >= 0 on the kernel |q| (Example 4.4)
EXP2_DIM = 1  # dimension d of the driving Brownian motion; not fixed by the paper, d=1 chosen for tractability
EXP2_S = 0.0  # evaluation time s (years)
EXP2_U = 0.4  # intermediate time u (years), 0 <= s < u < t
EXP2_T = 1.0  # terminal time t (years)
EXP2_N_STEPS = 60  # number of time steps on [s, t] for the piecewise-constant scenario processes; not specified in the paper
EXP2_N_PATHS = 100_000  # Monte Carlo paths; not specified in the paper
EXP2_N_SCENARIOS = 400  # number of (beta, q) scenarios sampled from U to approximate the ess max in (4.1)/Example 4.4; not specified in the paper
EXP2_SEED = 2  # RNG seed
EXP2_LARGE_PENALTY_CAP = 1.0e10  # numerical stand-in for +infinity
EXP2_REGRESSION_DEGREE = 5  # polynomial degree of the Longstaff-Schwartz-style basis for the BSDE-based time-consistency check (Proposition 4.1); not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 3 -- Example 4.5, penalized interest rates
# ---------------------------------------------------------------------------
EXP3_THETA = 1.0  # tolerance parameter theta > 0 (Example 4.5); value not prescribed by the paper
EXP3_R_UPPER = 0.1  # upper rate bound R > 0 (Example 4.5)
EXP3_S = 0.0  # evaluation time s (years)
EXP3_U = 0.5  # intermediate time u (years), 0 <= s < u < t
EXP3_T = 1.0  # terminal time t (years)
EXP3_Z = 0.95  # deterministic discount factor z in (0, 1]; admissibility requires z >= (1+R*(t-s)/2)^-2 = 0.9070 here
EXP3_Z_GRID = (0.91, 0.93, 0.95, 0.97, 0.99)  # grid of admissible z values used to compare factorizations (all above the 0.9070 admissibility bound)

# ---------------------------------------------------------------------------
# Experiment 4 -- BSDE numerical solution and Proposition 4.1
# ---------------------------------------------------------------------------
EXP4_R = EXP2_R  # driver of Example 4.4 reused as the g of Section 4 (paper leaves the driver choice open for the general BSDE experiment)
EXP4_R_UPPER = EXP2_R_UPPER
EXP4_NU = EXP2_NU
EXP4_S = 0.0  # evaluation time s (years)
EXP4_U = 0.5  # intermediate time u (years)
EXP4_T = 1.0  # terminal time t (years)
EXP4_N_STEPS = 100  # time steps on [s, t] for the backward-Euler BSDE scheme; not specified in the paper
EXP4_N_PATHS = 100_000  # Monte Carlo paths for the BSDE scheme; not specified in the paper
EXP4_REGRESSION_DEGREE = 5  # polynomial degree of the Longstaff-Schwartz-style basis used for conditional expectations; not specified in the paper
EXP4_SEED = 4  # RNG seed
EXP4_N_SCENARIOS_CHECK = 300  # number of random admissible scenarios used to sanity-check the cocycle property (4.2); not specified in the paper
