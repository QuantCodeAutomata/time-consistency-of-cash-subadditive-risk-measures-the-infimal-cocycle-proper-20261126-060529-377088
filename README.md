# Time-consistency of cash-subadditive risk measures: The infimal cocycle property

Paper: Ferrari, Laeven, Rosazza Gianin & Zullino, *"Time-consistency of
cash-subadditive risk measures: The infimal cocycle property"*.

This repository reproduces four numerical experiments that verify, by Monte
Carlo simulation and exact analytical computation, the infimal cocycle
property (IC) for cash-subadditive dynamic risk measures discussed in the
paper. No market data is required: every experiment is a self-contained
numerical verification of a theoretical result (time-consistency, minimal
penalties, optimal factorizations, BSDE dual representations) on a simulated
Brownian filtration.

## What each experiment reproduces

* **exp1 -- `exp.exp1_discounted_expectation`** (Remark 3.9): the discounted
  expectation risk measure rho_{s,t}(X) = -e^{-r(t-s)} E[X|F_s]. Verifies
  time-consistency, that the minimal penalty c_{s,t}(Z) vanishes only at
  Z = e^{-r(t-s)}, that the restriction factorization gives an infinite
  penalty functional Gamma, and that the proportional factorization attains
  the infimum Gamma = 0 = c_{s,t}(Z) in (IC).
* **exp2 -- `exp.exp2_ambiguous_rates`** (Example 4.4): the worst-case
  discounted-expectation risk measure under an ambiguous short-rate corridor
  [r, R] with kernel-ambiguity bound nu, driver
  g(v,y,z) = R y^- - r y^+ + nu|z|. Verifies the convex conjugate G, the
  stability of the admissible density set D_{s,t} under products (the (IC)
  property in this model), and the suboptimality of the restriction
  factorization when r > 0 versus its optimality when r = 0.
* **exp3 -- `exp.exp3_penalized_rates`** (Example 4.5): the penalized
  interest-rate model with driver
  g(v,y,z) = max_{0<=beta<=R} {-beta y - beta^2/(2 theta)}. Computes the
  minimal penalty c_{s,t}(z) = 2(1-sqrt(z))^2 / (theta(t-s)) in closed form,
  derives the optimal factorization via the Cauchy-Schwarz argument of the
  paper, and compares it against the restriction and constant-rate
  factorizations across a grid of discount factors z.
* **exp4 -- `exp.exp4_bsde`** (Section 4, Proposition 4.1): a regression
  Monte Carlo BSDE solver for the driver of Example 4.4. Verifies the dual
  representation (4.1), the scenario-penalty cocycle identity (4.2),
  time-consistency of the BSDE solution, and that the optimal factorization
  attains the infimum in (IC) (Proposition 4.1).

## Repository layout

```
src/            discounted_expectation.py, ambiguous_rates.py, penalized_rates.py,
                bsde_solver.py, brownian_utils.py, config.py (all parameters)
exp/            one script per experiment, each writing JSON metrics and PNG
                figures to results/
data/           generate_synthetic.py simulates the Brownian filtration;
                no market data is used anywhere in this repository
tests/          unit tests for every src/ module
results/        RESULTS.md plus the JSON metrics and PNG figures of the latest run
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1_discounted_expectation
python -m exp.exp2_ambiguous_rates
python -m exp.exp3_penalized_rates
python -m exp.exp4_bsde
```

Each command overwrites its JSON metrics file and PNG figure(s) in
`results/`. All parameters (discount rates, corridor bounds, Monte Carlo
sample sizes, time grids) are defined in `src/config.py`.

`reproduce.sh` runs the full test suite followed by all four experiments.

## Reference

Ferrari, Laeven, Rosazza Gianin & Zullino, "Time-consistency of
cash-subadditive risk measures: The infimal cocycle property."
