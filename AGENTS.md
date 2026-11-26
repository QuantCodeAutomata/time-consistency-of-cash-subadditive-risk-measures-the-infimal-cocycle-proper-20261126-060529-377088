# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A numerical reproduction of four experiments from Ferrari, Laeven, Rosazza
Gianin & Zullino, "Time-consistency of cash-subadditive risk measures: The
infimal cocycle property": the discounted-expectation counterexample
(Remark 3.9), the ambiguous interest-rate corridor (Example 4.4), the
penalized interest-rate model (Example 4.5), and the general BSDE-induced
cash-subadditive risk measure with its dual representation and the infimal
cocycle property (Section 4, Proposition 4.1). No market data is used: every
experiment simulates a Brownian filtration and verifies the paper's formulas
and factorization properties directly.

## Layout

```
src/     discounted_expectation.py, ambiguous_rates.py, penalized_rates.py,
         bsde_solver.py, brownian_utils.py, config.py
exp/     exp1_discounted_expectation.py, exp2_ambiguous_rates.py,
         exp3_penalized_rates.py, exp4_bsde.py
data/    generate_synthetic.py (Brownian-motion simulation only)
tests/   one test module per src/ module
results/ RESULTS.md plus the JSON metrics and PNG figures of the latest run
```

## Commands

Setup:

```bash
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest
```

Experiments (paper configuration, no reduced variant is needed since there
is no market-data download and no universe to narrow):

```bash
python -m exp.exp1_discounted_expectation
python -m exp.exp2_ambiguous_rates
python -m exp.exp3_penalized_rates
python -m exp.exp4_bsde
```

`reproduce.sh` runs the tests followed by all four experiments.

## Conventions

* All experiment parameters (discount rates, corridor bounds, time grids,
  Monte Carlo sample sizes, regression degrees, random seeds) live only in
  `src/config.py` as module-level, upper-case variables, grouped by
  experiment, each with a comment stating its meaning, unit and source (a
  paper section/equation, or "not specified in the paper").
* `data/generate_synthetic.py` is the only place that generates the
  simulated Brownian paths used by the experiments; `src/` and `exp/` call
  it rather than sampling paths themselves.
* Library modules are flat files directly under `src/`; experiment scripts
  live only in `exp/` and import from `src/` and `data/`; all tests live in
  `tests/`.
* Running an experiment script overwrites its JSON metrics file and PNG
  figure(s) in `results/`. The numbers and verdicts written by hand into
  `results/RESULTS.md` must be checked against the regenerated files after
  any rerun and updated if they changed.
* No API keys or external services are used by this repository.
