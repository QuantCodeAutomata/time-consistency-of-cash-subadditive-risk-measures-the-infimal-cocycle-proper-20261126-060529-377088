#!/usr/bin/env bash
# Runs the test suite and all four experiments of this repository.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python -m pytest
python -m exp.exp1_discounted_expectation
python -m exp.exp2_ambiguous_rates
python -m exp.exp3_penalized_rates
python -m exp.exp4_bsde
