#!/usr/bin/env bash
# Run the framework-free test scripts under coverage and print a report.
#   bash scripts/coverage.sh
# Runs the full numpy test suite (everything except the optional jax calibrate
# test, which the quality job does not install) so the coverage number reflects
# the whole package, not just the core engine.
set -e
cd "$(dirname "$0")/.."

coverage erase
for t in tests/test_*.py; do
  case "$t" in
    */test_calibrate.py) continue ;;   # needs the optional jax extra
  esac
  coverage run -a --source=terra "$t" >/dev/null
done
coverage report
