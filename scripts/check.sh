#!/usr/bin/env bash
# Test everything the way CI does — one command.
#
#   bash scripts/check.sh
#
# Runs the same gates as .github/workflows/ci.yml (the `test` + `quality` jobs):
# lint, type-check, the full numpy test suite, the smoke tests, and coverage.
# The optional jax "calibrate" job is not run here (it needs the extra); pass
# --with-calibrate to include it if jax + numpyro are installed.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
run() { echo; echo ">> $*"; "$@"; }

echo "=================  Terra check  ================="

run "$PY" -m ruff check terra tests scripts
run "$PY" -m mypy terra

echo; echo ">> test suite"
fail=0
for t in tests/test_*.py; do
  case "$t" in */test_calibrate.py) continue ;; esac   # optional jax extra
  if "$PY" "$t" >/tmp/terra_test.out 2>&1; then
    echo "   ok   $(basename "$t")"
  else
    echo "   FAIL $(basename "$t")"; tail -15 /tmp/terra_test.out; fail=1
  fi
done
[ "$fail" = 0 ] || { echo; echo "TEST FAILURES — see above"; exit 1; }

echo; echo ">> smoke tests"
run "$PY" scripts/run_demo.py --no-plot
run "$PY" -m terra.cli validate --domain aquaculture
run "$PY" -m terra.cli node --domain aquaculture --selftest
run "$PY" -m terra.cli bridge --simulate --cycles 40

run bash scripts/coverage.sh

if [ "${1:-}" = "--with-calibrate" ]; then
  echo; echo ">> calibration (optional jax extra)"
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "$PY" tests/test_calibrate.py
fi

echo; echo "=================  ALL GREEN  ================="
