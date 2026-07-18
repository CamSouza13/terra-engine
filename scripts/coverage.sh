#!/usr/bin/env bash
# Run the framework-free test scripts under coverage and print a report.
#   bash scripts/coverage.sh
set -e
cd "$(dirname "$0")/.."

coverage erase
for t in engine ingest control validate diagnostics cli node robust; do
  coverage run -a --source=terra "tests/test_${t}.py" >/dev/null
done
coverage report
