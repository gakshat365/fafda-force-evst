#!/usr/bin/env bash
# Runs the whole analysis, step by step. Downloads are cached, so re-runs are fast.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"

$PY -m pytest -q
for step in 01_build_grid 02_fetch_gbif 03_extract_environment 04_build_dataset \
            05_model_richness 06_model_occurrence 07_protected_areas 08_figures; do
    echo "=== ${step} ==="
    $PY "scripts/${step}.py"
done
