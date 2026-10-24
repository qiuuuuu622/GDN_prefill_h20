#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=6
PY=/opt/sglang/bin/python
"$PY" scripts/generate_cp_selected.py
"$PY" scripts/09_environment.py > logs/environment.log 2>&1
"$PY" -u scripts/02_pilot.py > logs/pilot.log 2>&1
"$PY" -u scripts/03_validate.py > logs/validation.log 2>&1
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --report-api-errors no --track-stream-ordered-races all --padding 32 --error-exitcode 99 "$PY" -u scripts/03_validate.py --memcheck > logs/memcheck.log 2>&1
"$PY" -u scripts/04_benchmark.py > logs/benchmark.log 2>&1
"$PY" -u scripts/05_graph_stress.py > logs/graph_stress.log 2>&1
"$PY" -u scripts/06_precision_probe.py > logs/precision_probe.log 2>&1
"$PY" -u scripts/07_edge_contract.py > logs/edge_contract.log 2>&1
"$PY" -u scripts/08_boundary_audit.py > logs/boundary_audit.log 2>&1
"$PY" scripts/21_report.py
"$PY" scripts/22_manifest.py
