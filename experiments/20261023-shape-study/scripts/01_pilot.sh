#!/bin/bash
set -euo pipefail
cd /root/gdn35/20261023-shape-study
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
exec /opt/sglang/bin/python scripts/benchmark.py --tp 1 --suite pilot --iterations 100 --repeats 5 --output evidence/pilot.jsonl
