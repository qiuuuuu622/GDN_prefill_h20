#!/bin/bash
set -euo pipefail
cd /root/gdn35/20261023-shape-study
export CUDA_VISIBLE_DEVICES=4 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONUNBUFFERED=1
exec /opt/sglang/bin/python scripts/benchmark.py --tp 1 --dv --suite full --iterations 100 --repeats 7 --output evidence/dv_full.jsonl
