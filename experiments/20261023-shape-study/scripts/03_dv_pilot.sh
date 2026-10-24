#!/bin/bash
set -euo pipefail
cd /root/gdn35/20261023-shape-study
export CUDA_VISIBLE_DEVICES=4 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONUNBUFFERED=1
exec /opt/sglang/bin/python scripts/benchmark.py --tp 1 --dv --selected-lengths '[[257],[8192],[8192,1024,1024,1024],[8192,8192]]' --iterations 100 --repeats 5 --output evidence/dv_pilot.jsonl
