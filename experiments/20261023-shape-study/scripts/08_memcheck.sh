#!/bin/bash
set -euo pipefail
cd /root/gdn35/20261023-shape-study
export CUDA_VISIBLE_DEVICES=4 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONUNBUFFERED=1
exec /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 3 /opt/sglang/bin/python scripts/memcheck.py
