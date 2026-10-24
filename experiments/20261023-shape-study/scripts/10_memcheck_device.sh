#!/bin/bash
set -euo pipefail
cd /root/gdn35/20261023-shape-study
export CUDA_VISIBLE_DEVICES=4 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONUNBUFFERED=1
# The default run records 41 cuGetProcAddress_v2 API probe errors during
# baseline CuTe compilation. Keep that log; this run disables only CUDA API
# return-code reporting, while retaining device-memory and stream-race checks.
exec /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --report-api-errors no --track-stream-ordered-races all --padding 32 --error-exitcode 3 /opt/sglang/bin/python scripts/memcheck.py
