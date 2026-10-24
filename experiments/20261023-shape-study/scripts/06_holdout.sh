#!/bin/bash
set -euo pipefail
cd /root/gdn35/20261023-shape-study
export CUDA_VISIBLE_DEVICES=4 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONUNBUFFERED=1
exec /opt/sglang/bin/python scripts/benchmark.py --tp 1 --dv --adaptive --seed 9871 --gate near_one --iterations 100 --repeats 7 --selected-lengths '[[768],[1536],[3072],[6144],[12288],[24576],[6144,768],[6144,768,768,768],[6144,768,768,768,768,768,768,768],[12288,1536,1536,1536],[24576,768,768,768],[8192,2048,2048,2048],[8192,4096,4096,4096],[12288,12288],[3072,3072,3072,3072],[1536,1536,1536,1536,1536,1536,1536,1536],[768,12288,1536],[3000,10000,1000],[16384,8192],[8192,0,0],[4096,2048],[1024,4096,1024],[64],[128],[1],[65]]' --output evidence/holdout.jsonl
