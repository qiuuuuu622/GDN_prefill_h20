> Historical report preserved from commit `9a3aaeabf8efee7394cf46ff37f163101c72c5e9`. Paths below are relative to the repository root; timings are repository-reported, not rerun during this consolidation.

# GDN prefill on H20 (SM90): mixed-length split dispatch

FlashInfer GDN (`chunk_gated_delta_rule`) prefill optimization for **Qwen3.5-MoE-122B FP8, TP4** on **H20** (78 SMs).
Per-rank GDN shape: `Hq=4, Hv=16, D=128`. Base: flashinfer `597e3a67c` (0.7.2 dev).

## Result

CUDA-graph device time, bf16 I/O, fp32 state, H20 GPU3:

| batch (token lengths) | single launch | split | gain |
|---|---|---|---|
| 8192 + 3×1024 | 0.753 ms | 0.519 ms | **1.45x** |
| 8192 + 1×1024 | 0.756 | 0.519 | 1.46x |
| 8192 + 7×1024 | 0.755 | 0.618 | 1.22x |
| 16384 + 3×1024 | 1.500 | 0.884 | **1.70x** |
| 16384 + 7×2048 | 1.507 | 1.166 | 1.29x |
| 32768 + 3×1024 | 2.999 | 1.605 | **1.87x** |
| 4096 + 3×512 | 0.381 | 0.291 | 1.31x |
| 8192 + 3×4096 | 0.751 | (rejected) | – (split would be 0.98x) |
| 2×8192 + 2×1024 | 0.754 | (rejected) | – (0.94x) |
| 4×2048 | 0.196 | (rejected) | – (0.71x) |

Gate fitted on 54 shapes (`Hv∈{8,16}`, `L∈{4k,8k,16k}`, 1–7 short seqs of L/8..L/2): **every accepted case ≥1.11x, every rejected case ≤1.02x, 0 misclassified**.

## Why (measured)

1. **Non-CP cost = one wave, independent of head count.** One CTA per `(seq, v_head)`, 1 CTA/SM, each CTA walks its whole sequence. S=8192: 16/32/64 CTAs all take 0.756 ms; 128 CTAs 1.490 ms. TP4 single sequence uses 16/78 SMs (20%). Per active SM the kernel runs at 1.59 of 1.90 TFLOPS (84%) — the loss is occupancy, not the kernel.
2. **CP pays a second state pass.** SM90 CP = T-precompute / MN-precompute / fixup / prefill: 0.033 / 0.180 / 0.033 / 0.179 ms (S=8192, C=2048). MN and prefill each walk every chunk. CP ≈ 2× non-CP work, so whole-batch CP only wins when the batch is very unsaturated (1×8192: 1.82x; 2×8192: 0.97x; 4×8192: 0.50x).
3. **Mixed batches get the worst of both.** `use_cp="auto"` decides for the whole batch (`num_seqs*Hv < SMs/2`). `8192+3×1024` has 64 CTAs → non-CP → 0.753 ms, bounded by the 8192 sequence while most SMs idle.
4. Rejected alternatives (measured): splitting V-heads across streams (no new CTAs, 0.83–1.14x); fusing output into MN (circular dependency on the fixed-up chunk state; not feasible); Dv split inside the kernel (wgmma M=64 floor, register spill at 3 WGs, ≤1.69x, only wins at ns=2).

## What changed

- `flashinfer/gdn_split.py` (new): `plan_split()` cost model + `execute_split()`.
  - Longest sequence → `cp_delta_rule_dsl_sm90` on a side stream; the rest → `delta_rule_prefill_dsl_sm90` on the current stream; concurrent.
  - Packed varlen layout makes each group a contiguous slice → **zero copies** (views in, views out; states sliced per row). An early gather/scatter prototype lost ~0.2 ms and only reached 1.11x.
  - CP chunk = smallest power-of-two ×64 keeping `ceil(L/C)*Hv ≤ SMs`.
  - Gate: `split = max((2.7·L + rest)·Hv / (0.86·SMs), second_longest)` vs `base = max(L, total·Hv/SMs)`, accept if `split·1.1 < base`, `L ≥ 4096`.
- `flashinfer/gdn_prefill.py`: SM90 branch calls `_sm90_split_plan()` before the single launch. Active only when
  `use_cp="auto"`, **`cu_seqlens_cpu` given** (no D2H sync), `num_seqs ≥ 2`, no `state_indices`, no checkpoints, CP kernel available.
  Kill switch: `FLASHINFER_GDN_SM90_SPLIT=0`.
- `tests/gdn/test_prefill_sm90_split.py` (new): planner decisions + GPU equivalence vs single launch (long seq first/middle/last, zero-length seq, initial state).

Full diff: `patch/0001-*.patch` (`git am` onto flashinfer `597e3a67c`).

## Insertion into SGLang (pending)

SGLang SM90 path: `FlashInferGDNKernel.extend` (`sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`) → `chunk_gated_delta_rule(..., use_cp="auto")`. To enable the split, pass the host prefix sums of `forward_batch.extend_seq_lens_cpu` as `cu_seqlens_cpu=`. Not yet wired / not yet measured end-to-end in SGLang. Note SM90 sglang passes `initial_state = ssm_states[idx]` (gathered, packed) — compatible with the split.

## Test evidence (h20-2, GPU3, flashinfer HEAD via PYTHONPATH, pure JIT)

```
tests/gdn/test_prefill_sm90_split.py                       16 passed
tests/gdn/test_prefill_delta_rule.py
tests/gdn/test_prefill_cp_delta_rule.py
tests/gdn/test_prefill_auto_backend.py                     2990 passed, 866 skipped, 0 failed
```

Split vs single launch: output max|Δ| ≤ 7.8e-3 (bf16), final state max|Δ| ≤ 2.5e-3 (fp32). Zero-length sequences: neither route writes their final-state row (verified with NaN-prefilled buffer).

## Environment

`/opt/sglang` venv: torch 2.13.0+cu130, nvidia-cutlass-dsl 4.6.2, installed flashinfer 0.6.18. HEAD flashinfer run from `/root/fi_head` with a sitecustomize that hides the 0.6.18 `flashinfer_jit_cache`/`flashinfer_cubin` wheels (their shim lacks `get_jit_cache_providers`).

## Caveats

- `bench/` scripts are the exploration scripts. Early ones (`bench_gdn_tp4.py`, `bench_cp_phases.py`, `sweep2.py`) pass log-space `g` (logsigmoid) — out of domain, outputs are NaN. Their **timings** were re-measured with valid linear `g ∈ (0,1)` and match within 2%; correctness numbers above come only from valid inputs.
- Tuned and verified only on H20 (78 SMs); constants `2.7` / `0.86` are H20 fits.
- End-to-end TTFT impact not measured: GDN is 36/48 layers but shares prefill time with MoE and full attention.
