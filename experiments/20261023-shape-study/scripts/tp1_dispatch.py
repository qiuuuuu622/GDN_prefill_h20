"""Experimental H20 TP1 planner. Plan once per batch, reuse across GDN layers.

Qualified domain: Hq=Hk=16, Hv=32, Dk=Dv=128, BF16 IO, FP32 compact
states, linear FP32 gates, no checkpoints or indexed state pools.
No installed SGLang/FlashInfer entry point is replaced by this module.
"""
from dataclasses import dataclass
import heapq
import dvsplit
from flashinfer.gdn_prefill import chunk_gated_delta_rule


@dataclass(frozen=True)
class Plan:
    strategy: str
    lengths: tuple[int, ...]
    predicted_ratio: float


def _makespan(lengths, heads, sms):
    lanes = [0] * sms
    for length in lengths:
        if length == 0:
            continue
        for _ in range(heads):
            heapq.heapreplace(lanes, lanes[0] + length)
    return max(lanes)


def plan(lengths, sms=78):
    lengths = tuple(lengths)
    if not lengths or min(lengths) < 0:
        raise ValueError("Nonnegative host sequence lengths are required")
    if sms != 78 or not sum(lengths):
        return Plan("baseline", lengths, 1.0)
    if len(lengths) == 1:
        return Plan("dvsplit", lengths, 1 / 0.68)
    base = _makespan(lengths, 32, sms)
    candidate = 0.68 * _makespan(lengths, 64, sms)
    return Plan("dvsplit" if candidate * 1.1 < base else "baseline", lengths, base / candidate)


def run(plan, **x):
    q, k, v = x["q"], x["k"], x["v"]
    if (q.shape[1:] != (16, 128) or k.shape[1:] != (16, 128)
            or v.shape[1:] != (32, 128) or q.shape[0] != sum(plan.lengths)):
        raise ValueError("This experimental dispatch is qualified only for the TP1 shape")
    if plan.strategy == "dvsplit":
        return dvsplit.run(**x)
    return chunk_gated_delta_rule(**x, output_final_state=True, use_cp="auto")
