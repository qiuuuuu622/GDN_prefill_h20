"""Mixed-length split dispatch for the chunked GDN prefill kernel (SM90).

The non-CP kernel launches one CTA per ``(sequence, value head)`` and each CTA
walks its whole sequence, so a batch with one long sequence finishes only when
that sequence's ``num_v_heads`` CTAs do; every other SM idles.  The CP kernel
splits a sequence into chunks and fills the machine, but pays a second state
pass, so running it over a whole balanced batch loses.

The split runs the longest sequence through CP on a side stream and the rest
through the non-CP kernel on the current stream, concurrently.  Packed varlen
layout makes every group a contiguous slice (prefix | long | suffix), so no
tensor is copied: each kernel reads input views and writes output views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence


__all__ = ["SplitPlan", "plan_split"]

_BLOCK = 64
# Below this the longest sequence has too few chunks to pay for CP's second
# state pass (smallest length measured to win on H20).
_MIN_LONG_TOKENS = 4096
# SM-time of CP on one sequence, in units of "one CTA walking it once": two
# state passes at ~74% of the non-CP kernel's per-CTA efficiency.
_CP_WORK_FACTOR = 2.7
# Fraction of the machine the two concurrent launches can actually fill.
_SATURATION = 0.86


@dataclass(frozen=True)
class SplitPlan:
    long_idx: int
    cp_chunk_len: int


def _cp_chunk_len(longest: int, num_v_heads: int, num_sms: int) -> int:
    """Smallest power-of-two multiple of 64 keeping CP CTAs within one wave."""
    chunk_len = _BLOCK
    while -(-longest // chunk_len) * num_v_heads > num_sms:
        chunk_len *= 2
    return chunk_len


def plan_split(
    lengths: Sequence[int],
    num_v_heads: int,
    num_sms: int,
    min_gain: float = 1.1,
) -> Optional[SplitPlan]:
    """Return a plan when splitting the longest sequence out should win.

    Cost model in units of "one CTA advancing one token":

    * non-CP: the batch ends when the longest sequence's CTAs do, or when the
      machine has drained every CTA, whichever is later:
      ``max(L, total * Hv / SMs)``.
    * split: CP spends ``_CP_WORK_FACTOR * L * Hv`` and the short sequences
      ``rest * Hv``, shared by ``SMs * _SATURATION``; no short CTA can finish
      before its own sequence, so the second-longest length is a floor.

    Fitted on H20 (78 SMs) over ``Hv in {8, 16}``, ``L in {4k, 8k, 16k}``,
    1-7 short sequences of ``L/8`` to ``L/2``: every case it accepts measured
    >= 1.11x, every case it rejects measured <= 1.02x.
    """
    if len(lengths) < 2:
        return None
    long_idx = max(range(len(lengths)), key=lambda i: lengths[i])
    longest = lengths[long_idx]
    if longest < _MIN_LONG_TOKENS:
        return None
    rest = sum(lengths) - longest
    second = max(length for i, length in enumerate(lengths) if i != long_idx)

    base = max(longest, (longest + rest) * num_v_heads / num_sms)
    split = max(
        (_CP_WORK_FACTOR * longest + rest) * num_v_heads / (num_sms * _SATURATION),
        second,
    )
    if split * min_gain >= base:
        return None
    return SplitPlan(long_idx, _cp_chunk_len(longest, num_v_heads, num_sms))


