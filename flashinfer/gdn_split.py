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

import torch

__all__ = ["SplitPlan", "plan_split", "execute_split"]

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


_side_streams: dict[int, torch.cuda.Stream] = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    index = torch.cuda.current_device() if device.index is None else device.index
    stream = _side_streams.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index)
        _side_streams[index] = stream
    return stream


def execute_split(
    plan: SplitPlan,
    *,
    lengths: Sequence[int],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    cp_fn: Callable[..., None],
    noncp_fn: Callable[..., None],
) -> None:
    """Run ``plan`` and write results into ``output`` / ``output_state``.

    ``cp_fn`` and ``noncp_fn`` receive keyword arguments ``q, k, v, g, beta,
    cu_seqlens, initial_state, output_state, output, max_seqlen`` (plus
    ``cp_chunk_len`` for ``cp_fn``) describing one contiguous group.  States
    are packed per sequence (``[num_seqs, H, V, K]``), so group rows are
    contiguous slices too.
    """
    li = plan.long_idx
    num_seqs = len(lengths)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    s, e = starts[li], starts[li + 1]

    def rows(t: Optional[torch.Tensor], lo: int, hi: int) -> Optional[torch.Tensor]:
        return None if t is None else t[lo:hi]

    def group(lo: int, hi: int) -> dict:
        tok_lo, tok_hi = starts[lo], starts[hi]
        cu = cu_seqlens[lo : hi + 1]
        if tok_lo:
            cu = cu - cu_seqlens[lo]
        return dict(
            q=q[tok_lo:tok_hi],
            k=k[tok_lo:tok_hi],
            v=v[tok_lo:tok_hi],
            g=g[tok_lo:tok_hi],
            beta=beta[tok_lo:tok_hi],
            cu_seqlens=cu,
            initial_state=rows(initial_state, lo, hi),
            output_state=rows(output_state, lo, hi),
            output=output[tok_lo:tok_hi],
            max_seqlen=max(lengths[lo:hi]),
        )

    long_group = group(li, li + 1)
    short_groups = [group(lo, hi) for lo, hi in ((0, li), (li + 1, num_seqs)) if hi > lo]

    device = q.device
    side = _side_stream(device)
    current = torch.cuda.current_stream(device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        cp_fn(**long_group, cp_chunk_len=plan.cp_chunk_len)
    # cu_seqlens of the long group may be a fresh tensor from the main stream.
    long_group["cu_seqlens"].record_stream(side)
    for short in short_groups:
        noncp_fn(**short)
    current.wait_stream(side)
