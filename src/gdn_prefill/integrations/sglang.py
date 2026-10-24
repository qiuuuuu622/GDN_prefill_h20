"""Explicit SGLang metadata bridge; no import-time monkey patch or global logging.

This bridge does not claim a newly validated service deployment. The historical
35B hook remains in qwen35/runtime. Integrators prepare plans at scheduler batch
construction, then reuse them across GDN layers with the same geometry/metadata.
"""
from ..checkpoint import CheckpointRequest


def requests_from_tracking(*, lengths, prefix_lengths, track_lengths, track_mask, interval):
    """Convert consumer absolute-prefix targets to extend-local checkpoint targets.

    Mirrors the historical 35B consumer: an exact interval-aligned track position
    uses the final-state path and requests no intermediate checkpoint here. Slots
    correspond to compact consumer order; consumers with other semantics must
    supply explicit CheckpointRequest entries instead of using this helper.
    """
    n = len(lengths)
    if interval <= 0 or interval % 64:
        raise ValueError("Tracking interval must be a positive multiple of 64")
    if any(len(values) != n for values in (prefix_lengths, track_lengths, track_mask)):
        raise ValueError("Tracking metadata lengths must match the batch")
    requests = []
    for seq, (length, prefix, tracked, keep) in enumerate(zip(lengths, prefix_lengths, track_lengths, track_mask)):
        if not keep:
            continue
        relative = tracked - prefix
        if relative % interval == 0:
            continue
        target = relative // interval * interval
        if not 0 < target <= length:
            raise ValueError("Unsupported tracking target; preserve original SGLang planner")
        requests.append(CheckpointRequest(seq, target, len(requests)))
    return tuple(requests)
