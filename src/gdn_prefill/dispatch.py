"""Host-side plans. Cost coefficients are H20 measurements, not a hardware autotuner."""
from dataclasses import dataclass
import heapq
from .checkpoint import CheckpointPlan


@dataclass(frozen=True)
class Geometry:
    hq: int
    hv: int
    dk: int = 128
    dv: int = 128
    sms: int = 78
    capability: tuple[int, int] = (9, 0)
    dtype: str = "bfloat16"

    @property
    def measured(self):
        return ((self.hq, self.hv) in ((4, 16), (16, 32)) and
                (self.dk, self.dv, self.sms, self.capability, self.dtype) ==
                (128, 128, 78, (9, 0), "bfloat16"))


def choose_v(lengths, hv, sms):
    if len(lengths) == 1:
        return "v2"
    def makespan(heads):
        loads = [0] * sms
        for n in lengths:
            if n:
                for _ in range(heads):
                    heapq.heapreplace(loads, loads[0] + (n + 63) // 64)
        return max(loads)
    return "v2" if .67 * makespan(2 * hv) < .85 * makespan(hv) else "full"


@dataclass(frozen=True)
class DispatchPlan:
    lengths: tuple[int, ...]
    geometry: Geometry
    checkpoints: CheckpointPlan
    backend: str
    reason: str
    cp_chunk: int = 2048
    long_index: int | None = None


def make_plan(lengths, geometry, requests=(), *, policy="stable", backend="auto"):
    lengths = tuple(lengths)
    if not lengths or any(type(n) is not int or n < 0 for n in lengths) or not sum(lengths):
        raise ValueError("A nonempty batch with nonnegative integer lengths is required")
    if geometry.hq <= 0 or geometry.hv <= 0 or geometry.hv % geometry.hq:
        raise ValueError("Invalid head grouping")
    if policy not in ("stable", "experimental"):
        raise ValueError("Unknown policy")
    names = ("auto", "original", "full", "v2", "cp", "mixed_cp", "cp_checkpoint")
    if backend not in names:
        raise ValueError("Unknown backend")
    checkpoints = CheckpointPlan.create(lengths, requests)
    has_cp = bool(checkpoints.requests)
    reason = "explicit backend"
    if backend == "cp_checkpoint" and policy != "experimental":
        raise ValueError("CP checkpoint has unresolved non-CP regressions; experimental policy required")
    if backend in ("cp", "mixed_cp") and has_cp:
        raise ValueError("These CP backends do not implement checkpoint output")
    if backend == "cp_checkpoint" and not has_cp:
        raise ValueError("Use cp for calls without checkpoints")
    if backend in ("full", "v2", "cp", "mixed_cp", "cp_checkpoint"):
        if not geometry.measured or (has_cp and not checkpoints.single_per_sequence):
            raise ValueError("Explicit optimized backend does not support this contract")
    if backend == "auto":
        if not geometry.measured or len(lengths) > 8:
            backend, reason = "original", "outside measured optimization envelope"
        elif has_cp and not checkpoints.single_per_sequence:
            backend, reason = "original", "multiple checkpoints require dense fallback"
        elif has_cp:
            backend, reason = choose_v(lengths, geometry.hv, geometry.sms), "H20 selected-checkpoint CTA estimate"
        elif geometry.hq == 4:
            # Existing 122B mixed-CP planner; single/balanced calls keep upstream auto.
            from .backends.split_plan import plan_split
            split = plan_split(lengths, geometry.hv, geometry.sms)
            if split is not None:
                return DispatchPlan(lengths, geometry, checkpoints, "mixed_cp",
                                    "historical H20 mixed-length CP estimate", split.cp_chunk_len, split.long_idx)
            backend, reason = "original", "upstream no-checkpoint auto dispatch"
        else:
            backend, reason = "original", "35B no-checkpoint path unchanged"
    long_index = max(range(len(lengths)), key=lengths.__getitem__) if backend == "mixed_cp" else None
    return DispatchPlan(lengths, geometry, checkpoints, backend, reason, 2048, long_index)
