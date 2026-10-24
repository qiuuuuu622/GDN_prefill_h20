"""CPU checkpoint contract. Tokens are relative to the current extend, after consumption."""
from dataclasses import dataclass
from itertools import accumulate


@dataclass(frozen=True)
class CheckpointRequest:
    sequence: int
    token: int
    slot: int


@dataclass(frozen=True)
class CheckpointPlan:
    requests: tuple[CheckpointRequest, ...]
    packed: tuple[int, ...]
    dense_ids: tuple[int, ...]
    slots: tuple[int, ...]
    capacity: int
    single_per_sequence: bool

    @classmethod
    def create(cls, lengths, requests=()):
        requests = tuple(requests)
        starts = (0, *accumulate(n // 64 for n in lengths))
        packed = [0] * (len(lengths) * 2)
        counts = [0] * len(lengths)
        dense_ids, slots = [], []
        for req in requests:
            if any(type(v) is not int for v in (req.sequence, req.token, req.slot)):
                raise ValueError("Checkpoint fields must be integers")
            if not 0 <= req.sequence < len(lengths):
                raise ValueError("Invalid checkpoint sequence")
            if req.token <= 0 or req.token % 64 or req.token > lengths[req.sequence]:
                raise ValueError("Checkpoint must be an in-range positive 64-token boundary")
            if req.slot < 0 or req.slot in slots:
                raise ValueError("Checkpoint output slots must be nonnegative and unique")
            counts[req.sequence] += 1
            packed[2 * req.sequence:2 * req.sequence + 2] = [req.slot, req.token]
            dense_ids.append(starts[req.sequence] + req.token // 64 - 1)
            slots.append(req.slot)
        return cls(requests, tuple(packed), tuple(dense_ids), tuple(slots),
                   max(slots, default=-1) + 1, max(counts, default=0) <= 1)
