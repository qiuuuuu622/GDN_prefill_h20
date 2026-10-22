"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SM90 mixed-length split dispatch (flashinfer/gdn_split.py): planner decisions
on the host, and GPU equivalence of the split route against the single
non-CP launch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashinfer import gdn_split


# ---------------------------------------------------------------- planner ---

# (lengths, num_v_heads, expect_split) measured on H20 (78 SMs).
_PLANS = [
    ([8192, 1024, 1024, 1024], 16, True),  # 1.45x
    ([1024, 8192, 1024], 16, True),  # long sequence in the middle
    ([16384, 2048, 2048, 2048], 16, True),  # 1.54x
    ([8192] + [1024] * 7, 16, True),  # 1.22x
    ([8192, 4096, 4096, 4096], 16, False),  # 0.98x
    ([8192] + [2048] * 7, 16, False),  # 0.95x
    ([8192, 8192, 1024, 1024], 16, False),  # two long sequences
    ([2048] * 4, 16, False),  # balanced
    ([2048, 256], 16, False),  # longest too short for CP
    ([8192], 16, False),  # single sequence: CP heuristic's job
]


@pytest.mark.parametrize("lengths,num_v_heads,expect_split", _PLANS)
def test_plan_split_decisions(lengths, num_v_heads, expect_split):
    plan = gdn_split.plan_split(lengths, num_v_heads, 78)
    assert (plan is not None) == expect_split
    if plan is not None:
        assert lengths[plan.long_idx] == max(lengths)
        chunks = -(-max(lengths) // plan.cp_chunk_len)
        assert chunks * num_v_heads <= 78
        assert plan.cp_chunk_len % 64 == 0


def test_plan_split_chunk_is_smallest_one_wave():
    plan = gdn_split.plan_split([16384, 1024, 1024, 1024], 16, 78)
    assert plan is not None and plan.cp_chunk_len == 4096


# ------------------------------------------------------------------- GPU ---


def _skip_unless_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 split dispatch requires Hopper")
    from flashinfer.gdn_kernels import cp_delta_rule_dsl_sm90

    if cp_delta_rule_dsl_sm90 is None:
        pytest.skip("SM90 CP delta-rule kernel is unavailable")


def _inputs(lengths, num_q_heads, num_v_heads, head_size, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    total = sum(lengths)
    dtype = torch.bfloat16
    q = torch.randn(total, num_q_heads, head_size, dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(total, num_q_heads, head_size, device=device), p=2, dim=-1
    ).to(dtype)
    v = torch.randn(total, num_v_heads, head_size, dtype=dtype, device=device)
    g = torch.rand(total, num_v_heads, device=device).clamp_min(1e-30)
    beta = torch.rand(total, num_v_heads, device=device).sigmoid()
    h0 = 0.1 * torch.randn(
        len(lengths), num_v_heads, head_size, head_size, device=device
    )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int64, device=device)
    return q, k, v, g, beta, h0, cu, offsets


@pytest.mark.parametrize(
    "lengths",
    [
        [8192, 1024, 1024, 1024],
        [1024, 1024, 8192],
        [512, 8192, 0, 1024],
        [16384, 2048, 2048],
    ],
)
def test_split_matches_single_launch(lengths, monkeypatch):
    _skip_unless_sm90()
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from flashinfer.utils import get_device_sm_count

    num_q_heads, num_v_heads, head_size = 4, 16, 128
    if gdn_split.plan_split(lengths, num_v_heads, get_device_sm_count("cuda")) is None:
        pytest.skip("planner keeps a single launch on this device")
    q, k, v, g, beta, h0, cu, offsets = _inputs(
        lengths, num_q_heads, num_v_heads, head_size
    )

    def run(split: bool):
        monkeypatch.setenv("FLASHINFER_GDN_SM90_SPLIT", "1" if split else "0")
        return chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state=h0, output_final_state=True,
            cu_seqlens=cu, cu_seqlens_cpu=offsets,
        )

    ref_out, ref_state = run(split=False)
    out, state = run(split=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    # Neither route writes the final state of a zero-length sequence (the
    # single launch leaves that row of the auto-allocated buffer untouched).
    live = [i for i, length in enumerate(lengths) if length > 0]
    torch.testing.assert_close(state[live], ref_state[live], atol=1e-2, rtol=1e-2)


def test_split_needs_host_lengths(monkeypatch):
    """Without cu_seqlens_cpu the split would need a device sync: stay single."""
    _skip_unless_sm90()
    from flashinfer import gdn_prefill

    calls = []
    monkeypatch.setattr(
        gdn_prefill, "_run_sm90_split", lambda *a, **kw: calls.append(1)
    )
    lengths = [8192, 1024, 1024, 1024]
    q, k, v, g, beta, h0, cu, _ = _inputs(lengths, 4, 16, 128)
    gdn_prefill.chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=cu)
    assert not calls
