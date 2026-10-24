"""Prepared, explicitly owned metadata for GDN prefill.

Warm up each plan outside CUDA Graph capture. Keep the plan alive for the graph's
lifetime. Calls on one device must be ordered; upstream CP workspace is shared.
"""
from dataclasses import replace
from itertools import accumulate
from .dispatch import make_plan


class GDNPrefill:
    def __init__(self, *, policy="stable"):
        if policy not in ("stable", "experimental"):
            raise ValueError("Unknown policy")
        self.policy = policy

    def prepare(self, *, lengths, geometry, checkpoint_requests=(), device="cuda", backend="auto"):
        import torch
        from importlib.metadata import version
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GPU execution requires a CUDA device; use make_plan for CPU-only planning")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        props = torch.cuda.get_device_properties(device)
        actual = (props.multi_processor_count, (props.major, props.minor))
        if actual != (geometry.sms, geometry.capability):
            raise ValueError("Geometry does not describe the selected GPU")
        if version("flashinfer-python") != "0.6.18":
            raise RuntimeError("Unified runtime is validated on FlashInfer 0.6.18 only; use the historical patch for its original baseline")
        plan = make_plan(lengths, geometry, checkpoint_requests, policy=self.policy, backend=backend)
        if plan.backend != "original":
            from .backends.compat import verify_private_sources
            verify_private_sources()
        return PreparedPlan(plan, device)

    def run(self, *, plan, q, k, v, g, beta, initial_state, output, output_state, checkpoint_states=None):
        """Write caller-owned outputs; checkpoint_states is compact [slot,Hv,Dv,Dk]."""
        plan.run(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
                 output=output, output_state=output_state, checkpoint_states=checkpoint_states)
        return output, output_state


class PreparedPlan:
    def __init__(self, plan, device):
        import torch
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
        from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm90
        from flashinfer import utils
        from .backends.loader import load
        self.torch, self.spec, self.device = torch, plan, device
        self.original, self.cp, self.utils = chunk_gated_delta_rule, delta_rule_cp_sm90, utils
        self.keepalive = {}
        self.cu_seqlens = torch.tensor((0, *accumulate(plan.lengths)), device=device, dtype=torch.int64)
        ck = plan.checkpoints
        self.packed = torch.tensor(ck.packed, device=device, dtype=torch.int64)
        self.module = load(plan.backend) if plan.backend in ("full", "v2", "cp_checkpoint") else None
        self.dense = self.gathered = self.source_ids = self.destination_ids = self.dense_starts = None
        n, geo = len(plan.lengths), plan.geometry
        state_shape = (geo.hv, geo.dv, geo.dk)
        if plan.backend == "original" and ck.requests:
            starts = (0, *accumulate(n // 64 for n in plan.lengths))
            self.dense_starts = torch.tensor(starts, device=device, dtype=torch.int64)
            self.dense = torch.empty((starts[-1], *state_shape), device=device, dtype=torch.float32)
            self.gathered = torch.empty((len(ck.requests), *state_shape), device=device, dtype=torch.float32)
            self.source_ids = torch.tensor(ck.dense_ids, device=device, dtype=torch.int64)
            self.destination_ids = torch.tensor(ck.slots, device=device, dtype=torch.int64)
        self.combined = None
        if plan.backend == "cp_checkpoint":
            self.combined = torch.empty((n + ck.capacity, *state_shape), device=device, dtype=torch.float32)
        self.groups = []
        if plan.backend == "mixed_cp":
            li = plan.long_index
            offsets = (0, *accumulate(plan.lengths))
            for lo, hi in ((li, li + 1), (0, li), (li + 1, n)):
                if lo == hi or not sum(plan.lengths[lo:hi]):
                    continue
                is_long = (lo, hi) == (li, li + 1)
                # Short groups deliberately use original non-CP, matching the historical split.
                group = replace(plan, lengths=plan.lengths[lo:hi], backend="cp" if is_long else "original",
                                long_index=None, checkpoints=type(ck).create(plan.lengths[lo:hi]))
                self.groups.append((lo, hi, offsets[lo], offsets[hi], PreparedPlan(group, device), is_long))
            self.side_stream = torch.cuda.Stream(device=device)

    @property
    def backend(self):
        return self.spec.backend

    def update_checkpoints(self, requests):
        """Update targets in-place, preserving backend, capacity and allocation layout.

        Supported for selected full/V2 and experimental native CP. Call before
        graph replay, ordered on the replay stream. Does not mutate dispatch.
        """
        from .checkpoint import CheckpointPlan
        if self.backend not in ("full", "v2", "cp_checkpoint"):
            raise ValueError("This plan requires rebuilding for checkpoint changes")
        ck = CheckpointPlan.create(self.spec.lengths, requests)
        if not ck.single_per_sequence or ck.capacity > self.spec.checkpoints.capacity:
            raise ValueError("New checkpoint plan needs different capacity or backend")
        # The kernel was captured with a selected-checkpoint protocol, including zero targets.
        if not self.spec.checkpoints.requests:
            raise ValueError("Rebuild a plan to enable checkpoints on a no-checkpoint kernel")
        if self.backend == "cp_checkpoint":
            original_mapping = {(r.sequence, r.slot) for r in self.spec.checkpoints.requests}
            if any((r.sequence, r.slot) not in original_mapping for r in ck.requests):
                raise ValueError("Native CP external scatter requires fixed sequence-to-slot mapping")
        self.packed.copy_(self.torch.tensor(ck.packed, device=self.device, dtype=self.torch.int64))

    def _validate(self, x):
        torch, geo = self.torch, self.spec.geometry
        total, n = sum(self.spec.lengths), len(self.spec.lengths)
        dtype = getattr(torch, geo.dtype)
        expected = {
            "q": ((total, geo.hq, geo.dk), dtype), "k": ((total, geo.hq, geo.dk), dtype),
            "v": ((total, geo.hv, geo.dv), dtype), "output": ((total, geo.hv, geo.dv), dtype),
            "g": ((total, geo.hv), torch.float32), "beta": ((total, geo.hv), torch.float32),
            "output_state": ((n, geo.hv, geo.dv, geo.dk), torch.float32),
        }
        if x["initial_state"] is not None:
            expected["initial_state"] = expected["output_state"]
        if self.spec.checkpoints.requests:
            expected["checkpoint_states"] = ((self.spec.checkpoints.capacity, geo.hv, geo.dv, geo.dk), torch.float32)
        elif x["checkpoint_states"] is not None:
            raise ValueError("No checkpoint output was requested by this plan")
        for key, (shape, dtype) in expected.items():
            t = x[key]
            if t is None or tuple(t.shape) != shape or t.dtype != dtype or t.device != self.device or not t.is_contiguous():
                raise ValueError(f"Invalid {key}: expected contiguous {shape}, {dtype}, {self.device}")
        for dst in ("output", "output_state", "checkpoint_states"):
            if x[dst] is None:
                continue
            for src in ("q", "k", "v", "g", "beta", "initial_state"):
                if x[src] is not None and x[dst].untyped_storage().data_ptr() == x[src].untyped_storage().data_ptr():
                    raise ValueError("Input/output storage aliasing is not supported")

    def run(self, **x):
        self._validate(x)
        if self.backend == "mixed_cp":
            current = self.torch.cuda.current_stream(self.device)
            self.side_stream.wait_stream(current)
            for lo, hi, start, end, group, is_long in self.groups:
                sub = {k: x[k][start:end] for k in ("q", "k", "v", "g", "beta", "output")}
                sub.update(initial_state=None if x["initial_state"] is None else x["initial_state"][lo:hi],
                           output_state=x["output_state"][lo:hi], checkpoint_states=None)
                if is_long:
                    with self.torch.cuda.stream(self.side_stream):
                        group._execute(sub)
                else:
                    group._execute(sub, force_noncp=True)
            current.wait_stream(self.side_stream)
        else:
            self._execute(x)

    def _execute(self, x, force_noncp=False):
        s, ck = self.spec, self.spec.checkpoints
        kw = {k: x[k] for k in ("q", "k", "v", "g", "beta", "initial_state", "output", "output_state")}
        if self.backend == "original":
            self.original(**kw, cu_seqlens=self.cu_seqlens, output_final_state=True,
                          state_checkpoints=self.dense, checkpoint_cu_starts=self.dense_starts,
                          checkpoint_every_n_tokens=64 if ck.requests else 0,
                          use_cp=False if ck.requests or force_noncp else "auto")
            if ck.requests:
                self.torch.index_select(self.dense, 0, self.source_ids, out=self.gathered)
                x["checkpoint_states"].index_copy_(0, self.destination_ids, self.gathered)
        elif self.backend in ("full", "v2"):
            q, k, v, initial, out, state = (x[a] for a in ("q", "k", "v", "initial_state", "output", "output_state"))
            if self.backend == "v2":
                total, hv, _ = v.shape
                v = v.view(total, hv * 2, 64)
                out = out.view(total, hv * 2, 64)
                state = state.view(-1, hv * 2, 64, 128)
                initial = None if initial is None else initial.view(-1, hv * 2, 64, 128)
            self.module.delta_rule_prefill_dsl_sm90(out, state, q, k, v, initial, x["g"], x["beta"],
                self.cu_seqlens, s.geometry.dk ** -.5, x["checkpoint_states"],
                self.packed if ck.requests else None, -1 if ck.requests else 0)
        elif self.backend in ("cp", "cp_checkpoint"):
            if self.backend == "cp":
                self.cp.cp_delta_rule_dsl_sm90(x["output"], x["output_state"], x["q"], x["k"], x["v"],
                    x["g"], x["beta"], self.cu_seqlens, s.geometry.dk ** -.5,
                    initial_state=x["initial_state"], max_seqlen=max(s.lengths), cp_chunk_len=s.cp_chunk)
            else:
                c, total, maximum = self.cp, sum(s.lengths), max(s.lengths)
                t = c.cp_delta_rule_t_precompute_dsl_sm90(x["k"], x["beta"], self.cu_seqlens, total, max_seqlen=maximum, _skip_check=True)
                m, local = c.cp_delta_rule_mn_precompute_dsl_sm90(x["k"], x["v"], t, x["g"], self.cu_seqlens, total,
                    cp_chunk_len=s.cp_chunk, max_seqlen=maximum, _skip_check=True)
                fixed = c.cp_delta_rule_fixup_dsl_sm90(m, local, self.cu_seqlens, total, cp_chunk_len=s.cp_chunk,
                    initial_state=x["initial_state"], _skip_check=True)
                self.module.cp_delta_rule_prefill_selected_sm90(x["output"], self.combined, x["q"], x["k"], x["v"],
                    t, fixed, x["g"], s.geometry.dk ** -.5, self.cu_seqlens, total, cp_chunk_len=s.cp_chunk,
                    max_seqlen=maximum, initial_state=x["initial_state"], checkpoint_plan=self.packed, _skip_check=True)
                # Copy only live rows/selected slots; never overwrite unspecified state with workspace garbage.
                for i, length in enumerate(s.lengths):
                    if length:
                        x["output_state"][i].copy_(self.combined[i])
                # Preserve slots not requested by runtime metadata via masked scatter kernel.
                self._scatter_native_checkpoints(x["checkpoint_states"])
        # Keep captured allocations alive when upstream cache changes sizes.
        for value in self.utils._cache_buf.values():
            if isinstance(value, self.torch.Tensor):
                self.keepalive[value.data_ptr()] = value

    def _scatter_native_checkpoints(self, target):
        # Native kernel writes directly into a persistent buffer. All initially
        # requested output slots have values; dynamic disabling must preserve caller slots.
        # GPU mask is handled by torch.where with a preallocated destination.
        n = len(self.spec.lengths)
        for i in range(n):
            # Slot is fixed by prepare for external scatter; dynamic native remapping is rejected below.
            reqs = [r for r in self.spec.checkpoints.requests if r.sequence == i]
            for req in reqs:
                self.torch.where(self.packed[i * 2 + 1] > 0,
                                 self.combined[n + req.slot], target[req.slot], out=target[req.slot])
