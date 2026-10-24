"""Isolated GDN shape benchmark; does not patch installed packages."""
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import pathlib
import random
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_kernels import cp_delta_rule_dsl_sm90, chunk_gated_delta_rule_sm90
import original_gdn_split as split
from flashinfer import utils as fi_utils


def make_inputs(lengths, hq, hv, seed, gate):
    torch.manual_seed(seed)
    total = sum(lengths)
    q = F.normalize(torch.randn(total, hq, 128, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(total, hq, 128, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(total, hv, 128, device="cuda", dtype=torch.bfloat16)
    if gate == "slow":
        g = torch.exp(-0.05 * F.softplus(torch.randn(total, hv, device="cuda")))
    elif gate == "near_one":
        g = torch.full((total, hv), 0.999, device="cuda")
    else:
        g = torch.rand(total, hv, device="cuda").clamp_min(1e-6)
    beta = torch.randn(total, hv, device="cuda").sigmoid()
    h0 = 0.1 * torch.randn(len(lengths), hv, 128, 128, device="cuda")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    return dict(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu,
                initial_state=h0, output=torch.empty_like(v),
                output_state=torch.full_like(h0, float("nan")))


def cp_fn(*, q, k, v, g, beta, cu_seqlens, initial_state, output_state,
          output, max_seqlen, cp_chunk_len=None):
    return cp_delta_rule_dsl_sm90(output, output_state, q, k, v, g, beta,
                                 cu_seqlens, 128 ** -0.5, initial_state=initial_state,
                                 max_seqlen=max_seqlen, cp_chunk_len=cp_chunk_len)


def noncp_fn(*, q, k, v, g, beta, cu_seqlens, initial_state, output_state,
             output, max_seqlen=None):
    return chunk_gated_delta_rule_sm90(output, output_state, q, k, v, initial_state,
                                     g, beta, cu_seqlens, 128 ** -0.5, None, None, 0)


def check(x, ref, lengths):
    live = [i for i, n in enumerate(lengths) if n]
    result = {}
    for key, atol, rtol in [("output", 0.02, 0.02), ("output_state", 0.01, 0.01)]:
        a, b = x[key], ref[key]
        if key == "output_state":
            a, b = a[live], b[live]
        delta = (a.float() - b.float()).abs()
        result[key + "_max_abs"] = delta.max().item()
        result[key + "_rms"] = delta.square().mean().sqrt().item()
        result[key + "_finite"] = bool(torch.isfinite(a).all())
        result[key + "_close"] = bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))
    result["valid"] = all(result[k] for k in result if k.endswith(("_finite", "_close")))
    return result


def graph_time(graph, n):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--suite", choices=["pilot", "full", "accuracy"], default="pilot")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gate", default="slow")
    p.add_argument("--dv", action="store_true")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--selected-lengths", default=None)
    args = p.parse_args()
    outpath = pathlib.Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    versions = {x: importlib.metadata.version(x) for x in ["torch", "flashinfer-python", "nvidia-cutlass-dsl", "sglang"]}
    meta = dict(args=vars(args), versions=versions, gpu=torch.cuda.get_device_name(),
                sms=torch.cuda.get_device_properties(0).multi_processor_count,
                source_hashes={})
    for fn in [chunk_gated_delta_rule, cp_delta_rule_dsl_sm90, chunk_gated_delta_rule_sm90]:
        path = pathlib.Path(inspect.getfile(fn))
        meta["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    outpath.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))
    cases = [[8192], [8192, 1024, 1024, 1024], [8192, 8192]]
    if args.suite == "full":
        cases = [[512], [1024], [2048], [4096], [8192], [16384], [32768],
                 [8192, 1024], [8192] + [1024]*3, [8192] + [1024]*7,
                 [16384] + [1024]*3, [32768] + [1024]*3,
                 [4096] + [512]*3, [8192, 8192], [2048]*4, [1024]*8,
                 [1024, 8192, 1024], [511, 8193, 0, 1025]]
    elif args.suite == "accuracy":
        cases = [[257], [4097], [1023, 8193, 0, 513]]
    if args.selected_lengths:
        cases = json.loads(args.selected_lengths)
    with outpath.open("w") as log:
        for idx, lengths in enumerate(cases):
            hq, hv = 16 // args.tp, 32 // args.tp
            print("CASE", lengths, "H", hq, hv, flush=True)
            x = make_inputs(lengths, hq, hv, args.seed + idx, args.gate)
            noncp_fn(**x)
            torch.cuda.synchronize()
            ref = {k: x[k].clone() for k in ["output", "output_state"]}
            baseline_valid = bool(torch.isfinite(ref["output"]).all())
            assert baseline_valid
            methods = {
                "noncp": lambda: noncp_fn(**x),
                "public_auto": lambda: chunk_gated_delta_rule(**x, output_final_state=True, use_cp="auto"),
                "cp_auto_total": lambda: cp_fn(**x, max_seqlen=sum(lengths)),
                "cp_auto_exact": lambda: cp_fn(**x, max_seqlen=max(lengths)),
            }
            chunks = [512, 1024, 2048, 4096, 8192, 16384]
            for c in chunks:
                if c <= max(lengths):
                    methods[f"cp_c{c}"] = lambda c=c: cp_fn(**x, max_seqlen=max(lengths), cp_chunk_len=c)
            plan = split.plan_split(lengths, hv, meta["sms"])
            if len(lengths) > 1:
                li = max(range(len(lengths)), key=lengths.__getitem__)
                for c in chunks:
                    if c <= max(lengths) and c >= 1024:
                        methods[f"split_c{c}"] = lambda c=c: split.execute_split(
                            split.SplitPlan(li, c), lengths=lengths, **x, cp_fn=cp_fn, noncp_fn=noncp_fn)
            if args.dv:
                import dvsplit
                methods = {name: methods[name] for name in ["noncp", "public_auto"]}
                methods["dvsplit"] = lambda: dvsplit.run(**x)
                if len(lengths) > 1:
                    li = max(range(len(lengths)), key=lengths.__getitem__)
                    def dv_long_fn(cp_chunk_len=None, **kw):
                        return dvsplit.run(**kw)
                    methods["dv_long_only"] = lambda: split.execute_split(
                        split.SplitPlan(li, 0), lengths=lengths, **x,
                        cp_fn=dv_long_fn, noncp_fn=noncp_fn)
            if args.adaptive:
                import tp1_dispatch
                dispatch_plan = tp1_dispatch.plan(lengths, meta["sms"])
                methods = {name: methods[name] for name in ["noncp", "public_auto", "dvsplit"]}
                methods["adaptive"] = lambda: tp1_dispatch.run(dispatch_plan, **x)
            graphs, rows, retained_workspace = {}, {}, {}
            for name, fn in methods.items():
                row = dict(lengths=lengths, hq=hq, hv=hv, strategy=name,
                           original_plan=None if plan is None else vars(plan), gate=args.gate)
                if args.adaptive:
                    row["dispatch_plan"] = vars(dispatch_plan)
                try:
                    t0 = time.monotonic()
                    for _ in range(3): fn()
                    torch.cuda.synchronize()
                    row.update(check(x, ref, lengths))
                    row["warmup_seconds"] = time.monotonic() - t0
                    if row["valid"]:
                        # CP changes workspace sizes with chunk length. Keep all
                        # eager cache allocations alive for captured graph pointers.
                        retained_workspace[name] = list(fi_utils._cache_buf.values())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph): fn()
                        graph.replay()
                        torch.cuda.synchronize()
                        row["graph_check"] = check(x, ref, lengths)
                        row["valid"] = row["graph_check"]["valid"]
                        if row["valid"]:
                            graphs[name] = graph
                except Exception as exc:
                    row["error"] = repr(exc)
                    row["valid"] = False
                    print("ERROR", name, repr(exc), flush=True)
                rows[name] = row
                print("READY", name, row["valid"], round(row.get("warmup_seconds", 0), 2), flush=True)
            times = {name: [] for name in graphs}
            rng = random.Random(args.seed)
            for _ in range(args.repeats):
                order = list(graphs)
                rng.shuffle(order)
                for name in order:
                    for _ in range(3): graphs[name].replay()
                    times[name].append(graph_time(graphs[name], args.iterations))
            for name, row in rows.items():
                if name in times:
                    row["samples_ms"] = times[name]
                    row["median_ms"] = statistics.median(times[name])
                    row["min_ms"], row["max_ms"] = min(times[name]), max(times[name])
                log.write(json.dumps(row) + "\n")
                log.flush()
            ordered = sorted((r["median_ms"], n) for n, r in rows.items() if "median_ms" in r)
            print("RESULT", lengths, ordered, flush=True)
            graphs.clear()
            retained_workspace.clear()
            del x, ref, methods
            gc.collect()
            torch.cuda.empty_cache()
    print("COMPLETED", outpath, flush=True)


if __name__ == "__main__":
    main()
