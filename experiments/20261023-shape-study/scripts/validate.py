"""Correctness validation: FP32 token recurrence, chunk baseline, CUDA graphs."""
import argparse
import json
import pathlib
import torch
import benchmark as b
import dvsplit


def recurrent_reference(x, lengths):
    # State layout [head, value_dim, key_dim]. Forget old state before
    # computing the delta update, then read the updated state with query.
    q, k, v = [x[n].float() for n in ["q", "k", "v"]]
    hv, hq = v.shape[1], q.shape[1]
    q = q.repeat_interleave(hv // hq, dim=1)
    k = k.repeat_interleave(hv // hq, dim=1)
    outs, states = [], []
    offset = 0
    for i, length in enumerate(lengths):
        s = torch.zeros(hv, 128, 128, device="cuda") if x["initial_state"] is None else x["initial_state"][i].clone()
        for t in range(offset, offset + length):
            s = s * x["g"][t, :, None, None]
            pred = torch.bmm(s, k[t].unsqueeze(-1)).squeeze(-1)
            delta = (v[t] - pred) * x["beta"][t, :, None]
            s = s + delta.unsqueeze(-1) * k[t].unsqueeze(-2)
            outs.append(torch.bmm(s, q[t].unsqueeze(-1)).squeeze(-1) * 128 ** -0.5)
        states.append(s)
        offset += length
    return dict(output=torch.stack(outs), output_state=torch.stack(states))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = []
    cases = [[1], [63], [64], [65], [257], [4097], [0, 129, 7],
             [1023, 8193, 0, 513], [8192, 8192], [1024]*8]
    for gate in ["uniform", "slow", "near_one"]:
        for init in [False, True]:
            for i, lengths in enumerate(cases):
                x = b.make_inputs(lengths, 16, 32, 7919+i, gate)
                if not init:
                    x["initial_state"] = None
                b.noncp_fn(**x)
                torch.cuda.synchronize()
                ref = {k:x[k].clone() for k in ["output", "output_state"]}
                if sum(lengths) <= 257:
                    rec = recurrent_reference(x, lengths)
                    row = dict(lengths=lengths, gate=gate, initial_state=init, method="noncp_vs_fp32_recurrence")
                    row.update(b.check(ref, rec, lengths))
                    rows.append(row)
                methods = {"dvsplit": lambda: dvsplit.run(**x)}
                if len(lengths) > 1:
                    li = max(range(len(lengths)), key=lengths.__getitem__)
                    def dv_fn(cp_chunk_len=None, **kw): return dvsplit.run(**kw)
                    methods["dv_long_only"] = lambda: b.split.execute_split(
                        b.split.SplitPlan(li,0), lengths=lengths, **x, cp_fn=dv_fn, noncp_fn=b.noncp_fn)
                for name, fn in methods.items():
                    x["output"].fill_(float("nan"))
                    x["output_state"].fill_(float("nan"))
                    fn()
                    torch.cuda.synchronize()
                    row = dict(lengths=lengths, gate=gate, initial_state=init, method=name)
                    row.update(b.check(x, ref, lengths))
                    empty = [i for i,l in enumerate(lengths) if not l]
                    row["empty_state_unwritten"] = bool(torch.isnan(x["output_state"][empty]).all())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): fn()
                    for _ in range(3): graph.replay()
                    torch.cuda.synchronize()
                    row["graph_check"] = b.check(x, ref, lengths)
                    row["valid"] = row["valid"] and row["graph_check"]["valid"] and row["empty_state_unwritten"]
                    rows.append(row)
                    del graph
                pathlib.Path(args.output).write_text(json.dumps(rows, indent=2))
                print(gate, init, lengths, "PASS" if all(r["valid"] for r in rows) else "FAIL", flush=True)
    assert all(r["valid"] for r in rows), "Numerical check failed; see saved errors"
    print("COMPLETED", len(rows), "checks", flush=True)


if __name__ == "__main__": main()
