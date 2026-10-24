"""Isolated SGLang FlashInferGDNKernel.extend component A/B, no server."""
import gc
import json
import pathlib
import random
import statistics
import torch
from flashinfer import utils as fi_utils
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import FlashInferGDNKernel
import benchmark as b
import dvsplit
import tp1_dispatch


def make_adapted(original, plan):
    def adapted(*, q, k, v, g, beta, initial_state, output_final_state,
                cu_seqlens, output=None, output_state=None, **kwargs):
        eligible = (plan.strategy == "dvsplit" and kwargs.get("checkpoint_every_n_tokens", 0) == 0
                    and kwargs.get("state_indices") is None and not kwargs.get("use_qk_l2norm_in_kernel", False)
                    and kwargs.get("scale") is None)
        if not eligible:
            return original(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
                            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
                            output=output, output_state=output_state, **kwargs)
        if output is None: output = torch.empty_like(v)
        if output_state is None: output_state = torch.empty_like(initial_state)
        dvsplit.run(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens,
                    initial_state=initial_state, output_state=output_state, output=output)
        return (output, output_state) if output_final_state else output
    return adapted


def main():
    rows = []
    for lengths in [[512], [8192], [32768], [8192,1024,1024,1024], [8192,8192]]:
        print("CASE",lengths,flush=True)
        x = b.make_inputs(lengths,16,32,732,"slow")
        plan = tp1_dispatch.plan(lengths)
        q,k,v = [x[n].unsqueeze(0) for n in ["q","k","v"]]
        log_g,beta = x["g"].log().unsqueeze(0),x["beta"].unsqueeze(0)
        n=len(lengths)
        pool0=torch.randn(n+3,32,128,128,device="cuda")*.1
        indices=torch.arange(n,0,-1,device="cuda",dtype=torch.int64)
        graphs,outputs,pools,workspace,retained={}, {}, {}, {}, {}
        for method in ["baseline", "candidate"]:
            kernel=FlashInferGDNKernel()
            if method == "candidate": kernel._prefill_fn=make_adapted(kernel._prefill_fn,plan)
            pool=pool0.clone()
            def run():
                # Same reset in both arms provides identical initial-state inputs
                # on every replay. Included in component timing, not model serving.
                pool.copy_(pool0)
                return kernel.extend(q,k,v,log_g,beta,ssm_states=pool,
                                     cache_indices=indices,query_start_loc=x["cu_seqlens"])
            for _ in range(3):result=run()
            torch.cuda.synchronize()
            outputs[method]=result[0].clone()
            pools[method]=pool.clone()
            workspace[method]=list(fi_utils._cache_buf.values())
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): result=run()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(result[0],outputs[method],atol=0,rtol=0)
            graphs[method]=graph
            retained[method]=(pool,kernel,result)
        torch.testing.assert_close(outputs["candidate"],outputs["baseline"],atol=.02,rtol=.02)
        torch.testing.assert_close(pools["candidate"],pools["baseline"],atol=.01,rtol=.01)
        times={n:[] for n in graphs}
        rng=random.Random(4321)
        for _ in range(9):
            order=list(graphs);rng.shuffle(order)
            for method in order:times[method].append(b.graph_time(graphs[method],100))
        row=dict(lengths=lengths,plan=vars(plan),samples_ms=times,
                 median_ms={n:statistics.median(t) for n,t in times.items()},
                 output_max_abs=(outputs["candidate"].float()-outputs["baseline"].float()).abs().max().item(),
                 state_pool_max_abs=(pools["candidate"]-pools["baseline"]).abs().max().item(),
                 valid=True)
        rows.append(row)
        pathlib.Path("evidence/integration.json").write_text(json.dumps(rows,indent=2))
        print("RESULT",row,flush=True)
        graphs.clear();workspace.clear();retained.clear();gc.collect();torch.cuda.empty_cache()
    print("COMPLETED",len(rows),flush=True)


if __name__=="__main__":main()
