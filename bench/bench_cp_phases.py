import torch
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm90 import (
    cp_delta_rule_t_precompute_dsl_sm90 as t_pre,
    cp_delta_rule_mn_precompute_dsl_sm90 as mn_pre,
    cp_delta_rule_fixup_dsl_sm90 as fixup,
    cp_delta_rule_prefill_dsl_sm90 as pf,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import choose_cp_chunk_len_host

torch.manual_seed(0)
dev="cuda"; Hq,Hv,D=4,16,128
def timeit(fn,iters=30,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    a,c=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    c.record(); torch.cuda.synchronize(); return a.elapsed_time(c)/iters

for ns,S in [(1,8192),(2,8192),(4,8192),(1,16384)]:
    T=ns*S
    q=torch.randn(T,Hq,D,device=dev,dtype=torch.bfloat16)
    k=torch.randn(T,Hq,D,device=dev,dtype=torch.bfloat16)
    v=torch.randn(T,Hv,D,device=dev,dtype=torch.bfloat16)
    alpha=torch.rand(T,Hv,device=dev,dtype=torch.float32).sigmoid().log()
    beta=torch.rand(T,Hv,device=dev,dtype=torch.float32).sigmoid()
    cu=torch.tensor([i*S for i in range(ns+1)],device=dev,dtype=torch.int64)
    cl=choose_cp_chunk_len_host(S,Hv,78,device_capability=(9,0),total_seqlen=T,num_seqs=ns,device_name="NVIDIA H20")
    print(f"\n--- ns={ns} S={S} chunk_len={cl} ctas/head={-(-S//cl)}", flush=True)
    tt=timeit(lambda: t_pre(k,beta,cu,T,max_seqlen=S,_skip_check=True,_device=dev))
    print(f"  T precompute    {tt:8.3f} ms", flush=True)
    tcache={}
    tcache['t']=t_pre(k,beta,cu,T,max_seqlen=S,_skip_check=True,_device=dev)
    tm=timeit(lambda: mn_pre(k,v,tcache['t'],alpha,cu,T,cp_chunk_len=cl,max_seqlen=S,_skip_check=True,_device=dev))
    print(f"  MN precompute   {tm:8.3f} ms", flush=True)
    lt, ls = mn_pre(k,v,tcache['t'],alpha,cu,T,cp_chunk_len=cl,max_seqlen=S,_skip_check=True,_device=dev)
    tf=timeit(lambda: fixup(lt,ls,cu,T,cp_chunk_len=cl,_skip_check=True,_device=dev))
    print(f"  fixup           {tf:8.3f} ms", flush=True)
    fs = fixup(lt,ls,cu,T,cp_chunk_len=cl,_skip_check=True,_device=dev)
    o=torch.empty_like(v); st=torch.empty(ns,Hv,D,D,device=dev,dtype=torch.float32)
    tp=timeit(lambda: pf(o,st,q,k,v,tcache['t'],fs,alpha,1.0/(D**0.5),cu,T,cp_chunk_len=cl,max_seqlen=S,_skip_check=True,_device=dev))
    print(f"  prefill         {tp:8.3f} ms", flush=True)
    print(f"  SUM parts       {tt+tm+tf+tp:8.3f} ms", flush=True)
    tfull_cp=timeit(lambda: chunk_gated_delta_rule(q,k,v,alpha,beta,cu_seqlens=cu,use_cp=True))
    tfull_nc=timeit(lambda: chunk_gated_delta_rule(q,k,v,alpha,beta,cu_seqlens=cu,use_cp=False))
    print(f"  FULL cp={tfull_cp:.3f}  noncp={tfull_nc:.3f}  ratio={tfull_nc/tfull_cp:.2f}", flush=True)
