import torch
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm90 import (
    cp_delta_rule_t_precompute_dsl_sm90 as t_pre,
    cp_delta_rule_mn_precompute_dsl_sm90 as mn_pre,
    cp_delta_rule_fixup_dsl_sm90 as fixup,
    cp_delta_rule_prefill_dsl_sm90 as pf)
from flashinfer.gdn_prefill import chunk_gated_delta_rule
torch.manual_seed(0); dev="cuda"; Hq,Hv,D=4,16,128; SMS=78
def timeit(fn,iters=30,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    a,c=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    c.record(); torch.cuda.synchronize(); return a.elapsed_time(c)/iters
def run_cp(q,k,v,alpha,beta,cu,T,S,ns,C):
    tc=t_pre(k,beta,cu,T,max_seqlen=S,_skip_check=True,_device=dev)
    lt,ls=mn_pre(k,v,tc,alpha,cu,T,cp_chunk_len=C,max_seqlen=S,_skip_check=True,_device=dev)
    fs=fixup(lt,ls,cu,T,cp_chunk_len=C,_skip_check=True,_device=dev)
    o=torch.empty_like(v); st=torch.empty(ns,Hv,D,D,device=dev,dtype=torch.float32)
    def f():
        tc_=t_pre(k,beta,cu,T,max_seqlen=S,_skip_check=True,_device=dev)
        lt_,ls_=mn_pre(k,v,tc_,alpha,cu,T,cp_chunk_len=C,max_seqlen=S,_skip_check=True,_device=dev)
        fs_=fixup(lt_,ls_,cu,T,cp_chunk_len=C,_skip_check=True,_device=dev)
        pf(o,st,q,k,v,tc_,fs_,alpha,1.0/(D**0.5),cu,T,cp_chunk_len=C,max_seqlen=S,_skip_check=True,_device=dev)
    return timeit(f)
print(f"{'ns':>3}{'S':>7}{'C':>7}{'chunks':>7}{'CTAs':>7}{'waves':>6}{'ms':>9}")
for ns,S in [(1,8192),(2,8192),(4,8192),(1,16384),(1,4096)]:
    T=ns*S
    q=torch.randn(T,Hq,D,device=dev,dtype=torch.bfloat16); k=torch.randn(T,Hq,D,device=dev,dtype=torch.bfloat16)
    v=torch.randn(T,Hv,D,device=dev,dtype=torch.bfloat16)
    alpha=torch.rand(T,Hv,device=dev,dtype=torch.float32).sigmoid().log()
    beta=torch.rand(T,Hv,device=dev,dtype=torch.float32).sigmoid()
    cu=torch.tensor([i*S for i in range(ns+1)],device=dev,dtype=torch.int64)
    base=timeit(lambda: chunk_gated_delta_rule(q,k,v,alpha,beta,cu_seqlens=cu,use_cp=False))
    print(f"{ns:3d}{S:7d}{'nonCP':>7}{'-':>7}{ns*Hv:7d}{-(-ns*Hv//SMS):6d}{base:9.3f}")
    for C in [512,1024,2048,4096]:
        if C>S: continue
        try:
            nch=-(-S//C); ctas=ns*Hv*nch
            t=run_cp(q,k,v,alpha,beta,cu,T,S,ns,C)
            print(f"{ns:3d}{S:7d}{C:7d}{nch:7d}{ctas:7d}{-(-ctas//SMS):6d}{t:9.3f}")
        except Exception as e:
            print(f"{ns:3d}{S:7d}{C:7d} ERR {type(e).__name__} {str(e)[:60]}")
