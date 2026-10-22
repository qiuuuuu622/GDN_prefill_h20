import os, json, sys, torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule

torch.manual_seed(0)
dev="cuda"
# Qwen3.5-MoE-122B, TP4 per-rank GDN shape
Hq, Hv, D = 4, 16, 128

def make(S, ns, lens=None):
    lens = lens or [S]*ns
    T = sum(lens)
    q=torch.randn(T,Hq,D,device=dev,dtype=torch.bfloat16)
    k=torch.randn(T,Hq,D,device=dev,dtype=torch.bfloat16)
    v=torch.randn(T,Hv,D,device=dev,dtype=torch.bfloat16)
    g=torch.nn.functional.logsigmoid(torch.randn(T,Hv,device=dev,dtype=torch.float32))
    b=torch.rand(T,Hv,device=dev,dtype=torch.float32).sigmoid()
    cu=torch.tensor([0]+list(torch.cumsum(torch.tensor(lens),0).tolist()),device=dev,dtype=torch.int64)
    return q,k,v,g,b,cu

def timeit(fn,iters=30,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    a,c=torch.cuda.Event(True),torch.cuda.Event(True)
    a.record()
    for _ in range(iters): fn()
    c.record(); torch.cuda.synchronize()
    return a.elapsed_time(c)/iters

print(f"GPU={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count} Hq={Hq} Hv={Hv} D={D}")
rows=[]
print(f"{'case':>26} {'cp=auto':>9} {'cp=False':>9} {'cp=True':>9}  ratio")
for S in [1024,2048,4096,8192,16384]:
    for ns in [1,2,4,8]:
        q,k,v,g,b,cu=make(S,ns)
        res={}
        for cp in ["auto",False,True]:
            try:
                fn=lambda cp=cp: chunk_gated_delta_rule(q,k,v,g,b,cu_seqlens=cu,use_cp=cp)
                fn(); torch.cuda.synchronize()
                res[cp]=timeit(fn)
            except Exception as e:
                res[cp]=None
        f=lambda x: f"{x:9.3f}" if x else "     ERR "
        rat = (res[False]/res[True]) if (res[False] and res[True]) else None
        print(f"S={S:<6}d ns={ns:<3d} work={ns*Hv:<4d} {f(res['auto'])} {f(res[False])} {f(res[True])}  {rat and round(rat,2)}")
        rows.append(dict(S=S,ns=ns,work=ns*Hv,auto=res['auto'],noncp=res[False],cp=res[True]))
json.dump(rows,open("/root/gdn_tp4.json","w"),indent=1)
