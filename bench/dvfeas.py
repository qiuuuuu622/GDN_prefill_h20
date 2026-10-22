import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule
torch.manual_seed(0); dev="cuda"; Hq,Hv,D=4,16,128
def timeit(fn,iters=30,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    a,c=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    c.record(); torch.cuda.synchronize(); return a.elapsed_time(c)/iters
def mk(S,ns):
    T=ns*S
    q=torch.randn(T,Hq,D,device=dev,dtype=torch.bfloat16); k=torch.randn(T,Hq,D,device=dev,dtype=torch.bfloat16)
    v=torch.randn(T,Hv,D,device=dev,dtype=torch.bfloat16)
    al=torch.rand(T,Hv,device=dev,dtype=torch.float32).sigmoid().log()
    be=torch.rand(T,Hv,device=dev,dtype=torch.float32).sigmoid()
    cu=torch.tensor([i*S for i in range(ns+1)],device=dev,dtype=torch.int64)
    return q,k,v,al,be,cu

# per-chunk FLOP per (token,head) for D=128, chunk BLK=64
# O1 KV@Q 128x128x64 ; SK KV@K^T 128x64x128 ; NewV 128x64x64 ; O2 128x64x64 ; KV+= 128x64x128 ; QK 64x128x64
FLOP_STATE = 2*(128*64*128)*2 + 2*(128*64*64)*2   # O1+SK + NewV+O2 + KV+=  -> recompute below
per_chunk = 2*(128*64*128) + 2*(128*64*128) + 2*(128*64*64) + 2*(128*64*64) + 2*(128*64*128)
qk_chunk  = 2*(64*128*64)
tot_chunk = per_chunk + qk_chunk
print(f"per-chunk FLOP/head: state={per_chunk/1e6:.2f}M  qk(aux)={qk_chunk/1e6:.2f}M  total={tot_chunk/1e6:.2f}M")
print(f"aux fraction (does NOT scale with Dv) = {qk_chunk/tot_chunk*100:.1f}%")
print()
print(f"{'CTAs':>5}{'S':>7}{'chunks':>7}{'ms':>9}{'us/chunk':>10}{'TFLOPS/SM':>11}")
res=[]
for S in [1024,2048,4096,8192,16384,32768]:
    ns=1
    q,k,v,al,be,cu=mk(S,ns)
    t=timeit(lambda: chunk_gated_delta_rule(q,k,v,al,be,cu_seqlens=cu,use_cp=False),iters=20,warm=6)
    ctas=ns*Hv; chunks=S//64
    # each CTA = 1 seq x 1 head -> work per CTA = chunks * tot_chunk
    tflops = (chunks*tot_chunk)/ (t*1e-3) /1e12   # total across all CTAs
    per_sm = tflops/ctas
    res.append((S,t))
    print(f"{ctas:5d}{S:7d}{chunks:7d}{t:9.3f}{t*1000/chunks:10.2f}{per_sm:11.2f}")
print()
print("peak/SM bf16 H20 = 148/78 =", round(148/78,2), "TFLOPS")
print()
# linear fit on last points
import numpy as np
S=np.array([r[0] for r in res],dtype=float); T=np.array([r[1] for r in res])
A=np.vstack([S,np.ones_like(S)]).T
slope,icept=np.linalg.lstsq(A,T,rcond=None)[0]
print(f"fit: time = {slope*1e6:.4f} us per 1k tokens * S/1000 + {icept*1000:.1f} us intercept")
print(f"per-chunk slope = {slope*64*1000:.2f} us ; intercept/flight = {icept*1000:.1f} us")
print()
print("DV split model:")
for k_ in [2,4]:
    frac = qk_chunk/tot_chunk
    pred = icept + (S*slope)*(frac + (1-frac)/k_)
    print(f"  Dv/{k_}: CTAs 16->{16*k_}, predicted ms at S=8192 = {pred[np.argmin(abs(S-8192))]:.3f} (baseline {T[np.argmin(abs(S-8192))]:.3f})")
