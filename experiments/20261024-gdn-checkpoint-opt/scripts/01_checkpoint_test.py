"""Checkpoint DV split: exact baseline parity, checkpoint restart, graph/eager timing."""
import argparse, json, pathlib, statistics, time
import torch
import torch.nn.functional as F
from flashinfer.gdn_prefill import chunk_gated_delta_rule
import dvsplit

ROOT=pathlib.Path(__file__).resolve().parents[1]

def inputs(lengths, interval=64, seed=19, gate='slow', init=True):
    torch.manual_seed(seed)
    t=sum(lengths); hq,hv,d=16,32,128
    q=F.normalize(torch.randn(t,hq,d,device='cuda'),dim=-1).bfloat16()
    k=F.normalize(torch.randn_like(q,dtype=torch.float32),dim=-1).bfloat16()
    v=torch.randn(t,hv,d,device='cuda',dtype=torch.bfloat16)
    g=torch.full((t,hv),.999,device='cuda') if gate=='near_one' else torch.exp(-.05*F.softplus(torch.randn(t,hv,device='cuda')))
    beta=torch.randn(t,hv,device='cuda').sigmoid()
    h0=.1*torch.randn(len(lengths),hv,d,d,device='cuda') if init else torch.zeros(len(lengths),hv,d,d,device='cuda')
    cu=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int64)
    counts=[n//interval for n in lengths] if interval else [0]*len(lengths)
    cps=torch.tensor([0]+list(__import__('itertools').accumulate(counts)),device='cuda',dtype=torch.int64) if interval else None
    return dict(q=q,k=k,v=v,g=g,beta=beta,initial_state=h0,cu_seqlens=cu,
        output=torch.full_like(v,float('nan')),output_state=torch.full_like(h0,float('nan')),
        state_checkpoints=torch.full((sum(counts),hv,d,d),float('nan'),device='cuda') if interval else None,
        checkpoint_cu_starts=cps,checkpoint_every_n_tokens=interval)

def baseline(x):
    return chunk_gated_delta_rule(**x,output_final_state=True,use_cp='auto')

def snapshot(x):
    return {k:x[k].clone() for k in ['output','output_state','state_checkpoints'] if x[k] is not None}

def compare(x,ref,lengths):
    row={}
    for k,b in ref.items():
        a=x[k]
        if k=='output_state':
            live=[i for i,n in enumerate(lengths) if n];a=a[live];b=b[live]
        err=(a.float()-b.float()).abs()
        row[k]=dict(exact=bool(torch.equal(a,b)),finite=bool(torch.isfinite(a).all()),max_abs=err.max().item() if a.numel() else 0)
    row['valid']=all(v['exact'] and v['finite'] for v in row.values())
    return row

def timing(fn,n=30):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    values=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):g.replay()
        b.record();b.synchronize();values.append(a.elapsed_time(b)/n)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start=time.perf_counter();a.record()
    for _ in range(n):fn()
    b.record();b.synchronize()
    return dict(graph_median_ms=statistics.median(values),graph_samples_ms=values,eager_event_ms=a.elapsed_time(b)/n,eager_wall_ms=(time.perf_counter()-start)*1000/n)

def restart_check(x,lengths):
    # Exact chunk-boundary continuation; no changed recurrence grouping.
    if len(lengths)!=1 or lengths[0]<=64:return None
    cut=64; n=lengths[0]
    z={k:(v[cut:].contiguous() if k in ['q','k','v','g','beta'] else v) for k,v in x.items()}
    z.update(initial_state=x['state_checkpoints'][0:1].clone(),cu_seqlens=torch.tensor([0,n-cut],device='cuda',dtype=torch.int64),
        output=torch.empty_like(x['output'][cut:]),output_state=torch.empty_like(x['output_state']),
        state_checkpoints=None,checkpoint_cu_starts=None,checkpoint_every_n_tokens=0)
    dvsplit.run(**z);torch.cuda.synchronize()
    return dict(output_exact=bool(torch.equal(z['output'],x['output'][cut:])),state_exact=bool(torch.equal(z['output_state'],x['output_state'])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',default='pilot');a=p.parse_args()
    cases=[[65],[512],[8192]] if a.suite=='pilot' else [[1],[63],[64],[65],[127],[128],[129],[257],[2239],[3230],[8191],[8192],[8193],[0,129,7],[1024,512],[8192,8192]]
    rows=[]
    for gate in (['slow'] if a.suite=='pilot' else ['slow','near_one']):
      for lengths in cases:
        x=inputs(lengths,gate=gate);baseline(x);torch.cuda.synchronize();ref=snapshot(x)
        dvsplit.run(**x);torch.cuda.synchronize()
        row=dict(lengths=lengths,gate=gate,check=compare(x,ref,lengths),restart=restart_check(x,lengths))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):dvsplit.run(**x)
        graph.replay();torch.cuda.synchronize();row['graph_check']=compare(x,ref,lengths)
        row['valid']=row['check']['valid'] and row['graph_check']['valid'] and (row['restart'] is None or all(row['restart'].values()))
        if gate=='slow' and lengths in [[512],[8192],[2239],[3230],[8192,8192]]:
            row['baseline']=timing(lambda:baseline(x));row['dvsplit']=timing(lambda:dvsplit.run(**x))
            row['speedup']=row['baseline']['graph_median_ms']/row['dvsplit']['graph_median_ms']
        rows.append(row);print(json.dumps(row),flush=True)
        (ROOT/'results'/f'checkpoint_{a.suite}.json').write_text(json.dumps(rows,indent=2))
        assert row['valid'],row
        del graph,x,ref
        torch.cuda.empty_cache()
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
