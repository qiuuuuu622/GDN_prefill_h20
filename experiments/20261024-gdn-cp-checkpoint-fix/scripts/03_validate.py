"""Boundary semantics, dynamic CUDA graph targets, and checkpoint continuation."""
import argparse,importlib,json,pathlib,torch
from itertools import accumulate
b=importlib.import_module('01_experiment')
from native_cp import NativeCP
from unified import Plan,cp
R=pathlib.Path(__file__).resolve().parents[1]
def metric(a,b,tol=.01):
 return dict(exact=bool(torch.equal(a,b)),finite=bool(torch.isfinite(a).all()),max_abs=float((a.float()-b.float()).abs().max()) if a.numel() else 0,close=bool(torch.allclose(a,b,atol=tol,rtol=tol)))
def prefix_states(x,lengths,targets):
 states=[];off=0
 for i,(n,t) in enumerate(zip(lengths,targets)):
  if t:
   z={k:x[k][off:off+t].contiguous() for k in ('q','k','v','g','beta')}
   z.update(initial_state=x['initial_state'][i:i+1],output=torch.empty_like(x['output'][off:off+t]),output_state=torch.empty_like(x['output_state'][i:i+1]),cu_seqlens=torch.tensor([0,t],device='cuda',dtype=torch.int64))
   cp(z,t);states.append(z['output_state'].clone())
  off+=n
 return torch.cat(states) if states else torch.empty_like(x['state_checkpoints'][:0])
def run(args):
 rows=[]
 cases=[[0,129,7],[65,127,128,513,1025,2047,2207,2080],[8193],[4096,4096],[8192,1024,1024,1024],[1024]*8,[32768]]
 if args.memcheck:cases=cases[:2]
 for model,(hq,hv) in [('122b_tp4',(4,16)),('35b_tp1',(16,32))]:
  for gate in ['slow','near_one']:
   for init in ['random','zero']:
    for lengths in cases:
     x=b.inputs(lengths,hq,hv,gate)
     if init=='zero':x['initial_state'].zero_()
     initial=x['initial_state'].clone();b.unified.original(x);ref=b.snapshot(x)
     oc=b.unified.no_checkpoint(b.copyout(x));cp(oc,max(lengths));torch.cuda.synchronize()
     targets=[min(1280,n//64*64) if n>=64 and i%3!=1 else 0 for i,n in enumerate(lengths)]
     # Keep selected and unselected sequences together, but ensure >=1 target.
     if not any(targets):targets=[n//64*64 for n in lengths]
     p=Plan.create(lengths,targets);z=p.allocate(x);fn=NativeCP(z,p)
     fn();torch.cuda.synchronize()
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):fn()
     for phase,ts in [('eager',targets),('graph_changed',[n//64*64 if t else 0 for n,t in zip(lengths,targets)])]:
      pp=Plan.create(lengths,ts)
      if phase!='eager':
       fn.gpuplan.copy_(torch.tensor(pp.packed,device='cuda',dtype=torch.int64));fn.states.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
      pre=prefix_states(x,lengths,ts);torch.cuda.synchronize()
      live=[i for i,n in enumerate(lengths) if n]
      original_cp={'output':metric(z['output'],oc['output'],.02),'final':metric(z['output_state'][live],oc['output_state'][live])}
      boundary=metric(z['state_checkpoints'],pre)
      safe=bool(torch.equal(initial,x['initial_state'])) and all(bool(torch.isnan(z['output_state'][i]).all()) for i,n in enumerate(lengths) if not n)
      row=dict(model=model,lengths=lengths,gate=gate,initial=init,phase=phase,targets=ts,original_cp=original_cp,checkpoint_vs_cp_prefix=boundary,inputs_and_empty_unchanged=safe,vs_noncp=b.check(z,ref,lengths,pp.dense_ids),resume_vs_noncp=b.resume(x,z,ref,pp))
      row['boundary_pass']=boundary['exact'] and all(v['exact'] for v in original_cp.values()) and safe
      rows.append(row);(R/'results'/('memcheck.json' if args.memcheck else 'validation.json')).write_text(json.dumps(rows,indent=2))
      print(json.dumps({k:row[k] for k in ['model','lengths','gate','initial','phase','boundary_pass']}),flush=True)
      assert row['boundary_pass'],row
     del graph,x,oc,z,fn,ref;torch.cuda.empty_cache()
 print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--memcheck',action='store_true');run(p.parse_args())
