"""GPU differential checks for the packaged API; no model weights or service startup."""
import argparse,importlib.util,json,pathlib,sys
from itertools import accumulate
import torch
import torch.nn.functional as F
from gdn_prefill import GDNPrefill,Geometry,CheckpointRequest as Req
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm90 as cp
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('historical_split',ROOT/'flashinfer/gdn_split.py');legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)

def inputs(lengths,hq,hv,seed,initial=True):
 torch.manual_seed(seed);total=sum(lengths);state=torch.randn(len(lengths),hv,128,128,device='cuda')*.1
 return dict(q=F.normalize(torch.randn(total,hq,128,device='cuda'),dim=-1).bfloat16(),k=F.normalize(torch.randn(total,hq,128,device='cuda'),dim=-1).bfloat16(),v=torch.randn(total,hv,128,device='cuda',dtype=torch.bfloat16),g=torch.exp(-.05*F.softplus(torch.randn(total,hv,device='cuda'))),beta=torch.randn(total,hv,device='cuda').sigmoid(),initial_state=state if initial else None,output=torch.full((total,hv,128),float('nan'),device='cuda',dtype=torch.bfloat16),output_state=torch.full_like(state,float('nan')))

def cp_call(**kw):
 cp.cp_delta_rule_dsl_sm90(kw['output'],kw['output_state'],kw['q'],kw['k'],kw['v'],kw['g'],kw['beta'],kw['cu_seqlens'],128**-.5,initial_state=kw['initial_state'],max_seqlen=kw['max_seqlen'],cp_chunk_len=kw['cp_chunk_len'])

def reference(x,lengths,plan,requests):
 kw=dict(x,output=torch.full_like(x['output'],float('nan')),output_state=torch.full_like(x['output_state'],float('nan')))
 cu=torch.tensor((0,*accumulate(lengths)),device='cuda',dtype=torch.int64)
 dense=None
 if requests:
  starts=(0,*accumulate(n//64 for n in lengths));dense=torch.empty((starts[-1],x['v'].shape[1],128,128),device='cuda')
  chunk_gated_delta_rule(**kw,cu_seqlens=cu,state_checkpoints=dense,checkpoint_cu_starts=torch.tensor(starts,device='cuda',dtype=torch.int64),checkpoint_every_n_tokens=64,use_cp=False,output_final_state=True)
 elif plan.backend=='mixed_cp':
  def noncp(**args):
   args.pop('max_seqlen',None);chunk_gated_delta_rule(**args,use_cp=False,output_final_state=True)
  legacy.execute_split(legacy.SplitPlan(plan.spec.long_index,plan.spec.cp_chunk),lengths=lengths,**kw,cu_seqlens=cu,cp_fn=cp_call,noncp_fn=noncp)
 elif plan.backend=='cp':
  cp_call(**kw,cu_seqlens=cu,max_seqlen=max(lengths),cp_chunk_len=plan.spec.cp_chunk)
 else:chunk_gated_delta_rule(**kw,cu_seqlens=cu,use_cp='auto',output_final_state=True)
 return kw,dense

def metric(a,b,tol):
 return dict(exact=bool(torch.equal(a,b)),close=bool(torch.allclose(a,b,atol=tol,rtol=tol)),max_abs=float((a.float()-b.float()).abs().max()) if a.numel() else 0,finite=bool(torch.isfinite(a).all()))

def main(args):
 rows=[];op=GDNPrefill()
 cases=[([0,129,7],[Req(1,64,2)]),([8193],[Req(0,4096,0)]),([4096,4096],[Req(0,2048,1),Req(1,1280,0)]),([8192,1024,1024,1024],[Req(i,512 if i else 4096,i) for i in range(4)]),([1024]*8,[Req(i,512,i) for i in range(8)]),([129,513],[Req(0,64,2),Req(0,128,0),Req(1,512,4)]),([8192,1024,1024,1024],[]),([512],[])]
 if args.memcheck:cases=[cases[0],cases[2],cases[5],cases[6]]
 for hq,hv in [(4,16),(16,32)]:
  for initial in [True,False]:
   for lengths,requests in cases:
    x=inputs(lengths,hq,hv,20261024,initial);plan=op.prepare(lengths=lengths,geometry=Geometry(hq,hv),checkpoint_requests=requests)
    ck=torch.full((plan.spec.checkpoints.capacity,hv,128,128),float('nan'),device='cuda') if requests else None
    ref,dense=reference(x,lengths,plan,requests);torch.cuda.synchronize()
    op.run(plan=plan,**x,checkpoint_states=ck);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):op.run(plan=plan,**x,checkpoint_states=ck)
    for phase in ['eager','graph']:
     if phase=='graph':graph.replay();torch.cuda.synchronize()
     live=[i for i,n in enumerate(lengths) if n]
     checks={'output':metric(x['output'],ref['output'],.02),'final':metric(x['output_state'][live],ref['output_state'][live],.01)}
     if requests:
      checks['checkpoint']=metric(ck[list(plan.spec.checkpoints.slots)],dense[list(plan.spec.checkpoints.dense_ids)],.01)
      unused=[i for i in range(ck.shape[0]) if i not in plan.spec.checkpoints.slots];assert bool(torch.isnan(ck[unused]).all())
     assert all(v['exact'] and v['finite'] for v in checks.values()),checks
     assert all(bool(torch.isnan(x['output_state'][i]).all()) for i,n in enumerate(lengths) if not n)
     rows.append(dict(hq=hq,hv=hv,lengths=lengths,initial=initial,backend=plan.backend,phase=phase,checks=checks,pass_exact=True))
    if requests and plan.backend in ('v2','full'):
     changed=[Req(r.sequence,64,r.slot) for r in requests if r.sequence%2==0]
     ck.fill_(float('nan'));plan.update_checkpoints(changed);graph.replay();torch.cuda.synchronize()
     starts=(0,*accumulate(n//64 for n in lengths))
     for r in changed:assert torch.equal(ck[r.slot],dense[starts[r.sequence]]),r
     absent=[i for i in range(ck.shape[0]) if i not in [r.slot for r in changed]];assert bool(torch.isnan(ck[absent]).all())
     rows.append(dict(hq=hq,hv=hv,lengths=lengths,initial=initial,backend=plan.backend,phase='changed_metadata',pass_exact=True))
    args.out.write_text(json.dumps(rows,indent=2));print(hq,hv,lengths,initial,plan.backend,'PASS',len(rows),flush=True)
    del graph,plan,x,ref,dense,ck;torch.cuda.empty_cache()
 # Explicit experimental native CP: compare checkpoint with original CP prefixes.
 for hq,hv in [(4,16),(16,32)]:
  x=inputs([8192],hq,hv,20261024);op=GDNPrefill(policy='experimental');requests=[Req(0,1280,0)]
  plan=op.prepare(lengths=[8192],geometry=Geometry(hq,hv),checkpoint_requests=requests,backend='cp_checkpoint')
  ck=torch.full((1,hv,128,128),float('nan'),device='cuda');op.run(plan=plan,**x,checkpoint_states=ck);torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):op.run(plan=plan,**x,checkpoint_states=ck)
  for target in [1280,4096,0]:
   plan.update_checkpoints([Req(0,target,0)] if target else []);ck.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
   if target:
    pre={k:v[:target] for k,v in x.items() if k in ('q','k','v','g','beta','output')};pre['output']=torch.empty_like(pre['output']);pre.update(initial_state=x['initial_state'],output_state=torch.empty_like(x['output_state']))
    cp_call(**pre,cu_seqlens=torch.tensor([0,target],device='cuda',dtype=torch.int64),max_seqlen=target,cp_chunk_len=2048);torch.cuda.synchronize();assert torch.equal(ck,pre['output_state'])
   else:assert bool(torch.isnan(ck).all())
   rows.append(dict(hq=hq,hv=hv,backend='cp_checkpoint',target=target,phase='graph_checkpoint_vs_cp_prefix',pass_exact=True))
  args.out.write_text(json.dumps(rows,indent=2));del graph,plan,x,ck;torch.cuda.empty_cache()
 print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=pathlib.Path);p.add_argument('--memcheck',action='store_true');main(p.parse_args())
