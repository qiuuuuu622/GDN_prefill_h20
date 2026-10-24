"""Whole public-entry CUDA Graph device timing; excludes host prepare and dispatch."""
import argparse,json,pathlib,random,statistics
import torch
from gdn_prefill import GDNPrefill,Geometry,CheckpointRequest as Req
from validate_entry import inputs,reference,metric

def main(args):
 rows=[];rng=random.Random(20261024)
 for hq,hv in [(4,16),(16,32)]:
  for lengths in [[8192],[4096,4096],[8192,1024,1024,1024],[1024]*8]:
   reqs=[Req(i,min(1280,n//2//64*64),i) for i,n in enumerate(lengths)]
   x=inputs(lengths,hq,hv,20261024);op=GDNPrefill(policy='experimental');methods={};graphs={};checks={};samples={}
   for backend in ['original','full','v2','auto','cp_checkpoint']:
    plan=op.prepare(lengths=lengths,geometry=Geometry(hq,hv),checkpoint_requests=reqs,backend=backend)
    ck=torch.empty((len(reqs),hv,128,128),device='cuda');methods[backend]=(plan,ck)
    if backend=='original':ref,dense=reference(x,lengths,plan,reqs)
    for _ in range(3):op.run(plan=plan,**x,checkpoint_states=ck)
    torch.cuda.synchronize()
    check=dict(output=metric(x['output'],ref['output'],.02),final=metric(x['output_state'],ref['output_state'],.01),checkpoint=metric(ck,dense[list(plan.spec.checkpoints.dense_ids)],.01))
    assert all(v['close'] for v in check.values()),(backend,check)
    if backend!='cp_checkpoint':assert all(v['exact'] for v in check.values()),(backend,check)
    checks[backend]=check
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):op.run(plan=plan,**x,checkpoint_states=ck)
    graph.replay();torch.cuda.synchronize();graphs[backend]=graph;samples[backend]=[]
   orders=[]
   for _ in range(11):
    order=list(graphs);rng.shuffle(order);orders.append(order)
    for backend in order:
     a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
     for _ in range(40):graphs[backend].replay()
     b.record();b.synchronize();samples[backend].append(a.elapsed_time(b)/40)
   post={}
   for backend,graph in graphs.items():
    graph.replay();torch.cuda.synchronize();plan,ck=methods[backend]
    post[backend]=dict(output=metric(x['output'],ref['output'],.02),final=metric(x['output_state'],ref['output_state'],.01),checkpoint=metric(ck,dense[list(plan.spec.checkpoints.dense_ids)],.01))
    assert all(v['close'] for v in post[backend].values()),post[backend]
   row=dict(hq=hq,hv=hv,lengths=lengths,targets=[r.token for r in reqs],auto=methods['auto'][0].backend,timing={k:dict(median_ms=statistics.median(v),samples_ms=v) for k,v in samples.items()},orders=orders,checks=checks,post_timing=post)
   rows.append(row);args.out.write_text(json.dumps(rows,indent=2));print(hq,hv,lengths,{k:round(v['median_ms'],5) for k,v in row['timing'].items()},flush=True)
   del graphs,methods,graph,plan,x,ck,ref,dense;torch.cuda.empty_cache()
 print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);main(p.parse_args())
