import importlib,json,pathlib,torch
b=importlib.import_module('01_experiment')
from native_cp import NativeCP
from unified import Plan,cp
R=pathlib.Path(__file__).resolve().parents[1]
def metric(a,b,tol=.01):
 return dict(exact=bool(torch.equal(a,b)),finite=bool(torch.isfinite(a).all()),max_abs=float((a.float()-b.float()).abs().max()),close=bool(torch.allclose(a,b,atol=tol,rtol=tol)))
def prefix(x,t):
 z={k:x[k][:t].contiguous() for k in ('q','k','v','g','beta')}
 z.update(initial_state=x['initial_state'],output=torch.empty_like(x['output'][:t]),output_state=torch.empty_like(x['output_state']),cu_seqlens=torch.tensor([0,t],device='cuda',dtype=torch.int64))
 return z
rows=[]
for model,(hq,hv) in [('122b_tp4',(4,16)),('35b_tp1',(16,32))]:
 for gate in ['slow','near_one']:
  x=b.inputs([8192],hq,hv,gate);b.unified.original(x);torch.cuda.synchronize();ref=b.snapshot(x)
  oc=b.unified.no_checkpoint(b.copyout(x));cp(oc,8192);torch.cuda.synchronize()
  for target in [1280,4096]:
   p=Plan.create([8192],[target]);z=p.allocate(x);fn=NativeCP(z,p);fn();torch.cuda.synchronize()
   pre=prefix(x,target);cp(pre,target);torch.cuda.synchronize()
   row=dict(model=model,gate=gate,target=target,output_vs_cp=metric(z['output'],oc['output'],.02),final_vs_cp=metric(z['output_state'],oc['output_state']),checkpoint_vs_cp_prefix=metric(z['state_checkpoints'],pre['output_state']),vs_noncp=b.check(z,ref,[8192],p.dense_ids),resume=b.resume(x,z,ref,p))
   rows.append(row);(R/'results/pilot.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
print('COMPLETE',len(rows),flush=True)
