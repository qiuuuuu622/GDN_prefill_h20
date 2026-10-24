import importlib,json,pathlib,torch
b=importlib.import_module('01_experiment');v=importlib.import_module('03_validate')
from native_cp import NativeCP
from unified import Plan,cp
R=pathlib.Path(__file__).resolve().parents[1];rows=[]
for model,(hq,hv) in [('122b_tp4',(4,16)),('35b_tp1',(16,32))]:
 for n in [129,8193]:
  for target in [0,64]:
   x=b.inputs([n],hq,hv);x['initial_state']=None;p=Plan.create([n],[target]);z=p.allocate(x);fn=NativeCP(z,p);fn();torch.cuda.synchronize()
   oc=b.unified.no_checkpoint(b.copyout(x));cp(oc,n);torch.cuda.synchronize()
   checks={k:v.metric(z[k],oc[k],.02 if k=='output' else .01) for k in ['output','output_state']}
   if target:
    pre={k:x[k][:target] for k in ['q','k','v','g','beta']};pre.update(initial_state=None,output=torch.empty_like(x['output'][:target]),output_state=torch.empty_like(x['output_state']),cu_seqlens=torch.tensor([0,target],device='cuda',dtype=torch.int64));cp(pre,target);torch.cuda.synchronize();checks['checkpoint']=v.metric(z['state_checkpoints'],pre['output_state'])
   row=dict(model=model,n=n,target=target,initial_none=True,checks=checks,empty_plan_rows=z['state_checkpoints'].shape[0]);row['pass']=all(a['exact'] for a in checks.values()) and z['state_checkpoints'].shape[0]==int(target>0);assert row['pass'],row;rows.append(row)
# Construction rejects unsupported inputs before launching kernels.
x=b.inputs([129],4,16);p=Plan.create([129],[64]);invalid=[]
for name,z,plan,chunk in [('bad_chunk',x,p,65),('wrong_dtype',dict(x,q=x['q'].float()),p,2048),('bad_lengths',x,Plan.create([128],[64]),2048),('bad_gate_shape',dict(x,g=x['g'][:1]),p,2048),('bad_cu_dtype',dict(x,cu_seqlens=x['cu_seqlens'].int()),p,2048),('bad_initial',dict(x,initial_state=x['initial_state'].bfloat16()),p,2048)]:
 try:NativeCP(z,plan,chunk)
 except ValueError as e:invalid.append(dict(case=name,rejected=True,error=str(e)))
 else:raise AssertionError(name)
(R/'results/edge_contract.json').write_text(json.dumps(dict(rows=rows,invalid=invalid),indent=2));print('COMPLETE',len(rows),'valid',len(invalid),'rejected',flush=True)
