"""Additional zero-initial-state, empty sequence and invalid-plan checks."""
import importlib.util,pathlib,json,torch
from unified import Plan
import unified,selected,dvsplit
from cp_boundary import CPBoundary,MixedBoundary
R=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('tests',R/'scripts/01_experiment.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
invalid=[([],[]),([65],[128]),([65],[-64]),([128],[65]),([128],[[64,128]]),([128],[]),([-1],[0])]
for args in invalid:
 try:Plan.create(*args)
 except ValueError:pass
 else:raise AssertionError(args)
rows=[]
for model,hq,hv in [('122b_tp4',4,16),('35b_tp1',16,32)]:
 for zero_init in [False,True]:
  lengths=[0,129,8193];targets=[0,64,8192];x=t.inputs(lengths,hq,hv)
  if zero_init:x['initial_state'].zero_()
  unified.original(x);torch.cuda.synchronize();ref=t.snapshot(x);plan=Plan.create(lengths,targets)
  for name,fn in [('full',selected.run),('dv',dvsplit.run),('cp_boundary',None),('mixed_boundary',None)]:
   z=plan.allocate(x)
   call=(lambda z=z,fn=fn:fn(**z)) if fn else CPBoundary(z,plan,64) if name=='cp_boundary' else MixedBoundary(z,plan,2048)
   call();torch.cuda.synchronize();result=t.check(z,ref,lengths,plan.dense_ids)
   assert torch.isnan(z['output_state'][0]).all(),'empty sequence state row overwritten'
   if name in ('full','dv'):assert result['exact']
   else:assert result['output']['finite'] and result['state_checkpoints']['finite']
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):call()
   graph.replay();torch.cuda.synchronize();graphcheck=t.check(z,ref,lengths,plan.dense_ids)
   if name in ('full','dv'):assert graphcheck['exact']
   row=dict(model=model,zero_init=zero_init,method=name,empty_row_untouched=True,check=result,graph=graphcheck)
   rows.append(row);t.save(R/'results/edge_memcheck.json',dict(invalid_plans_rejected=len(invalid),cases=rows))
   print(json.dumps(dict(model=model,zero_init=zero_init,method=name,exact=result['exact'],close=result['close'])),flush=True)
   del graph
print('COMPLETE_EDGE',len(rows),flush=True)
