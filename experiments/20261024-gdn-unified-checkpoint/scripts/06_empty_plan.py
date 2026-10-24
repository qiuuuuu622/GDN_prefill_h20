import pathlib,importlib.util,torch,json
from unified import Plan
import unified,selected,dvsplit
R=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('tests',R/'scripts/01_experiment.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
rows=[]
for model,hq,hv in [('122b_tp4',4,16),('35b_tp1',16,32)]:
 lengths=[0,129,513];x=unified.no_checkpoint(t.inputs(lengths,hq,hv));unified.original(x);torch.cuda.synchronize();ref=t.snapshot(x)
 for name,fn in [('full',selected.run),('dv',dvsplit.run)]:
  plan=Plan.create(lengths,[0,0,0]);z=plan.allocate(x);assert z['checkpoint_every_n_tokens']==0 and z['state_checkpoints'] is None
  fn(**z);torch.cuda.synchronize();a=t.check(z,ref,lengths)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):fn(**z)
  graph.replay();torch.cuda.synchronize();b=t.check(z,ref,lengths)
  assert a['exact'] and b['exact']
  rows.append(dict(model=model,method=name,empty_plan_no_checkpoint=True,eager=a,graph=b))
(R/'results/empty_plan.json').write_text(json.dumps(rows,indent=2)+'\n');print('COMPLETE_EMPTY_PLAN',len(rows))
