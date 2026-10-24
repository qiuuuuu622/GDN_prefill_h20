"""Separate extraction error from original CP-vs-nonCP numerical differences."""
import pathlib,importlib.util,json,torch
import unified
from unified import Plan
from cp_boundary import CPBoundary
R=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('tests',R/'scripts/01_experiment.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
rows=[]
for model,hq,hv in [('122b_tp4',4,16),('35b_tp1',16,32)]:
 for gate in ['slow','near_one']:
  for target,chunk in [(4096,2048),(1280,256)]:
   x=t.inputs([8192],hq,hv,gate);unified.original(x);torch.cuda.synchronize();ref=t.snapshot(x);plan=Plan.create([8192],[target]);z=plan.allocate(x);CPBoundary(z,plan,chunk)();torch.cuda.synchronize()
   saved=z['state_checkpoints'].clone()
   prefix={k:x[k][:target] for k in ['q','k','v','g','beta']}
   prefix.update(initial_state=x['initial_state'],output=torch.empty_like(x['output'][:target]),output_state=torch.empty_like(x['output_state']),cu_seqlens=torch.tensor([0,target],device='cuda',dtype=torch.int64))
   unified.cp(prefix,target,chunk);torch.cuda.synchronize()
   def cmp(a,b):
    d=(a-b).abs();return dict(exact=bool(torch.equal(a,b)),max_abs=d.max().item(),close_01=bool(torch.allclose(a,b,atol=.01,rtol=.01)))
   row=dict(model=model,gate=gate,target=target,chunk=chunk,fixed_vs_original_cp_prefix=cmp(saved,prefix['output_state']),original_cp_prefix_vs_dense=cmp(prefix['output_state'],ref['state_checkpoints'][list(plan.dense_ids)]),fixed_vs_dense=cmp(saved,ref['state_checkpoints'][list(plan.dense_ids)]))
   rows.append(row);t.save(R/'results/cp_prefix_checks.json',rows);print(json.dumps(row),flush=True)
print('COMPLETE_PREFIX',len(rows),flush=True)
