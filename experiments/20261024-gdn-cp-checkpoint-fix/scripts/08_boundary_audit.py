import importlib,json,pathlib,torch
b=importlib.import_module('01_experiment');v=importlib.import_module('03_validate')
from native_cp import NativeCP
from cp_boundary import CPBoundary
from unified import Plan
R=pathlib.Path(__file__).resolve().parents[1];rows=[]
for model,(hq,hv) in [('122b_tp4',(4,16)),('35b_tp1',(16,32))]:
 for gate in ['slow','near_one']:
  x=b.inputs([8192],hq,hv,gate)
  for target in [2048,4096]:
   p=Plan.create([8192],[target]);old=p.allocate(x);new=p.allocate(x);oldfn=CPBoundary(old,p);newfn=NativeCP(new,p)
   oldfn();newfn();prefix=v.prefix_states(x,[8192],[target]);torch.cuda.synchronize()
   row=dict(model=model,gate=gate,target=target,cp_chunk=2048,old_vs_cp_prefix=v.metric(old['state_checkpoints'],prefix),native_vs_cp_prefix=v.metric(new['state_checkpoints'],prefix),native_vs_old=v.metric(new['state_checkpoints'],old['state_checkpoints']),output_unchanged=v.metric(new['output'],old['output'],.02),final_unchanged=v.metric(new['output_state'],old['output_state']))
   rows.append(row);assert row['native_vs_cp_prefix']['exact'] and row['output_unchanged']['exact'] and row['final_unchanged']['exact'];print(json.dumps(row),flush=True)
(R/'results/boundary_audit.json').write_text(json.dumps(rows,indent=2));print('COMPLETE',len(rows),flush=True)
