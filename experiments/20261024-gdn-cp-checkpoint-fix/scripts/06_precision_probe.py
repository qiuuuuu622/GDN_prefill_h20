"""Isolate fixup FP32 vs default; does not relax tolerances or change CP policy."""
import importlib,json,pathlib,torch
b=importlib.import_module('01_experiment');v=importlib.import_module('03_validate')
from native_cp import NativeCP
from unified import Plan
R=pathlib.Path(__file__).resolve().parents[1];rows=[]
for model,(hq,hv) in [('122b_tp4',(4,16)),('35b_tp1',(16,32))]:
 for seed in [20261024,20261025,20261026]:
  x=b.inputs([8192],hq,hv,'near_one',seed);b.unified.original(x);ref=b.snapshot(x)
  for target in [1280,4096]:
   p=Plan.create([8192],[target]);baseline=None
   for kind in ([None] if hv==16 else [None,'simt_row8']):
    z=p.allocate(x);fn=NativeCP(z,p,fixup_kind=kind);fn();torch.cuda.synchronize()
    row=dict(model=model,seed=seed,target=target,kind=kind or 'default',vs_noncp=b.check(z,ref,[8192],p.dense_ids),resume=b.resume(x,z,ref,p))
    if baseline is None:baseline=b.snapshot(z)
    else:row['vs_default']={k:v.metric(z[k],a,.02 if k=='output' else .01) for k,a in baseline.items()}
    rows.append(row);(R/'results/precision_probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
print('COMPLETE',len(rows),flush=True)
