import importlib,json,pathlib,torch,math
b=importlib.import_module('01_experiment')
from native_cp import NativeCP
from mixed_native import MixedNative
from cp_boundary import CPBoundary,MixedBoundary
from unified import Plan,cp
R=pathlib.Path(__file__).resolve().parents[1]
rows=[]
cases=[([512],[256]),([8192],[1280]),([8192],[4096]),([32768],[16384]),([8192,1024,1024,1024],[4096,512,512,512]),([4096,4096],[2048,2048]),([8192,8192],[4096,4096]),([1024]*8,[512]*8)]
for model,(hq,hv) in [('122b_tp4',(4,16)),('35b_tp1',(16,32))]:
 for lengths,ts in cases:
  x=b.inputs(lengths,hq,hv);b.unified.original(x);ref=b.snapshot(x);p=Plan.create(lengths,ts)
  methods={}
  for name,fn in [('full_selected',b.selected.run),('v2_selected',b.dvsplit.run)]:
   z=p.allocate(x);methods[name]=(lambda z=z,fn=fn:fn(**z),z,p.dense_ids)
  z=p.allocate(x);methods['native_cp_selected']=(NativeCP(z,p),z,p.dense_ids)
  oldchunk=math.gcd(2048,*ts)
  z=p.allocate(x);methods['old_fixup_extraction']=(CPBoundary(z,p,oldchunk),z,p.dense_ids)
  z=b.unified.no_checkpoint(b.copyout(x));methods['original_cp_no_checkpoint']=(lambda z=z:cp(z,max(lengths)),z,None)
  if len(lengths)>1:
   z=p.allocate(x);methods['mixed_native']=(MixedNative(z,p),z,p.dense_ids)
   z=p.allocate(x);methods['mixed_old_extraction']=(MixedBoundary(z,p,math.gcd(2048,ts[lengths.index(max(lengths))])),z,p.dense_ids)
  checks={}
  for name,(fn,z,ids) in methods.items():
   fn();torch.cuda.synchronize();rr=ref if ids is not None else {k:v for k,v in ref.items() if k!='state_checkpoints'}
   checks[name]=b.check(z,rr,lengths,ids)
   assert checks[name]['close'],(name,checks[name])
  times,orders=b.graph_times(methods,11,40,20261024)
  row=dict(model=model,lengths=lengths,targets=ts,native_chunk=2048,old_chunk=oldchunk,checks=checks,times=times,orders=orders)
  rows.append(row);(R/'results/benchmark.json').write_text(json.dumps(rows,indent=2));print(json.dumps(dict(model=model,lengths=lengths,targets=ts,times={k:v['median_ms'] for k,v in times.items()})),flush=True)
  del methods,x,ref,z;torch.cuda.empty_cache()
print('COMPLETE',len(rows),flush=True)
