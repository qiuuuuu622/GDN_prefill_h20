import importlib.util,pathlib,json,torch,math
from cp_boundary import CPBoundary,MixedBoundary
from unified import Plan,SegmentedCP
import selected,dvsplit,unified
R=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('tests',R/'scripts/01_experiment.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
rows=[]
cases=[([8192],[4096]),([8192],[1280]),([8192,1024,1024,1024],[4096,512,512,512]),([1024,8192,1024],[512,4096,512]),([4096,4096],[2048,2048]),([0,129,8193],[0,64,8192])]
for model,hq,hv in [('122b_tp4',4,16),('35b_tp1',16,32)]:
 for gate in ['slow','near_one']:
  for lengths,targets in cases:
   x=t.inputs(lengths,hq,hv,gate);unified.original(x);torch.cuda.synchronize();ref=t.snapshot(x);plan=Plan.create(lengths,targets)
   methods={};z=t.copyout(x);methods['original_dense']=(lambda z=z:unified.original(z),z,None)
   for name,fn in [('full_selected',selected.run),('dv_selected',dvsplit.run)]:
    z=plan.allocate(x);methods[name]=(lambda z=z,fn=fn:fn(**z),z,plan.dense_ids)
   chunk=math.gcd(2048,*[v for v in targets if v]);z=plan.allocate(x);methods['cp_boundary']=(CPBoundary(z,plan,chunk),z,plan.dense_ids)
   long=max(range(len(lengths)),key=lengths.__getitem__);longchunk=math.gcd(2048,targets[long]);z=plan.allocate(x);methods['mixed_boundary']=(MixedBoundary(z,plan,longchunk),z,plan.dense_ids)
   valid={};checks={};resumed={}
   for name,(fn,z,ids) in methods.items():
    fn();torch.cuda.synchronize();checks[name]=t.check(z,ref,lengths,ids)
    resumed[name]=t.resume(x,z,ref,plan) if ids is not None else []
    if name in ('cp_boundary','mixed_boundary'):
     # Check wrapping the original CP pipeline did not change token output/final state.
     if name=='cp_boundary':
      plain=unified.no_checkpoint(t.copyout(x));unified.cp(plain,max(lengths),chunk);torch.cuda.synchronize()
      checks[name]['same_cp_output_exact']=bool(torch.equal(plain['output'],z['output']))
      live=[i for i,n in enumerate(lengths) if n]
      checks[name]['same_cp_final_exact']=bool(torch.equal(plain['output_state'][live],z['output_state'][live]))
    if checks[name]['close'] and all(v['close'] for v in resumed[name]):valid[name]=(fn,z,ids)
   timing={};orders=[]
   if gate=='slow':timing,orders=t.graph_times(valid,7,50,20261024+len(rows))
   # Re-run after timing; graph buffers can share cached workspace allocations.
   post={}
   for name,(fn,z,ids) in valid.items():
    fn();torch.cuda.synchronize();post[name]=t.check(z,ref,lengths,ids)
    assert post[name]['close'],(name,post[name])
   row=dict(model=model,gate=gate,lengths=lengths,targets=targets,cp_chunk=chunk,mixed_cp_chunk=longchunk,checks=checks,resume=resumed,post=post,timing=timing,orders=orders)
   rows.append(row);t.save(R/'results/cp_boundary.json',rows)
   print(json.dumps(dict(model=model,gate=gate,lengths=lengths,targets=targets,cp_chunk=chunk,valid=list(valid),rejected=[k for k in methods if k not in valid],timings={k:v['median_ms'] for k,v in timing.items()})),flush=True)
   del x,ref,methods,valid;torch.cuda.empty_cache()
print('COMPLETE_BOUNDARY',len(rows),flush=True)
