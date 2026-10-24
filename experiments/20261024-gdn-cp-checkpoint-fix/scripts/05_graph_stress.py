import importlib,json,pathlib,random,torch
b=importlib.import_module('01_experiment');v=importlib.import_module('03_validate')
from native_cp import NativeCP
from unified import Plan,cp
R=pathlib.Path(__file__).resolve().parents[1];rows=[];rng=random.Random(20261024)
for model,(hq,hv) in [('122b_tp4',(4,16)),('35b_tp1',(16,32))]:
 for gate in ['slow','near_one']:
  for lengths in [[4096]*4,[0,129,8193,1024]]:
   x=b.inputs(lengths,hq,hv,gate);p=Plan.create(lengths,[n//64*64 for n in lengths]);z=p.allocate(x);fn=NativeCP(z,p);fn();torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):fn()
   oc=b.unified.no_checkpoint(b.copyout(x))
   for step in range(25):
    if step%5==0:x['initial_state'].normal_(0,.1)
    ts=[rng.randrange(n//64+1)*64 if n>=64 and rng.random()>.4 else 0 for n in lengths]
    if step%6==0:ts=[0]*len(lengths)
    pp=Plan.create(lengths,ts);cp(oc,max(lengths));fn.gpuplan.copy_(torch.tensor(pp.packed,device='cuda',dtype=torch.int64));fn.states.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
    pre=v.prefix_states(x,lengths,ts);live=[i for i,n in enumerate(lengths) if n];nc=len(pp.dense_ids)
    row=dict(model=model,gate=gate,lengths=lengths,step=step,targets=ts,checkpoint=v.metric(z['state_checkpoints'][:nc],pre),output=v.metric(z['output'],oc['output'],.02),final=v.metric(z['output_state'][live],oc['output_state'][live]),unused_untouched=bool(torch.isnan(z['state_checkpoints'][nc:]).all()))
    row['pass']=all(row[k]['exact'] for k in ['checkpoint','output','final']) and row['unused_untouched']
    rows.append(row);assert row['pass'],row
   (R/'results/graph_stress.json').write_text(json.dumps(rows,indent=2));print(model,gate,lengths,'PASS',len(rows),flush=True)
   del graph,fn,z,x,oc;torch.cuda.empty_cache()
print('COMPLETE',len(rows),flush=True)
