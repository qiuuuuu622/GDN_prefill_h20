import argparse,json,pathlib,random,statistics,time,hashlib,importlib.metadata,inspect,traceback
from itertools import accumulate
import torch
import torch.nn.functional as F
import selected,dvsplit,unified
from unified import Plan,SegmentedCP
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer import utils
import original_gdn_split as split
R=pathlib.Path(__file__).resolve().parents[1]

def inputs(lengths,hq,hv,gate='slow',seed=20261024):
 torch.manual_seed(seed);n=sum(lengths)
 q=F.normalize(torch.randn(n,hq,128,device='cuda'),dim=-1).bfloat16();k=F.normalize(torch.randn(n,hq,128,device='cuda'),dim=-1).bfloat16()
 v=torch.randn(n,hv,128,device='cuda',dtype=torch.bfloat16)
 g=torch.full((n,hv),.999,device='cuda') if gate=='near_one' else torch.exp(-.05*F.softplus(torch.randn(n,hv,device='cuda')))
 h=.1*torch.randn(len(lengths),hv,128,128,device='cuda')
 return dict(q=q,k=k,v=v,g=g,beta=torch.randn(n,hv,device='cuda').sigmoid(),initial_state=h,cu_seqlens=torch.tensor([0]+list(accumulate(lengths)),device='cuda',dtype=torch.int64),output=torch.full_like(v,float('nan')),output_state=torch.full_like(h,float('nan')),state_checkpoints=torch.full((sum(n//64 for n in lengths),hv,128,128),float('nan'),device='cuda'),checkpoint_cu_starts=torch.tensor([0]+list(accumulate(n//64 for n in lengths)),device='cuda',dtype=torch.int64),checkpoint_every_n_tokens=64)

def copyout(x):return dict(x,**{k:torch.full_like(x[k],float('nan')) for k in ('output','output_state','state_checkpoints') if x[k] is not None})
def snapshot(x):return {k:x[k].clone() for k in ('output','output_state','state_checkpoints') if x[k] is not None}
def check(x,ref,lengths,ids=None):
 result={}
 for key,b in ref.items():
  a=x[key]
  if key=='output_state':
   live=[i for i,n in enumerate(lengths) if n];a=a[live];b=b[live]
  if key=='state_checkpoints' and ids is not None:b=b[list(ids)]
  diff=(a.float()-b.float()).abs();tol=.02 if key=='output' else .01
  result[key]=dict(exact=bool(torch.equal(a,b)),finite=bool(torch.isfinite(a).all()),max_abs=diff.max().item() if diff.numel() else 0,tolerance=tol,close=bool(torch.allclose(a,b,atol=tol,rtol=tol)))
 result['exact']=all(v['exact'] and v['finite'] for v in result.values());result['close']=all(v['close'] and v['finite'] for v in result.values() if isinstance(v,dict))
 return result

def resume(x,z,ref,plan):
 pos=0;slot=0;checks=[]
 for i,(n,t) in enumerate(zip(plan.lengths,plan.targets)):
  if t:
   if t<n:
    tail={k:x[k][pos+t:pos+n].contiguous() for k in ('q','k','v','g','beta')}
    tail.update(initial_state=z['state_checkpoints'][slot:slot+1],cu_seqlens=torch.tensor([0,n-t],device='cuda',dtype=torch.int64),output=torch.empty_like(x['output'][pos+t:pos+n]),output_state=torch.empty_like(x['output_state'][i:i+1]),state_checkpoints=None,checkpoint_cu_starts=None,checkpoint_every_n_tokens=0)
    selected.run(**tail);torch.cuda.synchronize()
    rr={'output':ref['output'][pos+t:pos+n],'output_state':ref['output_state'][i:i+1]}
    checks.append(check(tail,rr,[n-t]))
   slot+=1
  pos+=n
 return checks

def checkpoint_methods(x,plan,include_cp=True):
 methods={};dense=copyout(x)
 methods['original_dense']=(lambda z=dense:unified.original(z),dense,None)
 for name,fn in [('full',selected.run),('dv',dvsplit.run)]:
  dense=copyout(x);compact=plan.allocate(x)
  methods[name+'_dense']=(lambda fn=fn,z=dense:fn(**z),dense,None)
  methods[name+'_selected']=(lambda fn=fn,z=compact:fn(**z),compact,plan.dense_ids)
 if include_cp:
  compact=plan.allocate(x);methods['segmented_cp_selected']=(SegmentedCP(compact,plan),compact,plan.dense_ids)
 return methods

def graph_times(methods,repeats,replays,seed):
 graphs={};checks={};samples={};randomizer=random.Random(seed)
 for name,(fn,z,ids) in methods.items():
  for _ in range(3):fn()
  torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):fn()
  graph.replay();torch.cuda.synchronize();graphs[name]=graph;samples[name]=[]
 # Full interleaved order is recorded; all candidate outputs and workspace refs retained.
 orders=[]
 for _ in range(repeats):
  order=list(graphs);randomizer.shuffle(order);orders.append(order)
  for name in order:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(replays):graphs[name].replay()
   b.record();b.synchronize();samples[name].append(a.elapsed_time(b)/replays)
 return {k:dict(median_ms=statistics.median(v),samples_ms=v) for k,v in samples.items()},orders

def save(path,rows):path.write_text(json.dumps(rows,indent=2)+'\n')
def targets(lengths,mixed=False):return [max(64,n//2//64*64) if n>=64 and (not mixed or i%2==1) else 0 for i,n in enumerate(lengths)]

def correctness(args):
 cases=[[65,129],[0,129,7],[63,64,65,127],[8193],[4096,4096],[128,8064],[2048]*4,[1024]*8]
 if args.pilot:cases=[[65,129],[8192]]
 if args.memcheck:cases=[[0,129,7],[65,127,128,513,1025,2047,2207,2080]]
 rows=[]
 for model,(hq,hv) in [('122b_tp4',(4,16)),('35b_tp1',(16,32))]:
  for gate in (['slow'] if args.pilot or args.memcheck else ['slow','near_one']):
   for lengths in cases:
    x=inputs(lengths,hq,hv,gate);unified.original(x);torch.cuda.synchronize();ref=snapshot(x)
    for mixed in ([False] if args.pilot else [False,True]):
     ts=targets(lengths,mixed)
     if not any(ts):continue
     plan=Plan.create(lengths,ts)
     methods=checkpoint_methods(x,plan,include_cp=not args.memcheck)
     for name,(fn,z,ids) in methods.items():
      if name=='original_dense':continue
      fn();torch.cuda.synchronize();eager=check(z,ref,lengths,ids)
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):fn()
      graph.replay();torch.cuda.synchronize();graphcheck=check(z,ref,lengths,ids)
      resumed=resume(x,z,ref,plan) if ids is not None else []
      changed=None
      if name in ('full_selected','dv_selected'):
       last=[n//64*64 if t else 0 for n,t in zip(lengths,ts)];newplan=Plan.create(lengths,last)
       z['checkpoint_cu_starts'].copy_(torch.tensor(newplan.packed,device='cuda',dtype=torch.int64));z['state_checkpoints'].fill_(float('nan'));graph.replay();torch.cuda.synchronize();changed=check(z,ref,lengths,newplan.dense_ids)
      exact=eager['exact'] and graphcheck['exact'] and all(v['exact'] for v in resumed) and (changed is None or changed['exact'])
      close=eager['close'] and graphcheck['close'] and all(v['close'] for v in resumed)
      row=dict(model=model,lengths=lengths,gate=gate,targets=ts,method=name,eager=eager,graph=graphcheck,resume=resumed,changed_targets=changed,exact=exact,close=close)
      rows.append(row);save(R/'results'/('memcheck.json' if args.memcheck else 'pilot.json' if args.pilot else 'correctness.json'),rows)
      print(json.dumps({k:row[k] for k in ('model','lengths','gate','method','exact','close')}),flush=True)
      if name!='segmented_cp_selected':assert exact,row
      del graph
    del x,ref,methods;torch.cuda.empty_cache()
 print('COMPLETE_CORRECTNESS',len(rows),flush=True)

def benchmark(args):
 cases=[[512],[8192],[16384],[32768],[8192,1024,1024,1024],[4096,4096],[8192,8192],[1024]*8]
 if args.pilot:cases=[[8192],[8192,1024,1024,1024]]
 rows=[]
 for model,(hq,hv) in [('122b_tp4',(4,16)),('35b_tp1',(16,32))]:
  for lengths in cases:
   x=inputs(lengths,hq,hv);unified.original(x);torch.cuda.synchronize();ref=snapshot(x);plan=Plan.create(lengths,targets(lengths))
   methods=checkpoint_methods(x,plan)
   # Non-checkpoint comparisons share inputs but belong to a separate workload contract.
   plain=unified.no_checkpoint(x);plainref={k:v for k,v in ref.items() if k!='state_checkpoints'}
   for name,fn in [('nocp_original',lambda z:unified.original(z)),('nocp_dv',lambda z:dvsplit.run(**z)),('no_checkpoint_auto',lambda z:chunk_gated_delta_rule(**z,output_final_state=True,use_cp='auto')),('no_checkpoint_cp',lambda z:unified.cp(z,max(lengths)))]:
    z=copyout(plain);methods[name]=(lambda fn=fn,z=z:fn(z),z,None)
   p=split.plan_split(lengths,hv,78)
   if p:
    z=copyout(plain)
    def cp_adapter(max_seqlen,cp_chunk_len=None,**kw):return unified.cp(kw,max_seqlen,cp_chunk_len or 2048)
    def noncp_adapter(max_seqlen=None,**kw):return selected.run(**kw)
    methods['no_checkpoint_mixed_cp']=(lambda z=z,p=p:split.execute_split(p,lengths=lengths,**{k:v for k,v in z.items() if k not in ('state_checkpoints','checkpoint_cu_starts','checkpoint_every_n_tokens')},cp_fn=cp_adapter,noncp_fn=noncp_adapter),z,None)
   valid={};checks={}
   for name,(fn,z,ids) in methods.items():
    try:
     fn();torch.cuda.synchronize();result=check(z,plainref if z['state_checkpoints'] is None else ref,lengths,ids)
     checks[name]=result
     if (result['close'] if name in ('segmented_cp_selected','no_checkpoint_cp','no_checkpoint_auto','no_checkpoint_mixed_cp') else result['exact']):valid[name]=(fn,z,ids)
    except Exception as e:checks[name]={'error':repr(e)}
   timing,orders=graph_times(valid,args.repeats,args.replays,20261024+len(rows))
   # Validate again after repeated randomized graph replay to catch lifetime/data aliasing issues.
   post={}
   for name,(fn,z,ids) in valid.items():
    post[name]=check(z,plainref if z['state_checkpoints'] is None else ref,lengths,ids)
    assert post[name]['close'],(name,post[name])
   row=dict(model=model,hq=hq,hv=hv,lengths=lengths,targets=list(plan.targets),mixed_cp_plan=vars(p) if p else None,correctness=checks,post_timing=post,timing=timing,orders=orders,dense_checkpoint_bytes=x['state_checkpoints'].numel()*4,selected_checkpoint_bytes=len(plan.dense_ids)*hv*128*128*4)
   rows.append(row);save(R/'results'/('benchmark_pilot.json' if args.pilot else 'benchmark.json'),rows)
   print(json.dumps(dict(model=model,lengths=lengths,timings={k:v['median_ms'] for k,v in timing.items()},rejected=[k for k in methods if k not in valid])),flush=True)
   del x,ref,methods,valid;torch.cuda.empty_cache()
 print('COMPLETE_BENCHMARK',len(rows),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['correctness','benchmark'],default='correctness');p.add_argument('--pilot',action='store_true');p.add_argument('--memcheck',action='store_true');p.add_argument('--repeats',type=int,default=7);p.add_argument('--replays',type=int,default=50);args=p.parse_args()
 save(R/'evidence'/f'environment_{args.mode}.json',dict(args=vars(args),versions={p:importlib.metadata.version(p) for p in ['torch','flashinfer-python','sglang','nvidia-cutlass-dsl']},gpu=torch.cuda.get_device_name(),sms=torch.cuda.get_device_properties(0).multi_processor_count,source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'scripts').glob('*.py')}))
 (correctness if args.mode=='correctness' else benchmark)(args)
