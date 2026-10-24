"""Multi-sequence selected-state parity, metadata replay, resume and timing."""
import argparse,importlib.util,json,pathlib,torch,itertools
import dvsplit,selected
R=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('test',R/'scripts/01_checkpoint_test.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
def compact(x,lengths,targets):
 packed=[];dense_ids=[];destination=0;dense_start=0
 for n,target in zip(lengths,targets):
  assert target==0 or (0<target<=n and target%64==0)
  packed += [destination if target else 0,target]
  if target:dense_ids.append(dense_start+target//64-1);destination+=1
  dense_start+=n//64
 assert destination>0
 z=dict(x,output=torch.full_like(x['output'],float('nan')),output_state=torch.full_like(x['output_state'],float('nan')),state_checkpoints=torch.full((destination,32,128,128),float('nan'),device='cuda'),checkpoint_cu_starts=torch.tensor(packed,device='cuda',dtype=torch.int64),checkpoint_every_n_tokens=-1)
 return z,dense_ids
def verify(z,ref,lengths,ids):
 torch.testing.assert_close(z['output'],ref['output'],atol=0,rtol=0)
 live=[i for i,n in enumerate(lengths) if n]
 torch.testing.assert_close(z['output_state'][live],ref['output_state'][live],atol=0,rtol=0)
 torch.testing.assert_close(z['state_checkpoints'],ref['state_checkpoints'][ids],atol=0,rtol=0)
 assert torch.isfinite(z['output']).all() and torch.isfinite(z['state_checkpoints']).all()
def resume(x,z,ref,lengths,targets):
 pos=0;dest=0;count=0
 for i,(n,target) in enumerate(zip(lengths,targets)):
  if target:
   if target<n:
    tail={k:(v[pos+target:pos+n].contiguous() if k in ['q','k','v','g','beta'] else v) for k,v in x.items()}
    tail.update(initial_state=z['state_checkpoints'][dest:dest+1].clone(),cu_seqlens=torch.tensor([0,n-target],device='cuda',dtype=torch.int64),output=torch.empty_like(x['output'][pos+target:pos+n]),output_state=torch.empty_like(x['output_state'][i:i+1]),state_checkpoints=None,checkpoint_cu_starts=None,checkpoint_every_n_tokens=0)
    selected.run(**tail);torch.cuda.synchronize()
    torch.testing.assert_close(tail['output'],ref['output'][pos+target:pos+n],atol=0,rtol=0)
    torch.testing.assert_close(tail['output_state'],ref['output_state'][i:i+1],atol=0,rtol=0);count+=1
   dest+=1
  pos+=n
 return count
def main():
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--memcheck',action='store_true');args=p.parse_args()
 cases=[[65,129],[4096,4096],[128,8064]] if args.pilot else [[65,129],[63,65,129],[0,129,8193],[4096,4096],[128,8064],[8192,8192],[2048]*4,[1024]*8,[65,127,128,513,1025,2047,2207,2080]]
 if args.memcheck:cases=[[0,129,7,65],[65,127,128,513,1025,2047,2207,2080]]
 rows=[];gates=['slow'] if args.pilot or args.memcheck else ['slow','near_one']
 for gate in gates:
  for lengths in cases:
   x=t.inputs(lengths,gate=gate,init=True);t.baseline(x);torch.cuda.synchronize();ref=t.snapshot(x)
   for pattern in ['all','mixed']:
    targets=[max(64,(n//2//64)*64) if n>=64 and (pattern=='all' or i%2==1) else 0 for i,n in enumerate(lengths)]
    if not any(targets):continue
    for name,fn in [('full',selected.run),('dv',dvsplit.run)]:
     z,ids=compact(x,lengths,targets);fn(**z);torch.cuda.synchronize();verify(z,ref,lengths,ids)
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):fn(**z)
     graph.replay();torch.cuda.synchronize();verify(z,ref,lengths,ids)
     resumed=resume(x,z,ref,lengths,targets)
     # Replay same graph with changed per-sequence targets at same tensor address.
     last=[(n//64)*64 if target else 0 for n,target in zip(lengths,targets)]
     changed,lastids=compact(x,lengths,last);z['checkpoint_cu_starts'].copy_(changed['checkpoint_cu_starts']);z['state_checkpoints'].fill_(float('nan'))
     graph.replay();torch.cuda.synchronize();verify(z,ref,lengths,lastids)
     row=dict(lengths=lengths,gate=gate,pattern=pattern,method=name,targets=targets,graph_changed_targets=last,resumed_sequences=resumed,valid=True)
     if gate=='slow' and pattern=='all' and not args.memcheck:
      row['timing']=t.timing(lambda:fn(**z),n=15)
      row['baseline']=t.timing(lambda:t.baseline(x),n=15)
      row['speedup']=row['baseline']['graph_median_ms']/row['timing']['graph_median_ms']
     rows.append(row);print(json.dumps(row),flush=True)
     (R/'results'/('multi_memcheck.json' if args.memcheck else 'multi_pilot.json' if args.pilot else 'multi_checks.json')).write_text(json.dumps(rows,indent=2))
     del graph,z,changed
   del x,ref;torch.cuda.empty_cache()
 print('COMPLETE_MULTI',len(rows),flush=True)
if __name__=='__main__':main()
