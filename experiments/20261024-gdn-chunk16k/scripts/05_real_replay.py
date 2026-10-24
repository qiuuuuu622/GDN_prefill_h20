"""Replay real multi-sequence service inputs against dense original and both kernels."""
import pathlib,json,importlib.util,torch,itertools
import dvsplit,selected
from dispatch import method_for_lengths
R=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('test',R/'scripts/01_checkpoint_test.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
rows=[]
for path in sorted((R/'captures').glob('*.pt')):
 saved=torch.load(path,weights_only=True);x={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in saved.items()}
 lengths=(x['cu_seqlens'][1:]-x['cu_seqlens'][:-1]).tolist();counts=[n//64 for n in lengths];starts=[0]+list(itertools.accumulate(counts))
 dense=dict(x,checkpoint_every_n_tokens=64,checkpoint_cu_starts=torch.tensor(starts,device='cuda',dtype=torch.int64),state_checkpoints=torch.empty((starts[-1],32,128,128),device='cuda'),output=torch.empty_like(x['output']),output_state=torch.empty_like(x['output_state']))
 t.baseline(dense);torch.cuda.synchronize()
 packed=x['checkpoint_cu_starts'].view(-1,2);keep=packed[:,1]>0;src=dense['checkpoint_cu_starts'][:-1][keep]+packed[:,1][keep]//64-1
 reference=dict(output=dense['output'].clone(),output_state=dense['output_state'].clone(),state_checkpoints=dense['state_checkpoints'][src].clone())
 for key in reference:torch.testing.assert_close(x[key],reference[key],atol=0,rtol=0)
 row=dict(file=path.name,lengths=lengths,method=method_for_lengths(tuple(lengths)),valid=True,baseline=t.timing(lambda:t.baseline(dense),n=20))
 for name,fn in [('full',selected.run),('dv',dvsplit.run)]:
  z=dict(x,**{key:torch.empty_like(x[key]) for key in reference});fn(**z);torch.cuda.synchronize()
  for key in reference:torch.testing.assert_close(z[key],reference[key],atol=0,rtol=0)
  row[name]=t.timing(lambda:fn(**z),n=20)
 row['speedup']=row['baseline']['graph_median_ms']/row[row['method']]['graph_median_ms']
 row['choice_vs_best']=row[row['method']]['graph_median_ms']/min(row['full']['graph_median_ms'],row['dv']['graph_median_ms'])
 rows.append(row);print(json.dumps(row),flush=True);(R/'results/real_replay.json').write_text(json.dumps(rows,indent=2))
 del saved,x,z,dense,reference;torch.cuda.empty_cache()
assert rows,'No real multi-sequence fixtures captured'
assert all(x['speedup']>=.97 for x in rows),rows
print('COMPLETE_REAL_REPLAY',len(rows),flush=True)
