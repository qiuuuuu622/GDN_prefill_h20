"""Validate dispatch ranking across balanced/unbalanced sequence distributions."""
import importlib.util,json,pathlib,torch
import dvsplit,selected
from dispatch import method_for_lengths
R=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('test',R/'scripts/02_multi_test.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);t=m.t
cases=[[8192],[64,8128],[256,7936],[1024,7168],[2048,6144],[3072,5120],[4096,4096],[6144,2048],[7168,1024],[256,256,7680],[1024,1024,6144],[2048,2048,4096],[2730,2731,2731],[2048]*4,[512]*3+[6656],[1024]*8,[64]*7+[7744],[256]*7+[6400],[64]*8]
rows=[]
for lengths in cases:
 x=t.inputs(lengths);t.baseline(x);torch.cuda.synchronize();ref=t.snapshot(x)
 targets=[n//64*64 for n in lengths];z,ids=m.compact(x,lengths,targets)
 row=dict(lengths=lengths,choice=method_for_lengths(tuple(lengths)))
 row['baseline']=t.timing(lambda:t.baseline(x),n=20)
 for name,fn in [('full',selected.run),('dv',dvsplit.run)]:
  fn(**z);torch.cuda.synchronize();m.verify(z,ref,lengths,ids);row[name]=t.timing(lambda:fn(**z),n=20)
 row['speedup']=row['baseline']['graph_median_ms']/row[row['choice']]['graph_median_ms']
 row['choice_vs_best']=row[row['choice']]['graph_median_ms']/min(row['full']['graph_median_ms'],row['dv']['graph_median_ms'])
 rows.append(row);print(json.dumps(row),flush=True);(R/'results/dispatch_sweep.json').write_text(json.dumps(rows,indent=2))
 del x,z,ref;torch.cuda.empty_cache()
print('COMPLETE_SWEEP',len(rows),flush=True)
