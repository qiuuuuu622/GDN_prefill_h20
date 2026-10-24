"""Compare compact mapping with original planner; fallback and mutable graph inputs."""
import ast,__future__,types,pathlib,json,importlib.util,torch
import service_patch as patch,selected,dvsplit
R=pathlib.Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('test',R/'scripts/02_multi_test.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);t=m.t
tree=ast.parse((R/'evidence/gdn_flashinfer.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='maybe_build_flashinfer_checkpoint_plan')
env={'torch':torch,'mamba_cache_chunk_size':lambda:64};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original planner>','exec',flags=__future__.annotations.compiler_flag),env)
rows=[]
for lengths in [[65,129],[0,129,7,65],[64,128,256,513],[65,127,128,513,1025,2047,2207,2080]]:
 for bits in range(1,1<<len(lengths)):
  mask=torch.tensor([bool(bits&(1<<i)) and n>=65 for i,n in enumerate(lengths)])
  relative=torch.tensor(lengths);ls=torch.tensor(lengths);use=mask&(relative%64!=0)
  if not use.any():continue
  prefix=torch.arange(len(lengths))*64;batch=types.SimpleNamespace(extend_seq_lens=ls,mamba_track_mask=mask,mamba_track_seqlens=relative+prefix,extend_prefix_lens=prefix)
  meta=types.SimpleNamespace(track_ssm_h_src=torch.zeros(int(use.sum()),dtype=torch.int64),track_ssm_h_dst=torch.zeros(int(use.sum()),dtype=torch.int64))
  env['maybe_build_flashinfer_checkpoint_plan'](batch,meta,'cpu')
  count,packed=patch.compact_plan(ls,mask,relative,64);packed=packed.view(-1,2);keep=packed[:,1]>0
  restored=meta.state_checkpoint_cu_starts[:-1][keep]+packed[:,1][keep]//64-1
  assert torch.equal(restored,meta.track_ssm_h_src)
  assert torch.equal(packed[:,0][keep],torch.arange(count))
  rows.append(dict(lengths=lengths,mask_bits=bits,valid=True))
gpu=[]
for lengths,targets in [([65,129],[64,128]),([0,129,7,65],[0,64,0,64]),([65,127,128,513,1025,2047,2207,2080],[64,0,128,0,1024,0,2176,0])]:
 x=t.inputs(lengths,init=False);t.baseline(x);torch.cuda.synchronize();ref=t.snapshot(x);z,ids=m.compact(x,lengths,targets)
 patch.dense_reference(lambda **kw:t.baseline(kw),z);torch.cuda.synchronize();m.verify(z,ref,lengths,ids)
 for method in [selected.run,dvsplit.run]:method(**z);torch.cuda.synchronize();m.verify(z,ref,lengths,ids)
 gpu.append(dict(kind='zero_init_and_fallback',lengths=lengths,valid=True))
 del x,z,ref
for name,method in [('full',selected.run),('dv',dvsplit.run)]:
 x=t.inputs([4096,4096]);z,ids=m.compact(x,[4096,4096],[64,2048]);method(**z);torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):method(**z)
 for lengths,targets in [([4096,4096],[64,2048]),([128,8064],[64,8000]),([8064,128],[4032,64])]:
  cu=torch.tensor([0,lengths[0],8192],device='cuda',dtype=torch.int64);x['cu_seqlens'].copy_(cu)
  counts=[n//64 for n in lengths];x['checkpoint_cu_starts'].copy_(torch.tensor([0,counts[0],sum(counts)],device='cuda',dtype=torch.int64))
  t.baseline(x);ref=t.snapshot(x);changed,ids=m.compact(x,lengths,targets)
  z['checkpoint_cu_starts'].copy_(changed['checkpoint_cu_starts']);graph.replay();torch.cuda.synchronize();m.verify(z,ref,lengths,ids)
  gpu.append(dict(kind='graph_changed_sequence_lengths',method=name,lengths=lengths,valid=True))
 del graph,x,z,ref,changed;torch.cuda.empty_cache()
result=dict(planner_mapping_cases=len(rows),planner_all_valid=all(x['valid'] for x in rows),gpu_cases=gpu)
(R/'results/planner_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
