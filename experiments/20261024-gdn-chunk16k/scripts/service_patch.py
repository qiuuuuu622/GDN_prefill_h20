"""Process-local multi-sequence selected checkpoint dispatch; no CP substitution."""
import json,os,pathlib,torch
import dvsplit,selected
from dispatch import method_for_lengths
R=pathlib.Path('/root/gdn35/20261024-gdn-chunk16k')
FIXED=os.environ.get('GDN35_VARIANT')
def mode():return FIXED or (R/'config/mode.txt').read_text().strip()
def append(name,row):
 with (R/'evidence'/name).open('a') as f:f.write(json.dumps(row)+'\n')
def compact_plan(lengths,mask,relative,interval):
 use=mask & (relative%interval!=0);targets=(relative//interval)*interval
 if bool(((targets<=0)|(targets>lengths))[use].any()):return None
 count=int(use.sum())
 if not count:return None
 indices=torch.cumsum(use.to(torch.int64),0)-1
 packed=torch.stack((torch.where(use,indices,0),torch.where(use,targets,0)),dim=1).flatten()
 return count,packed
def dense_reference(original,kw):
 cu=kw['cu_seqlens'];n=cu.numel()-1;counts=(cu[1:]-cu[:-1])//64
 starts=torch.cat((torch.zeros(1,device=cu.device,dtype=torch.int64),counts.cumsum(0)))
 count=int(starts[-1].item());shape=kw['state_checkpoints'].shape[1:]
 dense=dict(kw,checkpoint_every_n_tokens=64,checkpoint_cu_starts=starts,state_checkpoints=torch.empty((count,*shape),device=cu.device,dtype=kw['state_checkpoints'].dtype))
 result=original(**dense)
 packed=kw['checkpoint_cu_starts'].view(n,2);keep=packed[:,1]>0
 src=starts[:-1][keep]+packed[:,1][keep]//64-1;dst=packed[:,0][keep]
 kw['state_checkpoints'].index_copy_(0,dst,dense['state_checkpoints'].index_select(0,src))
 return result
def install(module):
 original_plan=module.maybe_build_flashinfer_checkpoint_plan
 def plan(batch,metadata,device):
  m=mode()
  if m not in ('multi','verify_multi'):return original_plan(batch,metadata,device)
  if metadata.track_ssm_h_src is None or metadata.track_ssm_h_src.numel()==0:return
  lengths=batch.extend_seq_lens.to(device='cpu',dtype=torch.int64)
  if not 1<=lengths.numel()<=8:return original_plan(batch,metadata,device)
  mask=batch.mamba_track_mask.to(device='cpu',dtype=torch.bool)
  relative=batch.mamba_track_seqlens.to(device='cpu',dtype=torch.int64)-batch.extend_prefix_lens.to(device='cpu',dtype=torch.int64)
  built=compact_plan(lengths,mask,relative,module.mamba_cache_chunk_size())
  if built is None:return original_plan(batch,metadata,device)
  count,packed=built
  assert count==metadata.track_ssm_h_dst.numel()
  metadata.track_ssm_h_src=torch.arange(count,device=device,dtype=torch.int64)
  metadata.state_checkpoint_cu_starts=packed.to(device,non_blocking=True)
  metadata.num_state_checkpoints=count
  metadata.state_checkpoint_every_n_tokens=-1 if method_for_lengths(tuple(lengths.tolist()))=='dv' else -2
  if m=='verify_multi':append('plans.jsonl',dict(lengths=lengths.tolist(),mask=mask.tolist(),relative=relative.tolist(),packed=packed.tolist(),count=count))
 module.maybe_build_flashinfer_checkpoint_plan=plan
 original_init=module.FlashInferGDNKernel.__init__
 def init(self,*args,**kwargs):
  original_init(self,*args,**kwargs);original=self._prefill_fn;captured=set()
  def adapted(**kw):
   m=mode();q,k,v=kw['q'],kw['k'],kw['v'];state=kw['initial_state'];cu=kw['cu_seqlens'];interval=kw.get('checkpoint_every_n_tokens',0);n=cu.numel()-1
   eligible=(1<=n<=8 and tuple(q.shape[1:])==(16,128) and tuple(k.shape[1:])==(16,128) and tuple(v.shape[1:])==(32,128) and q.dtype==k.dtype==v.dtype==torch.bfloat16 and state is not None and state.dtype==torch.float32 and state.is_contiguous() and kw.get('state_indices') is None and not kw.get('use_qk_l2norm_in_kernel',False) and kw.get('scale') is None and torch.cuda.get_device_properties(q.device).multi_processor_count==78)
   chosen=m in ('multi','verify_multi') and eligible and interval!=0
   method='dv' if interval==-1 or n==1 else 'full'
   if m=='verify_multi':append('routes.jsonl',dict(tokens=q.shape[0],sequences=n,checkpoint=interval,selected=chosen,method=method if chosen else 'original'))
   if not chosen:return dense_reference(original,kw) if interval<0 else original(**kw)
   keys=['q','k','v','g','beta','cu_seqlens','initial_state','output','output_state','state_checkpoints','checkpoint_cu_starts','checkpoint_every_n_tokens'];inputs={key:kw.get(key) for key in keys}
   if inputs['output'] is None:inputs['output']=torch.empty_like(v)
   if inputs['output_state'] is None:inputs['output_state']=torch.empty_like(state)
   (dvsplit.run if method=='dv' else selected.run)(**inputs)
   if m=='verify_multi':
    refs=dict(kw,output=torch.empty_like(inputs['output']),output_state=torch.empty_like(inputs['output_state']),state_checkpoints=torch.empty_like(inputs['state_checkpoints']))
    (dense_reference(original,refs) if interval<0 else original(**refs))
    errors={}
    for key in ['output','output_state','state_checkpoints']:
     torch.testing.assert_close(inputs[key],refs[key],atol=0,rtol=0)
     errors[key]=(inputs[key].float()-refs[key].float()).abs().max().item() if inputs[key].numel() else 0
    append('real_checks.jsonl',dict(tokens=q.shape[0],sequences=n,method=method,errors=errors,valid=True))
    shape=(n,q.shape[0])
    if n>1 and shape not in captured and len(captured)<6:
     captured.add(shape);torch.save({key:(value.detach().cpu() if isinstance(value,torch.Tensor) else value) for key,value in inputs.items()},R/'captures'/f'n{n}_t{q.shape[0]}.pt')
   return (inputs['output'],inputs['output_state']) if kw.get('output_final_state',False) else inputs['output']
  self._prefill_fn=adapted
 module.FlashInferGDNKernel.__init__=init
 print('GDN35_MULTI_CHECKPOINT_PATCH_INSTALLED',os.getpid(),flush=True)
