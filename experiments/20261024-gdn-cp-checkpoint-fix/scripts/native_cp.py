"""Isolated native CP checkpoint adapter; leaves installed FlashInfer unchanged."""
import importlib.util,pathlib,sys
import torch
from flashinfer import utils
from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm90 as c
from unified import RETAINED
name='flashinfer.gdn_kernels.delta_rule_dsl.local_cp_selected_sm90'
spec=importlib.util.spec_from_file_location(name,pathlib.Path(__file__).with_name('cp_selected_kernel.py'))
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
class NativeCP:
 def __init__(self,x,plan,chunk=2048,fixup_kind=None):
  if chunk <= 0 or chunk % 64:
   raise ValueError('CP chunk must be a positive multiple of 64')
  if not sum(plan.lengths):
   raise ValueError('All-empty batches are handled by the scheduler, not this adapter')
  if tuple((x['cu_seqlens'][1:]-x['cu_seqlens'][:-1]).cpu().tolist()) != plan.lengths:
   raise ValueError('Plan lengths must match cu_seqlens')
  if x['cu_seqlens'].dtype != torch.int64 or int(x['cu_seqlens'][0]) != 0 or sum(plan.lengths) != x['q'].shape[0]:
   raise ValueError('Invalid cu_seqlens or token count')
  hq=x['q'].shape[1];hv=x['v'].shape[1];total=sum(plan.lengths)
  if (hq,hv) not in ((4,16),(16,32)):
   raise ValueError('This experiment validates only Hq4/Hv16 and Hq16/Hv32')
  shapes={'q':(total,hq,128),'k':(total,hq,128),'v':(total,hv,128),'output':(total,hv,128),'g':(total,hv),'beta':(total,hv)}
  for key,shape in shapes.items():
   value=x[key];dtype=torch.bfloat16 if key in ('q','k','v','output') else torch.float32
   if tuple(value.shape)!=shape or value.dtype!=dtype or not value.is_contiguous() or value.device!=x['q'].device or not value.is_cuda:
    raise ValueError(f'Invalid {key} shape, dtype, layout, or device')
  initial=x['initial_state']
  if initial is not None and (tuple(initial.shape)!=(len(plan.lengths),hv,128,128) or initial.dtype!=torch.float32 or not initial.is_contiguous() or initial.device!=x['q'].device):
   raise ValueError('Invalid initial state')
  self.x=x;self.plan=plan;self.chunk=chunk;self.fixup_kind=fixup_kind
  n=len(plan.lengths);hv=x['v'].shape[1]
  self.states=torch.full((n+len(plan.dense_ids),hv,128,128),float('nan'),device=x['q'].device)
  x['output_state']=self.states[:n];x['state_checkpoints']=self.states[n:]
  self.gpuplan=torch.tensor(plan.packed,device=x['q'].device,dtype=torch.int64)
 def __call__(self):
  x=self.x;cu=x['cu_seqlens'];total=x['q'].shape[0];n=max(self.plan.lengths)
  t=c.cp_delta_rule_t_precompute_dsl_sm90(x['k'],x['beta'],cu,total,max_seqlen=n,_skip_check=True)
  m,local=c.cp_delta_rule_mn_precompute_dsl_sm90(x['k'],x['v'],t,x['g'],cu,total,cp_chunk_len=self.chunk,max_seqlen=n,_skip_check=True)
  kw={} if self.fixup_kind is None else {'_kernel_kind':self.fixup_kind}
  fixed=c.cp_delta_rule_fixup_dsl_sm90(m,local,cu,total,cp_chunk_len=self.chunk,initial_state=x['initial_state'],_skip_check=True,**kw)
  module.cp_delta_rule_prefill_selected_sm90(x['output'],self.states,x['q'],x['k'],x['v'],t,fixed,x['g'],128**-.5,cu,total,cp_chunk_len=self.chunk,max_seqlen=n,initial_state=x['initial_state'],checkpoint_plan=self.gpuplan,_skip_check=True)
  for value in utils._cache_buf.values():
   if isinstance(value,torch.Tensor):RETAINED[value.data_ptr()]=value
