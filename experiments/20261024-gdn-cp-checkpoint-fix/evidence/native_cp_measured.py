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
  assert tuple((x['cu_seqlens'][1:]-x['cu_seqlens'][:-1]).cpu().tolist())==plan.lengths
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
