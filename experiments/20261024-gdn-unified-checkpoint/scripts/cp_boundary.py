"""Experimental checkpoint extraction from corrected CP boundary workspaces.

Does not modify CP arithmetic. Only supports targets aligned with cp_chunk_len.
This is a tolerance-validated candidate, never an exact replacement by default.
"""
import math
from itertools import accumulate
import torch
import selected
from unified import Plan,RETAINED
from flashinfer import utils
from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm90 as c

class CPBoundary:
 def __init__(self,x,plan,chunk=2048):
  if any(t and t%chunk for t in plan.targets):raise ValueError('checkpoint is not a CP boundary')
  self.x=x;self.plan=plan;self.chunk=chunk;offset=0;ids=[]
  for i,(n,target) in enumerate(zip(plan.lengths,plan.targets)):
   if target:
    m=min(i,offset);base=m+(offset-m)//chunk
    ids.append(base+target//chunk-1)
   offset+=n
  self.ids=torch.tensor(ids,device=x['q'].device,dtype=torch.int64)
  self.last_fixed=None
 def __call__(self):
  x=self.x;cu=x['cu_seqlens'];total=x['q'].shape[0];n=max(self.plan.lengths)
  t=c.cp_delta_rule_t_precompute_dsl_sm90(x['k'],x['beta'],cu,total,max_seqlen=n,_skip_check=True)
  m,local=c.cp_delta_rule_mn_precompute_dsl_sm90(x['k'],x['v'],t,x['g'],cu,total,cp_chunk_len=self.chunk,max_seqlen=n,_skip_check=True)
  fixed=c.cp_delta_rule_fixup_dsl_sm90(m,local,cu,total,cp_chunk_len=self.chunk,initial_state=x['initial_state'],_skip_check=True)
  c.cp_delta_rule_prefill_dsl_sm90(x['output'],x['output_state'],x['q'],x['k'],x['v'],t,fixed,x['g'],128**-.5,cu,total,cp_chunk_len=self.chunk,max_seqlen=n,initial_state=x['initial_state'],_skip_check=True)
  torch.index_select(fixed,0,self.ids,out=x['state_checkpoints'])
  self.last_fixed=fixed
  for value in utils._cache_buf.values():
   if isinstance(value,torch.Tensor):RETAINED[value.data_ptr()]=value

class MixedBoundary:
 """Longest CP with boundary extraction, rest selected full-V, parallel streams."""
 def __init__(self,x,plan,chunk=2048):
  self.side=torch.cuda.Stream(device=x['q'].device);self.short=[]
  long=max(range(len(plan.lengths)),key=plan.lengths.__getitem__)
  offsets=[0]+list(accumulate(plan.lengths));slots=[0]+list(accumulate(int(t>0) for t in plan.targets))
  def group(lo,hi):
   a,b=offsets[lo],offsets[hi];lengths=plan.lengths[lo:hi];sub=Plan.create(lengths,plan.targets[lo:hi])
   z={k:x[k][a:b] for k in ('q','k','v','g','beta','output')}
   z.update(initial_state=x['initial_state'][lo:hi],output_state=x['output_state'][lo:hi],cu_seqlens=torch.tensor([0]+list(accumulate(lengths)),device=x['q'].device,dtype=torch.int64),state_checkpoints=x['state_checkpoints'][slots[lo]:slots[hi]],checkpoint_cu_starts=torch.tensor(sub.packed,device=x['q'].device,dtype=torch.int64),checkpoint_every_n_tokens=-1)
   if not sub.dense_ids:z.update(state_checkpoints=None,checkpoint_cu_starts=None,checkpoint_every_n_tokens=0)
   return z,sub
  z,p=group(long,long+1)
  if not p.dense_ids:raise ValueError('reference requires a checkpoint on the CP sequence')
  self.long=CPBoundary(z,p,chunk)
  for lo,hi in ((0,long),(long+1,len(plan.lengths))):
   if hi>lo and sum(plan.lengths[lo:hi]):self.short.append(group(lo,hi)[0])
 def __call__(self):
  current=torch.cuda.current_stream();self.side.wait_stream(current)
  with torch.cuda.stream(self.side):self.long()
  for x in self.short:selected.run(**x)
  current.wait_stream(self.side)
