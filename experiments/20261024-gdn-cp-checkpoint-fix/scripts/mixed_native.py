"""Experimental mixed CP native-checkpoint/full-V runner, includes copy costs."""
import torch
from itertools import accumulate
from unified import Plan
from native_cp import NativeCP
import selected
class MixedNative:
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
  self.long_dst=(z["output_state"],z["state_checkpoints"])
  self.long=NativeCP(z,p,chunk)
  for lo,hi in ((0,long),(long+1,len(plan.lengths))):
   if hi>lo and sum(plan.lengths[lo:hi]):self.short.append(group(lo,hi)[0])
 def __call__(self):
  current=torch.cuda.current_stream();self.side.wait_stream(current)
  with torch.cuda.stream(self.side):
   self.long()
   self.long_dst[0].copy_(self.long.x["output_state"])
   self.long_dst[1].copy_(self.long.x["state_checkpoints"])
  for x in self.short:selected.run(**x)
  current.wait_stream(self.side)
