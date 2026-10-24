"""Shared per-sequence selected-checkpoint contract for Hq4/Hv16 and Hq16/Hv32.

No model-specific SGLang hook here. Reuses the measured full/V2 kernels unchanged.
Each sequence requests zero or one 64-token-aligned intermediate state. Multiple
requests per sequence are rejected rather than silently discarding a checkpoint.
"""
from dataclasses import dataclass
from itertools import accumulate
import torch
import selected, dvsplit
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_kernels import cp_delta_rule_dsl_sm90
from flashinfer import utils

@dataclass(frozen=True)
class Plan:
    lengths: tuple
    targets: tuple
    packed: tuple
    dense_ids: tuple

    @classmethod
    def create(cls, lengths, targets):
        lengths,targets=tuple(lengths),tuple(targets)
        if len(lengths)!=len(targets) or not lengths:raise ValueError('one target entry per sequence required')
        packed=[];ids=[];dense_start=0
        for n,target in zip(lengths,targets):
            if not isinstance(n,int) or n<0:raise ValueError('nonnegative integer sequence lengths required')
            if not isinstance(target,int) or target<0 or target>n or target%64:raise ValueError('target must be zero or an in-range 64-token boundary')
            packed.extend([len(ids) if target else 0,target])
            if target:ids.append(dense_start+target//64-1)
            dense_start+=n//64
        return cls(lengths,targets,tuple(packed),tuple(ids))

    def allocate(self, x):
        hv=x['v'].shape[1]
        if x['q'].shape[-1]!=128 or x['v'].shape[-1]!=128:raise ValueError('D128 only')
        if x['q'].shape[1]!=x['k'].shape[1] or hv%x['k'].shape[1]:raise ValueError('unsupported head grouping')
        return dict(x,output=torch.full_like(x['output'],float('nan')),output_state=torch.full_like(x['output_state'],float('nan')),
                    state_checkpoints=torch.full((len(self.dense_ids),hv,128,128),float('nan'),device=x['q'].device),
                    checkpoint_cu_starts=torch.tensor(self.packed,device=x['q'].device,dtype=torch.int64),checkpoint_every_n_tokens=-1)


def original(x):
    return chunk_gated_delta_rule(**x,output_final_state=True,use_cp=False)


def no_checkpoint(x):
    return dict(x,state_checkpoints=None,checkpoint_cu_starts=None,checkpoint_every_n_tokens=0)

# Retain workspace allocations because graph captures may outlive cache resizing.
RETAINED={}
def cp(x, max_seqlen, chunk=2048):
    cp_delta_rule_dsl_sm90(x['output'],x['output_state'],x['q'],x['k'],x['v'],x['g'],x['beta'],x['cu_seqlens'],128**-.5,initial_state=x['initial_state'],max_seqlen=max_seqlen,cp_chunk_len=chunk)
    for value in utils._cache_buf.values():
        if isinstance(value,torch.Tensor):RETAINED[value.data_ptr()]=value


class SegmentedCP:
    """Correct-state checkpoint boundary reference; serial CP calls, NOT native CP support.

    Split each original sequence at its requested checkpoint boundary, feed its
    completed state to the tail, and write the checkpoint directly as output_state.
    All views and cu_seqlens are prebuilt outside timing/capture.
    """
    def __init__(self,x,plan,chunk=2048):
        self.calls=[];self.final_copies=[];offset=0;slot=0
        for i,(n,target) in enumerate(zip(plan.lengths,plan.targets)):
            if not n:continue
            boundaries=sorted(set([0,target,n]));state=x['initial_state'][i:i+1]
            for start,end in zip(boundaries,boundaries[1:]):
                is_target=end==target and target>0
                state_out=x['state_checkpoints'][slot:slot+1] if is_target else x['output_state'][i:i+1]
                part={k:x[k][offset+start:offset+end] for k in ('q','k','v','g','beta')}
                part.update(cu_seqlens=torch.tensor([0,end-start],device=x['q'].device,dtype=torch.int64),initial_state=state,output=x['output'][offset+start:offset+end],output_state=state_out)
                self.calls.append((part,end-start,chunk));state=state_out
                if is_target:
                    slot+=1
                    if end==n:self.final_copies.append((x['output_state'][i:i+1],state_out))
            offset+=n
    def __call__(self):
        for x,n,chunk in self.calls:cp(x,n,chunk)
        for dst,src in self.final_copies:dst.copy_(src)
