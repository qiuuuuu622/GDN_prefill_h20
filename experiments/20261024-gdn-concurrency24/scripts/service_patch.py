"""Experiment-local TP1 GDN dispatch and guarded single-sequence checkpoint plan."""
import collections, json, os, pathlib
import torch
import dvsplit
ROOT=pathlib.Path('/root/gdn35/20261024-gdn-concurrency24')

FIXED_MODE=os.environ.get('GDN35_VARIANT')
def mode():return FIXED_MODE or (ROOT/'config/mode.txt').read_text().strip()
def append(name,row):
    with (ROOT/'evidence'/name).open('a') as f:f.write(json.dumps(row)+'\n')

def dense_reference(original,kw,compact_output=False):
    # Reconstruct the public dense API and gather the selected snapshot.
    assert kw['cu_seqlens'].numel()==2
    n=kw['q'].shape[0];target=-kw['checkpoint_every_n_tokens'];count=n//64
    dense=dict(kw,state_checkpoints=torch.empty((count,*kw['initial_state'].shape[1:]),device=kw['q'].device,dtype=kw['initial_state'].dtype),checkpoint_cu_starts=torch.tensor([0,count],device=kw['q'].device,dtype=torch.int64),checkpoint_every_n_tokens=64)
    result=original(**dense)
    kw['state_checkpoints'][0].copy_(dense['state_checkpoints'][target//64-1])
    return result

def install(module):
    original_plan=module.maybe_build_flashinfer_checkpoint_plan
    def plan(forward_batch, forward_metadata, device):
        m=mode()
        if m not in ('sparse','verify_sparse'):
            original_plan(forward_batch,forward_metadata,device)
        else:
            # The ordinary metadata planner uses CPU tensors here too. Optimize
            # only one tracked unaligned sequence requiring one selected state.
            if forward_metadata.track_ssm_h_src is None or forward_metadata.track_ssm_h_src.numel()==0:return
            interval=module.mamba_cache_chunk_size()
            lengths=forward_batch.extend_seq_lens.to(device='cpu',dtype=torch.int64)
            mask=forward_batch.mamba_track_mask.to(device='cpu',dtype=torch.bool)
            relative=forward_batch.mamba_track_seqlens.to(device='cpu',dtype=torch.int64)-forward_batch.extend_prefix_lens.to(device='cpu',dtype=torch.int64)
            target=int(relative[0]//interval)*interval if lengths.numel()==1 else 0
            compact=(lengths.numel()==1 and bool(mask[0]) and int(relative[0])%interval!=0 and target>0 and target<=int(lengths[0]))
            if not compact:
                original_plan(forward_batch,forward_metadata,device)
            else:
                # Save exactly the same boundary the baseline consumer selects.
                forward_metadata.track_ssm_h_src=torch.zeros(1,device=device,dtype=torch.int64)
                forward_metadata.state_checkpoint_cu_starts=torch.tensor([0,1],device=device,dtype=torch.int64)
                forward_metadata.num_state_checkpoints=1
                forward_metadata.state_checkpoint_every_n_tokens=-target
        if m in ('audit','verify','verify_sparse'):
            md=forward_metadata
            append('plans.jsonl',dict(mode=m,lengths=forward_batch.extend_seq_lens.tolist(),prefix=forward_batch.extend_prefix_lens.tolist(),track_lens=forward_batch.mamba_track_seqlens.tolist(),
                count=md.num_state_checkpoints,interval=md.state_checkpoint_every_n_tokens,selected=None if md.track_ssm_h_src is None else md.track_ssm_h_src.tolist()))
    module.maybe_build_flashinfer_checkpoint_plan=plan
    original_init=module.FlashInferGDNKernel.__init__
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs);original=self._prefill_fn
        counts=collections.Counter();captured=set();last=None
        def adapted(**kw):
            nonlocal last
            m=mode();q,k,v=kw['q'],kw['k'],kw['v'];state=kw['initial_state'];cu=kw['cu_seqlens'];interval=kw.get('checkpoint_every_n_tokens',0)
            eligible=(cu.numel()==2 and tuple(q.shape[1:])==(16,128) and tuple(k.shape[1:])==(16,128) and tuple(v.shape[1:])==(32,128) and q.dtype==torch.bfloat16 and v.dtype==torch.bfloat16 and state is not None and state.dtype==torch.float32 and state.is_contiguous() and kw.get('state_indices') is None and not kw.get('use_qk_l2norm_in_kernel',False) and kw.get('scale') is None and torch.cuda.get_device_properties(q.device).multi_processor_count==78)
            selected=m in ('optimized','sparse','verify','verify_sparse') and eligible and interval!=0
            if FIXED_MODE is None:counts[(m,selected)]+=1
            if FIXED_MODE is None and (m!=last or counts[(m,selected)]%30==0):
                print('GDN35_ROUTE',json.dumps(dict(mode=m,selected=selected,count=counts[(m,selected)],tokens=q.shape[0],sequences=cu.numel()-1,checkpoint=interval)),flush=True);last=m
            keys=['q','k','v','g','beta','cu_seqlens','initial_state','output','output_state','state_checkpoints','checkpoint_cu_starts','checkpoint_every_n_tokens']
            if m=='audit':
                out=original(**kw)
                key=(q.shape[0],interval)
                append('calls.jsonl',dict(tokens=q.shape[0],sequences=cu.numel()-1,interval=interval,checkpoints=0 if kw.get('state_checkpoints') is None else kw['state_checkpoints'].shape[0],eligible=eligible,
                    shape={n:list(kw[n].shape) for n in keys if isinstance(kw.get(n),torch.Tensor)},stride={n:list(kw[n].stride()) for n in keys if isinstance(kw.get(n),torch.Tensor)}))
                if key not in captured and len(captured)<6:
                    captured.add(key)
                    data={n:(kw[n].detach().cpu() if isinstance(kw.get(n),torch.Tensor) else kw.get(n)) for n in keys}
                    torch.save(data,ROOT/'captures'/f't{key[0]}_cp{key[1]}.pt')
                return out
            if not selected:
                if interval<0:
                    return dense_reference(original,kw,compact_output=True)
                return original(**kw)
            inputs={n:kw.get(n) for n in keys};inputs['checkpoint_every_n_tokens']=interval
            if inputs['output'] is None:inputs['output']=torch.empty_like(v)
            if inputs['output_state'] is None:inputs['output_state']=torch.empty_like(state)
            dvsplit.run(**inputs)
            if m in ('verify','verify_sparse'):
                refs=dict(kw,output=torch.empty_like(inputs['output']),output_state=torch.empty_like(inputs['output_state']))
                if kw.get('state_checkpoints') is not None:refs['state_checkpoints']=torch.empty_like(kw['state_checkpoints'])
                if interval<0:dense_reference(original,refs,compact_output=True)
                else:original(**refs)
                errors={}
                for n in ['output','output_state','state_checkpoints']:
                    if refs.get(n) is None:continue
                    aa,bb=inputs[n],refs[n]
                    errors[n]=(aa.float()-bb.float()).abs().max().item() if aa.numel() else 0
                    torch.testing.assert_close(aa,bb,atol=0 if interval else .02,rtol=0 if interval else .02)
                append('real_checks.jsonl',dict(mode=m,tokens=q.shape[0],interval=interval,errors=errors,valid=True))
            return (inputs['output'],inputs['output_state']) if kw.get('output_final_state',False) else inputs['output']
        self._prefill_fn=adapted
    module.FlashInferGDNKernel.__init__=init
    print('GDN35_CHECKPOINT_PATCH_INSTALLED',os.getpid(),flush=True)
