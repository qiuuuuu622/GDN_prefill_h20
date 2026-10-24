"""Single-sequence TP1 Dv split adapter with runtime A/B selection.

Identical control-file lookup in both arms. No device-to-host length read.
All unsupported calls fall back, including state checkpoints.
"""
import os,json,pathlib,collections
import torch
import dvsplit
ROOT=pathlib.Path('/root/gdn35/20261024-e2e-sequential-1p1d')

def install(module):
 original_init=module.FlashInferGDNKernel.__init__
 def init(self,*args,**kwargs):
  original_init(self,*args,**kwargs)
  original=self._prefill_fn
  counts=collections.Counter();last=None
  def adapted(**kw):
   nonlocal last
   mode=(ROOT/'config/mode.txt').read_text().strip()
   q,k,v=kw['q'],kw['k'],kw['v'];state=kw['initial_state'];cu=kw['cu_seqlens']
   eligible=(cu.numel()==2 and tuple(q.shape[1:])==(16,128) and tuple(k.shape[1:])==(16,128) and tuple(v.shape[1:])==(32,128) and q.dtype==torch.bfloat16 and v.dtype==torch.bfloat16 and state is not None and state.dtype==torch.float32 and state.is_contiguous() and kw.get('checkpoint_every_n_tokens',0)==0 and kw.get('state_checkpoints') is None and kw.get('state_indices') is None and not kw.get('use_qk_l2norm_in_kernel',False) and kw.get('scale') is None and torch.cuda.get_device_properties(q.device).multi_processor_count==78)
   selected=mode in ('optimized','verify') and eligible
   counts[(mode,selected)]+=1
   if mode!=last or counts[(mode,selected)]%30==0:
    print('GDN35_ROUTE',json.dumps(dict(pid=os.getpid(),mode=mode,selected=selected,count=counts[(mode,selected)],tokens=q.shape[0],sequences=cu.numel()-1,checkpoint=kw.get('checkpoint_every_n_tokens',0))),flush=True);last=mode
   if not selected:return original(**kw)
   output=kw.get('output');outstate=kw.get('output_state')
   if output is None:output=torch.empty_like(v)
   if outstate is None:outstate=torch.empty_like(state)
   args=dict(q=q,k=k,v=v,g=kw['g'],beta=kw['beta'],initial_state=state,cu_seqlens=cu,output=output,output_state=outstate)
   dvsplit.run(**args)
   if mode=='verify':
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm90 import delta_rule_prefill_dsl_sm90
    refout=torch.empty_like(output);refs=torch.empty_like(outstate)
    delta_rule_prefill_dsl_sm90(refout,refs,q,k,v,state,kw['g'],kw['beta'],cu,128**-0.5)
    torch.testing.assert_close(output,refout,atol=0,rtol=0)
    torch.testing.assert_close(outstate,refs,atol=0,rtol=0)
    with (ROOT/'evidence/real_activation_checks.jsonl').open('a') as f:f.write(json.dumps(dict(tokens=q.shape[0],output_max_abs=0,state_max_abs=0,valid=True))+'\n')
   return (output,outstate) if kw.get('output_final_state',False) else output
  self._prefill_fn=adapted
 module.FlashInferGDNKernel.__init__=init
 print('GDN35_PATCH_INSTALLED',os.getpid(),flush=True)
