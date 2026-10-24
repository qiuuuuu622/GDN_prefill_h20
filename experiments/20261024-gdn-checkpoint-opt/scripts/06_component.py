"""Complete SGLang extend timing including identical state-pool resets."""
import importlib.util,json,pathlib,torch
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import FlashInferGDNKernel
import dvsplit
spec=importlib.util.spec_from_file_location('test',pathlib.Path(__file__).with_name('01_checkpoint_test.py'));t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
R=t.ROOT
def adapted(**kw):
    keys=['q','k','v','g','beta','cu_seqlens','initial_state','output','output_state','state_checkpoints','checkpoint_cu_starts','checkpoint_every_n_tokens']
    args={k:kw.get(k) for k in keys}
    if args['output'] is None:args['output']=torch.empty_like(kw['v'])
    if args['output_state'] is None:args['output_state']=torch.empty_like(kw['initial_state'])
    dvsplit.run(**args)
    return args['output'],args['output_state']
def main():
    rows=[]
    for n in [2239,3230,8192]:
        x=t.inputs([n]);pool0=torch.zeros(3,32,128,128,device='cuda');pool0[1]=x['initial_state'][0]
        idx=torch.tensor([1],device='cuda',dtype=torch.int64);target=(n//64)*64
        baseline=None
        for method in ['baseline','dense_dv','compact_dv']:
            kernel=FlashInferGDNKernel()
            if method!='baseline':kernel._prefill_fn=adapted
            pool=pool0.clone();interval=-target if method=='compact_dv' else 64;count=1 if interval<0 else n//interval
            cu=torch.tensor([0,count],device='cuda',dtype=torch.int64)
            q,k,v=[x[name][None] for name in ['q','k','v']];g=x['g'].log()[None];beta=x['beta'][None]
            def run():
                pool.copy_(pool0)
                return kernel.extend(q,k,v,g,beta,ssm_states=pool,cache_indices=idx,query_start_loc=x['cu_seqlens'],state_checkpoint_cu_starts=cu,num_state_checkpoints=count,state_checkpoint_every_n_tokens=interval)
            result=run();torch.cuda.synchronize()
            if method=='baseline':baseline=(result[0].clone(),pool.clone(),result[2][:,-1].clone())
            else:
                torch.testing.assert_close(result[0],baseline[0],atol=0,rtol=0)
                torch.testing.assert_close(pool,baseline[1],atol=0,rtol=0)
                torch.testing.assert_close(result[2][:,-1],baseline[2],atol=0,rtol=0)
            row=dict(tokens=n,method=method,interval=interval,count=count,valid=True,timing=t.timing(run))
            rows.append(row);print(json.dumps(row),flush=True)
            (R/'results/component.json').write_text(json.dumps(rows,indent=2))
        del x,pool,baseline,result
        torch.cuda.empty_cache()
    print('COMPLETE_COMPONENT',len(rows),flush=True)
if __name__=='__main__':main()
