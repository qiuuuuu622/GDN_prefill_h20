"""Replay actual service tensors; test compact checkpoints against dense snapshots."""
import importlib.util,json,pathlib,torch
import dvsplit
spec=importlib.util.spec_from_file_location('test',pathlib.Path(__file__).with_name('01_checkpoint_test.py'));t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
R=t.ROOT

def clone_buffers(x):
    return dict(x,**{k:torch.empty_like(x[k]) for k in ['output','output_state','state_checkpoints'] if x.get(k) is not None})

def metrics(a,b):
    e=(a.float()-b.float()).abs()
    return dict(exact=bool(torch.equal(a,b)),max_abs=e.max().item() if e.numel() else 0,rms=e.square().mean().sqrt().item() if e.numel() else 0,finite=bool(torch.isfinite(a).all()))

def recurrence(x):
    q=x['q'].float().repeat_interleave(2,dim=1);k=x['k'].float().repeat_interleave(2,dim=1);v=x['v'].float()
    state=x['initial_state'][0].clone();outputs=[];checkpoints=[]
    for i in range(v.shape[0]):
        state=state*x['g'][i,:,None,None]
        delta=(v[i]-torch.bmm(state,k[i].unsqueeze(-1)).squeeze(-1))*x['beta'][i,:,None]
        state=state+delta.unsqueeze(-1)*k[i].unsqueeze(-2)
        outputs.append(torch.bmm(state,q[i].unsqueeze(-1)).squeeze(-1)*128**-.5)
        if (i+1)%64==0:checkpoints.append(state.clone())
    return dict(output=torch.stack(outputs),output_state=state[None],state_checkpoints=torch.stack(checkpoints))

def main():
    rows=[]
    for p in sorted((R/'captures').glob('*.pt')):
        saved=torch.load(p,weights_only=True);x={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in saved.items()};n=x['q'].shape[0]
        dense=clone_buffers(x);(dvsplit.run(**dense) if x['checkpoint_every_n_tokens'] else t.baseline(dense));torch.cuda.synchronize()
        row=dict(file=p.name,tokens=n,interval=x['checkpoint_every_n_tokens'],dense={k:metrics(dense[k],x[k]) for k in ['output','output_state','state_checkpoints'] if x[k] is not None})
        for k,v in row['dense'].items():
            assert v['finite']
            torch.testing.assert_close(dense[k],x[k],atol=0 if row['interval'] else .02,rtol=0 if row['interval'] else .02)
        row['timing_dense']=t.timing(lambda:dvsplit.run(**dense) if x['checkpoint_every_n_tokens'] else t.baseline(dense))
        row['timing_baseline']=t.timing(lambda:t.baseline(x))
        if x['checkpoint_every_n_tokens']:
            # These targets produce exactly one snapshot. All are original 64-token boundaries.
            targets=set([((n//2)//64+1)*64,(n//64)*64])
            for line in (R/'evidence/plans.jsonl').read_text().splitlines():
                plan=json.loads(line)
                if plan['mode']=='audit' and plan['lengths']==[n] and plan['interval']==64:
                    targets.update((i+1)*64 for i in (plan['selected'] or []))
            targets=sorted(targets)
            checks=[]
            for target in targets:
                if not (0<target<=n):continue
                compact=clone_buffers(x);compact.update(state_checkpoints=torch.empty_like(x['state_checkpoints'][:1]),checkpoint_cu_starts=torch.tensor([0,1],device='cuda',dtype=torch.int64),checkpoint_every_n_tokens=-target)
                dvsplit.run(**compact);torch.cuda.synchronize()
                check=dict(target=target,output=metrics(compact['output'],dense['output']),output_state=metrics(compact['output_state'],dense['output_state']),checkpoint=metrics(compact['state_checkpoints'][0],dense['state_checkpoints'][target//64-1]))
                assert all(check[k]['exact'] for k in ['output','output_state','checkpoint'])
                if target<n:
                    z={k:(v[target:].contiguous() if k in ['q','k','v','g','beta'] else v) for k,v in x.items()}
                    z.update(initial_state=compact['state_checkpoints'].clone(),cu_seqlens=torch.tensor([0,n-target],device='cuda',dtype=torch.int64),output=torch.empty_like(x['output'][target:]),output_state=torch.empty_like(x['output_state']),state_checkpoints=None,checkpoint_cu_starts=None,checkpoint_every_n_tokens=0)
                    dvsplit.run(**z);torch.cuda.synchronize()
                    check['restart_output']=metrics(z['output'],dense['output'][target:]);check['restart_state']=metrics(z['output_state'],dense['output_state'])
                    assert check['restart_output']['exact'] and check['restart_state']['exact']
                check['timing']=t.timing(lambda:dvsplit.run(**compact));checks.append(check)
            row['compact']=checks
        # Independent FP32 recurrence on real first 129 tokens, including both checkpoints.
        z={k:(v[:129].contiguous() if k in ['q','k','v','g','beta'] else v) for k,v in x.items()}
        z.update(cu_seqlens=torch.tensor([0,129],device='cuda',dtype=torch.int64),output=torch.empty_like(x['output'][:129]),output_state=torch.empty_like(x['output_state']),state_checkpoints=torch.empty((2,32,128,128),device='cuda'),checkpoint_cu_starts=torch.tensor([0,2],device='cuda',dtype=torch.int64),checkpoint_every_n_tokens=64)
        dvsplit.run(**z);ref=recurrence(z);torch.cuda.synchronize()
        row['fp32_reference']={k:metrics(z[k],ref[k]) for k in ref}
        row['fp32_close']={k:bool(torch.allclose(z[k].float(),ref[k].float(),atol=.02 if k=='output' else .01,rtol=.02 if k=='output' else .01)) for k in ref}
        original_short=clone_buffers(z);t.baseline(original_short);torch.cuda.synchronize()
        row['original_fp32_close']={k:bool(torch.allclose(original_short[k].float(),ref[k].float(),atol=.02 if k=='output' else .01,rtol=.02 if k=='output' else .01)) for k in ref}
        row['original_fp32_reference']={k:metrics(original_short[k],ref[k]) for k in ref}
        row['short_exact_original']={k:bool(torch.equal(z[k],original_short[k])) for k in ref}
        assert all(row['short_exact_original'].values()),row
        row['independent_reference_status']='pass' if all(row['fp32_close'].values()) else 'inherited_baseline_threshold_failure'
        rows.append(row);(R/'results/real_replay.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
        del x,saved,dense,z,ref
        torch.cuda.empty_cache()
    print('COMPLETE_REPLAY',len(rows),flush=True)
if __name__=='__main__':main()
