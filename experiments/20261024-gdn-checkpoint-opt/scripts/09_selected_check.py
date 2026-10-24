"""Selected checkpoint parity and resume tests, also used under Compute Sanitizer."""
import argparse,importlib.util,json,pathlib,torch
import dvsplit
spec=importlib.util.spec_from_file_location('test',pathlib.Path(__file__).with_name('01_checkpoint_test.py'));t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
def main():
    p=argparse.ArgumentParser();p.add_argument('--memcheck',action='store_true');args=p.parse_args();rows=[]
    for init in ([True] if args.memcheck else [False,True]):
      for n in ([65,8192] if args.memcheck else [65,129,513,1025,8192]):
        x=t.inputs([n],init=init,seed=127+n);t.baseline(x);ref=t.snapshot(x)
        for target in sorted(set([64,max(64,((n//2)//64)*64),(n//64)*64])):
            z=dict(x,output=torch.full_like(x['output'],float('nan')),output_state=torch.full_like(x['output_state'],float('nan')),state_checkpoints=torch.full_like(x['state_checkpoints'][:1],float('nan')),checkpoint_cu_starts=torch.tensor([0,1],device='cuda',dtype=torch.int64),checkpoint_every_n_tokens=-target)
            dvsplit.run(**z);torch.cuda.synchronize()
            for key in ['output','output_state']:torch.testing.assert_close(z[key],ref[key],atol=0,rtol=0)
            torch.testing.assert_close(z['state_checkpoints'][0],ref['state_checkpoints'][target//64-1],atol=0,rtol=0)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):dvsplit.run(**z)
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(z['state_checkpoints'][0],ref['state_checkpoints'][target//64-1],atol=0,rtol=0)
            if target<n:
                cont={k:(v[target:].contiguous() if k in ['q','k','v','g','beta'] else v) for k,v in z.items()}
                cont.update(initial_state=z['state_checkpoints'].clone(),cu_seqlens=torch.tensor([0,n-target],device='cuda',dtype=torch.int64),output=torch.empty_like(x['output'][target:]),output_state=torch.empty_like(x['output_state']),state_checkpoints=None,checkpoint_cu_starts=None,checkpoint_every_n_tokens=0)
                dvsplit.run(**cont);torch.cuda.synchronize()
                torch.testing.assert_close(cont['output'],ref['output'][target:],atol=0,rtol=0)
                torch.testing.assert_close(cont['output_state'],ref['output_state'],atol=0,rtol=0)
            rows.append(dict(tokens=n,target=target,initial_nonzero=init,valid=True));print(rows[-1],flush=True)
            del graph,z
        del x,ref
        torch.cuda.empty_cache()
    (t.ROOT/'results'/('selected_memcheck.json' if args.memcheck else 'selected_checks.json')).write_text(json.dumps(rows,indent=2))
    print('COMPLETE_SELECTED',len(rows),flush=True)
if __name__=='__main__':main()
