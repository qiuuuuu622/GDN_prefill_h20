import torch
import benchmark as b
import dvsplit

for lengths in [[1],[65],[257],[511,1025,0,129]]:
    x=b.make_inputs(lengths,16,32,147,"near_one")
    b.noncp_fn(**x)
    ref={n:x[n].clone() for n in ["output","output_state"]}
    dvsplit.run(**x)
    torch.cuda.synchronize()
    assert b.check(x,ref,lengths)["valid"]
    if len(lengths)>1:
        def fn(cp_chunk_len=None,**kw):return dvsplit.run(**kw)
        li=max(range(len(lengths)),key=lengths.__getitem__)
        b.split.execute_split(b.split.SplitPlan(li,0),lengths=lengths,**x,cp_fn=fn,noncp_fn=b.noncp_fn)
        torch.cuda.synchronize()
        assert b.check(x,ref,lengths)["valid"]
    print("PASS",lengths,flush=True)
print("COMPLETED",flush=True)
