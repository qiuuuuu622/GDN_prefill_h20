"""Validate compact-plan fallback to the original public kernel on real tensors."""
import json,pathlib,torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from service_patch import dense_reference
R=pathlib.Path(__file__).resolve().parents[1]
rows=[]
for filename,target in [('t8192_cp64.pt',128),('t8192_cp64.pt',1280),('t3230_cp64.pt',3200)]:
    saved=torch.load(R/'captures'/filename,weights_only=True)
    x={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in saved.items()}
    expected=x['state_checkpoints'][target//64-1].clone();out=x['output'].clone();state=x['output_state'].clone()
    x.update(output=torch.empty_like(out),output_state=torch.empty_like(state),state_checkpoints=torch.full_like(expected[None],float('nan')),checkpoint_cu_starts=torch.tensor([0,1],device='cuda',dtype=torch.int64),checkpoint_every_n_tokens=-target,output_final_state=True)
    dense_reference(chunk_gated_delta_rule,x,compact_output=True);torch.cuda.synchronize()
    torch.testing.assert_close(x['output'],out,atol=0,rtol=0)
    torch.testing.assert_close(x['output_state'],state,atol=0,rtol=0)
    torch.testing.assert_close(x['state_checkpoints'][0],expected,atol=0,rtol=0)
    rows.append(dict(file=filename,target=target,valid=True));print(rows[-1],flush=True)
(R/'results/fallback_checks.json').write_text(json.dumps(rows,indent=2))
print('COMPLETE_FALLBACK',len(rows),flush=True)
