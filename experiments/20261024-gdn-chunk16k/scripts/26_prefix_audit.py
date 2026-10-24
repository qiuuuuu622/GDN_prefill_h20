"""Content-free tokenizer/prefix audit, never print prompt text."""
import pathlib,json,itertools
from transformers import AutoTokenizer
R=pathlib.Path(__file__).resolve().parents[1]
t=AutoTokenizer.from_pretrained(str(R/'models/prefill_model'))
data=json.loads((R/'config/fixed10.json').read_text());seq=[]
for row in data:
 p=row['payload'];ids=t.apply_chat_template(p['messages'],tools=p['tools'],add_generation_prompt=True,tokenize=True)
 if hasattr(ids,'keys'):ids=ids['input_ids']
 assert isinstance(ids,list) and len(ids)>1000 and isinstance(ids[0],int),type(ids)
 seq.append(ids)
mat=[]
for a in seq:
 out=[]
 for b in seq:
  out.append(next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b))))
 mat.append(out)
result=dict(token_counts=[len(x) for x in seq],common_prefix_tokens=mat,method='local tokenizer apply_chat_template tools + add_generation_prompt, counts must match actual API prompt_tokens before interpreting')
(R/'results/prefix_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
