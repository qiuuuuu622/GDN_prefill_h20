import json,pathlib,hashlib,collections,re
from transformers import AutoTokenizer
root=pathlib.Path('/root/gdn35/20261023-e2e-1p1d')
src=pathlib.Path('/private/benchmark-data/122b_20260914_2030_2130_clean.jsonl')
tok=AutoTokenizer.from_pretrained('/prefill_model',local_files_only=True)
candidates=[];seen=set();counts=collections.Counter()
with src.open() as f:
 for line_no,line in enumerate(f,1):
  if line_no>1000:break
  x=json.loads(line)['t'];msg=x.get('messages');tools=x.get('tools')
  if not isinstance(msg,list) or not msg:counts['no_messages']+=1;continue
  image_removed=0
  for m in msg:
   content=m.get('content')
   if isinstance(content,list):
    kept=[]
    for part in content:
     if isinstance(part,dict) and part.get('type') in ['image','image_url','input_image']:
      image_removed+=1;continue
     kept.append(part)
    m['content']=kept if kept else ''
   if isinstance(m.get('content'),str):
    for marker in ['[base64_image_filtered]','[image_filtered]']:
     image_removed+=m['content'].count(marker);m['content']=m['content'].replace(marker,'')
  counts['image_blocks_or_markers_removed']+=image_removed
  if any(isinstance(m.get('content'),list) and any(p.get('type') not in ['text','input_text'] for p in m['content'] if isinstance(p,dict)) for m in msg):counts['other_multimodal_skipped']+=1;continue
  raw=json.dumps({'messages':msg,'tools':tools},ensure_ascii=False,sort_keys=True)
  sha=hashlib.sha256(raw.encode()).hexdigest()
  if sha in seen:counts['duplicate']+=1;continue
  seen.add(sha)
  try:ids=tok.apply_chat_template(msg,tools=tools or None,tokenize=True,add_generation_prompt=True)
  except Exception:counts['template_error']+=1;continue
  n=len(ids["input_ids"] if hasattr(ids,"keys") else ids)
  if not 1024<=n<=120000:counts['length_excluded']+=1;continue
  payload=dict(model='gdn35-e2e',messages=msg,temperature=0,max_tokens=512,seed=42,stream=True,stream_options={'include_usage':True})
  if tools:payload['tools']=tools
  if x.get('tool_choice') is not None:payload['tool_choice']=x['tool_choice']
  if x.get('chat_template_kwargs') is not None:payload['chat_template_kwargs']=x['chat_template_kwargs']
  candidates.append(dict(source_line=line_no,image_blocks_removed=image_removed,content_sha256=sha,prompt_tokens_estimate=n,payload=payload))
assert len(candidates)>=10,len(candidates)
candidates.sort(key=lambda x:(x['prompt_tokens_estimate'],x['source_line']))
selected=[candidates[round((i+.5)/10*(len(candidates)-1))] for i in range(10)]
for i,x in enumerate(selected):x['sample_id']=f'sample_{i:02d}'
p=root/'config/fixed10.json';p.write_text(json.dumps(selected,ensure_ascii=False,indent=2))
meta=dict(source=str(src),source_size=src.stat().st_size,selection='First 1000 source lines; deduplicate messages+tools; remove image blocks/placeholders then text-only; 1024..120000 template tokens; midpoint of ten length quantile bins',candidates=len(candidates),excluded=dict(counts),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),samples=[{k:v for k,v in x.items() if k!='payload'} for x in selected])
(root/'evidence/dataset_manifest.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2),flush=True)
