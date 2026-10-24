import json,pathlib,statistics,hashlib,collections
import numpy as np
R=pathlib.Path('/root/gdn35/20261023-e2e-1p1d')
rows=[json.loads(l) for l in (R/'results/requests.jsonl').read_text().splitlines()]
rows=[r for r in rows if r['phase']=='measured']
assert len(rows)==40,len(rows)
metrics=['ttft_ms','tpot_ms','e2e_ms'];modes=['baseline','optimized']
summary={};paired=[]
for mode in modes:
 arm=[r for r in rows if r['mode']==mode];assert len(arm)==20
 assert len({(r['sample_id'],r['rep']) for r in arm})==20
 per_sample=[]
 for sample in sorted({r['sample_id'] for r in arm}):
  rr=[r for r in arm if r['sample_id']==sample]
  per_sample.append(dict(sample_id=sample,**{k:statistics.median(r[k] for r in rr) for k in metrics},prompt_tokens=[r['usage']['prompt_tokens'] for r in rr],completion_tokens=[r['usage']['completion_tokens'] for r in rr]))
 summary[mode]=dict(requests=len(arm),per_sample_medians=per_sample)
 for name,rr in [('pooled_20',arm),('sample_medians_10',per_sample)]:
  summary[mode][name]={k:dict(mean=float(np.mean([r[k] for r in rr])),**{f'p{p}':float(np.percentile([r[k] for r in rr],p)) for p in [0,50,75,90,95,99,100]}) for k in metrics}
 for r in arm:
  events=json.loads((R/'results'/f'{r["tag"]}.sse.json').read_text());content=[];reason=[];tools={}
  for e in events:
   for c in e['data'].get('choices',[]):
    d=c.get('delta',{});content.append(d.get('content') or '');reason.append(d.get('reasoning_content') or d.get('reasoning') or '')
    for t in d.get('tool_calls',[]):
     a=tools.setdefault(t.get('index',0),{'name':'','arguments':''});fun=t.get('function',{})
     a['name']+=fun.get('name') or '';a['arguments']+=fun.get('arguments') or ''
  r['normalized_output_sha256']=hashlib.sha256(json.dumps(dict(content=''.join(content),reasoning=''.join(reason),tools=tools),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
for rep in range(1,3):
 for sample in sorted({r['sample_id'] for r in rows}):
  a=next(r for r in rows if r['rep']==rep and r['sample_id']==sample and r['mode']=='baseline');b=next(r for r in rows if r['rep']==rep and r['sample_id']==sample and r['mode']=='optimized')
  paired.append(dict(rep=rep,sample_id=sample,prompt_equal=a['usage']['prompt_tokens']==b['usage']['prompt_tokens'],output_tokens_equal=a['usage']['completion_tokens']==b['usage']['completion_tokens'],output_equal=a['normalized_output_sha256']==b['normalized_output_sha256'],**{k+'_reduction_pct':100*(a[k]-b[k])/a[k] for k in metrics}))
summary['pairs']=paired
summary['dataset_sha256']=hashlib.sha256((R/'config/fixed10.json').read_bytes()).hexdigest()
summary['real_activation_checks']=sum(1 for _ in (R/'evidence/real_activation_checks.jsonl').open())
(R/'results/summary.json').write_text(json.dumps(summary,indent=2))
(R/'results/measured_normalized.json').write_text(json.dumps(rows,indent=2))
print('PERCENTILES OF 10 PER-SAMPLE MEDIANS (2 repeats), ms or ms/token')
for k in metrics:
 print(k)
 for p in ['mean','p50','p75','p90','p95','p99']:
  a=summary['baseline']['sample_medians_10'][k][p];b=summary['optimized']['sample_medians_10'][k][p]
  print(p,round(a,3),round(b,3),f'{100*(a-b)/a:.2f}%')
print('PAIRS',dict(total=len(paired),prompt_equal=sum(p['prompt_equal'] for p in paired),output_tokens_equal=sum(p['output_tokens_equal'] for p in paired),output_equal=sum(p['output_equal'] for p in paired)))
