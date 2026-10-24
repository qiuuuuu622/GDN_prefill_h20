"""Audit the final stock ABBA comparison without filtering requests."""
import json,pathlib,statistics,hashlib
R=pathlib.Path(__file__).resolve().parents[1]
def percentile(xs,p):
    a=sorted(xs);i=(len(a)-1)*p/100;j=int(i)
    return a[j]+(a[min(j+1,len(a)-1)]-a[j])*(i-j)
def stats(xs):return dict(mean=statistics.mean(xs),**{f'p{p}':percentile(xs,p) for p in [50,75,90,95,99]})
def normalized_text(row):
    events=json.loads((R/'results'/f"{row['tag']}.sse.json").read_text());body={'content':'','reasoning':'','tools':{}}
    for e in events:
      for c in e['data'].get('choices',[]):
        d=c.get('delta',{});body['content']+=d.get('content') or '';body['reasoning']+=d.get('reasoning_content') or d.get('reasoning') or ''
        for t in d.get('tool_calls',[]):
            key=str(t.get('index',0));b=body['tools'].setdefault(key,{'name':'','arguments':''});f=t.get('function',{})
            b['name']+=f.get('name') or '';b['arguments']+=f.get('arguments') or ''
    return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def main():
    allrows=[json.loads(l) for l in (R/'results/requests.jsonl').read_text().splitlines()]
    rows=[x for x in allrows if x['phase']=='formal_stock_abba'];assert len(rows)==40,len(rows)
    result={'phase':'formal_stock_abba','requests':len(rows),'arms':{},'batches':[],'pairs':[]}
    for mode in ['baseline','sparse']:
        arm=[x for x in rows if x['mode']==mode];assert len(arm)==20
        result['arms'][mode]={m:stats([x[m] for x in arm]) for m in ['ttft_ms','tpot_ms','e2e_ms']}
        result['arms'][mode]['mean_output_tokens']=statistics.mean(x['usage']['completion_tokens'] for x in arm)
        for rep in [1,2]:
            batch=[x for x in arm if x['rep']==rep];assert len(batch)==10 and all(x['status']==200 for x in batch)
            result['batches'].append(dict(mode=mode,rep=rep,ttft=stats([x['ttft_ms'] for x in batch]),e2e=stats([x['e2e_ms'] for x in batch])))
    for rep in [1,2]:
      for i in range(10):
        sid=f'sample_{i:02d}';a=next(x for x in rows if x['rep']==rep and x['mode']=='baseline' and x['sample_id']==sid);b=next(x for x in rows if x['rep']==rep and x['mode']=='sparse' and x['sample_id']==sid)
        same_cache=a['prefill_batches']==b['prefill_batches'];assert same_cache,(rep,sid,a['prefill_batches'],b['prefill_batches'])
        assert a['usage']['prompt_tokens']==b['usage']['prompt_tokens']
        result['pairs'].append(dict(rep=rep,sample_id=sid,cache_equal=same_cache,output_text_equal=normalized_text(a)==normalized_text(b),a_output_tokens=a['usage']['completion_tokens'],b_output_tokens=b['usage']['completion_tokens'],a_ttft_ms=a['ttft_ms'],b_ttft_ms=b['ttft_ms'],saved_ttft_ms=a['ttft_ms']-b['ttft_ms']))
    for m in ['ttft_ms','tpot_ms','e2e_ms']:
        a=result['arms']['baseline'][m]['mean'];b=result['arms']['sparse'][m]['mean'];result[m+'_change_percent']=(b/a-1)*100
    result['cache_pairs_equal']=sum(x['cache_equal'] for x in result['pairs']);result['output_pairs_equal']=sum(x['output_text_equal'] for x in result['pairs'])
    result['paired_ttft_saved_ms']=stats([x['saved_ttft_ms'] for x in result['pairs']])
    (R/'results/formal_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
