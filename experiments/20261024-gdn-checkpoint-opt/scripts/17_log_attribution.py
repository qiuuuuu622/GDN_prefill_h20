"""Align sequential P request logs with client rows. Log intervals are not GPU timing."""
import datetime,json,pathlib,re
R=pathlib.Path(__file__).resolve().parents[1]
requests=[json.loads(l) for l in (R/'results/requests.jsonl').read_text().splitlines()]
out=[]
for variant in ['stock_a1','sparse_fixed','stock_a2']:
    lines=(R/f'logs/p_{variant}.log').read_text().splitlines();records=[];active=None
    for line in lines:
        line=re.sub(r'\x1b\[[0-9;]*m','',line)
        date=re.match(r'(\d{4}-\d\d-\d\dT[\d:.]+Z)',line)
        if not date:continue
        ts=datetime.datetime.fromisoformat(date[1].replace('Z','+00:00')).timestamp()
        is_generate='component="prefill"' in line and 'endpoint="generate"' in line
        if is_generate and 'request received' in line:
            rid=re.search(r'request_id="?([a-f0-9-]+)',line)[1];assert active is None
            active=dict(rid=rid,start=ts,batches=[])
        elif 'Prefill batch,' in line and active is not None:
            m=re.search(r'#new-token: (\d+), #cached-token: (\d+)',line)
            active['batches'].append(dict(ts=ts,new_tokens=int(m[1]),cached_tokens=int(m[2])))
        elif is_generate and 'request completed' in line and active is not None:
            rid=re.search(r'request_id="?([a-f0-9-]+)',line)[1];assert rid==active['rid'];active['end']=ts;records.append(active);active=None
    matching=[x for x in requests if x['phase']=='formal_shape_warm_'+variant or x['phase']=='formal_stock_abba' and ((variant=='sparse_fixed' and x['mode']=='sparse') or (variant=='stock_a1' and x['mode']=='baseline' and x['rep']==1) or (variant=='stock_a2' and x['mode']=='baseline' and x['rep']==2))]
    assert len(records)==len(matching),(variant,len(records),len(matching))
    for r,x in zip(records,matching):
        if x['phase']!='formal_stock_abba':continue
        assert [{k:b[k] for k in ['new_tokens','cached_tokens']} for b in r['batches']]==x['prefill_batches']
        row=dict(tag=x['tag'],mode=x['mode'],rep=x['rep'],sample_id=x['sample_id'],ttft_ms=x['ttft_ms'],p_handler_ms=(r['end']-r['start'])*1000,first_stats_after_received_ms=(r['batches'][0]['ts']-r['start'])*1000,last_stats_to_handler_end_ms=(r['end']-r['batches'][-1]['ts'])*1000)
        out.append(row)
(R/'results/formal_p_log_attribution.json').write_text(json.dumps(out,indent=2))
for sid in ['sample_01','sample_02','sample_03']:
    print(sid,[{k:round(v,2) if isinstance(v,float) else v for k,v in x.items() if k!='tag'} for x in out if x['sample_id']==sid])
