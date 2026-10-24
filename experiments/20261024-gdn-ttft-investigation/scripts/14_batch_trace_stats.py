import json,gzip,collections,sys,statistics
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
R=Path(sys.argv[1]);mode=sys.argv[2]
p=next((R/f'profiles/batch_{mode}').glob('*.gz'));trace=json.load(gzip.open(p,'rt'));base=trace['baseTimeNanoseconds']/1e9
es=[e for e in trace['traceEvents'] if e.get('ph')=='X' and e.get('dur',0)>0]
gc=[json.loads(l) for l in (R/'evidence/gc_events.jsonl').read_text().splitlines()];offset=statistics.median((e['start_wall_ns']-e['start_perf_ns'])/1e9 for e in gc)
rr=[json.loads(l) for l in (R/'results/requests.jsonl').read_text().splitlines()];rr=[x for x in rr if x['phase']=='batch_profile' and x['mode']==mode]
spec=spec_from_file_location('ts',R/'scripts/09_trace_stats.py');ts=module_from_spec(spec);spec.loader.exec_module(ts)
rows=[]
for x in rr:
 start=(x['client_start_monotonic_s']+offset-base)*1e6;end=start+x['ttft_ms']*1000
 active=[e for e in es if e['ts']>=start and e['ts']<=end];kernels=[e for e in active if e.get('cat')=='kernel'];apis=collections.defaultdict(list)
 for e in active:
  if e.get('cat') in ['cuda_runtime','cuda_driver']:apis[e['name']].append(e)
 api={n:dict(count=len(ee),total_ms=sum(e['dur'] for e in ee)/1000,max_ms=max(e['dur'] for e in ee)/1000) for n,ee in apis.items()};api=dict(sorted(api.items(),key=lambda x:-x[1]['total_ms']))
 long_cpu=sorted([e for e in active if e.get('cat') in ['cpu_op','user_annotation'] and e['dur']>=10000],key=lambda e:-e['dur'])[:20]
 row=dict(tag=x['tag'],ttft_ms=x['ttft_ms'],gpu_total_ms=sum(e['dur'] for e in kernels)/1000,gpu_busy_ms=ts.union_ms([(e['ts'],e['ts']+e['dur']) for e in kernels]),first_kernel_delay_ms=(min(e['ts'] for e in kernels)-start)/1000,last_kernel_to_ttft_ms=(end-max(e['ts']+e['dur'] for e in kernels))/1000,api=api,long_cpu=[dict(name=e['name'],dur_ms=e['dur']/1000,relative_ms=(e['ts']-start)/1000) for e in long_cpu])
 rows.append(row);print(json.dumps(row))
(R/f'results/batch_trace_attribution_{mode}.json').write_text(json.dumps(rows,indent=2))
