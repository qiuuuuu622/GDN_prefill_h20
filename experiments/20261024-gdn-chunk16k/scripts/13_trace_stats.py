import json,gzip,collections,sys,bisect
from pathlib import Path

def union_ms(intervals):
 total=0;end=None
 for a,b in sorted(intervals):
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total/1000

def analyze(path):
 raw=json.loads(gzip.open(path,'rt').read() if str(path).endswith('.gz') else path.read_text())
 events=raw['traceEvents'];dur=[e for e in events if e.get('ph')=='X' and e.get('dur',0)>0]
 cats=collections.Counter(e.get('cat','') for e in dur)
 kernels=[e for e in dur if e.get('cat')=='kernel']; byname=collections.defaultdict(list)
 for e in kernels:byname[e['name']].append(e)
 groups=[]
 for n,es in byname.items():groups.append(dict(name=n,count=len(es),sum_ms=sum(e['dur'] for e in es)/1000,mean_us=sum(e['dur'] for e in es)/len(es)))
 groups.sort(key=lambda x:-x['sum_ms'])
 scopes=[e for e in dur if e.get('cat')=='user_annotation' and e['name'].startswith('GDN35/')]
 sc=collections.defaultdict(list)
 for e in scopes:sc[e['name']].append(e)
 scope_stats={n:dict(count=len(es),total_cpu_ms=sum(e['dur'] for e in es)/1000,min_us=min(e['dur'] for e in es),max_us=max(e['dur'] for e in es)) for n,es in sc.items()}
 calls=collections.defaultdict(list)
 for e in dur:
  if e.get('cat') in ('cuda_runtime','cuda_driver'):calls[e['name']].append(e)
 api={n:dict(count=len(es),total_ms=sum(e['dur'] for e in es)/1000,max_ms=max(e['dur'] for e in es)/1000) for n,es in calls.items()}
 api=dict(sorted(api.items(),key=lambda x:-x[1]['total_ms']))
 steps=[dict(name=e['name'],ts=e['ts'],dur_ms=e['dur']/1000) for e in dur if 'Forward' in e['name'] or e['name'].startswith(('SGLang','GDN35'))]
 summary=dict(path=str(path),categories=dict(cats),kernel_count=len(kernels),kernel_total_ms=sum(e['dur'] for e in kernels)/1000,kernel_union_ms=union_ms([(e['ts'],e['ts']+e['dur']) for e in kernels]),kernel_span_ms=(max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels))/1000 if kernels else 0,kernel_groups=groups,scope_stats=scope_stats,cuda_api=api,steps=steps)
 return summary
if __name__=='__main__':
 path=Path(sys.argv[1]);s=analyze(path);out=Path(sys.argv[2]);out.write_text(json.dumps(s,indent=2));print(json.dumps({k:v for k,v in s.items() if k not in ['kernel_groups','steps']},indent=2));print('TOP_KERNELS',json.dumps(s['kernel_groups'][:12],indent=2));print('GDN_KERNELS',json.dumps([k for k in s['kernel_groups'] if any(t in k['name'].lower() for t in ['delta','gdn','state_checkpoint'])],indent=2))
