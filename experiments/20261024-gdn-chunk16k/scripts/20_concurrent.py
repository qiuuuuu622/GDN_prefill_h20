"""Fixed ten closed-loop requests, shadow verification then concurrent stock ABBA."""
import asyncio,importlib.util,json,pathlib,time,hashlib,re,statistics,subprocess,traceback
import aiohttp
R=pathlib.Path('/root/gdn35/20261024-gdn-chunk16k')
def module(name,file):
 s=importlib.util.spec_from_file_location(name,R/'scripts'/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=module('bench','04_benchmark.py');f=module('formal','08_formal.py')
def append(name,obj):
 with (R/'results'/name).open('a') as h:h.write(json.dumps(obj,ensure_ascii=False)+'\n')
def pct(xs,p):
 a=sorted(xs);i=(len(a)-1)*p/100;j=int(i);return a[j]+(a[min(j+1,len(a)-1)]-a[j])*(i-j)
def stats(xs):return dict(mean=statistics.mean(xs),**{f'p{p}':pct(xs,p) for p in [50,75,90,95,99]})
async def request(session,sample,mode,phase,rep,c):
 tag=f'{phase}_c{c}_r{rep}_{mode}_{sample["sample_id"]}'
 start=time.perf_counter();first=None;events=[];usage=None;done=False;error=None;status=None
 try:
  async with session.post('http://127.0.0.1:18000/v1/chat/completions',json=sample['payload']) as resp:
   status=resp.status
   if status!=200:raise RuntimeError(f'HTTP {status}: {(await resp.text())[:1000]}')
   async for raw in resp.content:
    line=raw.decode().strip()
    if not line.startswith('data:'):continue
    data=line[5:].strip();now=time.perf_counter()
    if data=='[DONE]':done=True;break
    x=json.loads(data);events.append(dict(elapsed_s=now-start,data=x))
    if x.get('error'):raise RuntimeError(str(x['error']))
    if x.get('usage'):usage=x['usage']
    for choice in x.get('choices',[]):
     d=choice.get('delta',{})
     meaningful=bool(d.get('content') or d.get('reasoning_content') or d.get('reasoning') or any(t.get('function',{}).get('arguments') or t.get('function',{}).get('name') for t in d.get('tool_calls',[])))
     if meaningful and first is None:first=now
  assert done and first is not None and usage and usage['completion_tokens']>1,(done,first,usage)
 except Exception as e:error=repr(e)
 end=time.perf_counter()
 (R/'results'/f'{tag}.sse.json').write_text(json.dumps(events,ensure_ascii=False))
 row=dict(tag=tag,phase=phase,rep=rep,mode=mode,concurrency=c,sample_id=sample['sample_id'],status=status,error=error,success=error is None,usage=usage,client_start_s=start,client_end_s=end,e2e_ms=(end-start)*1000,ttft_ms=None if first is None else (first-start)*1000,tpot_ms=None if error else (end-first)*1000/(usage['completion_tokens']-1))
 append('requests.jsonl',row);print('REQUEST',json.dumps(row),flush=True);return row
async def batch(rt,session,samples,mode,phase,rep,c):
 resets=await b.clear(rt);b.setmode(mode)
 append('cache_resets.jsonl',dict(phase=phase,rep=rep,mode=mode,concurrency=c,resets=resets))
 cpu_before=b.cpu_quota_stats();log=R/'logs/p_default.log';before=log.read_text();routefile=R/'evidence/routes.jsonl';rstart=len(routefile.read_text().splitlines()) if routefile.exists() else 0
 active=0;maxactive=0;sem=asyncio.Semaphore(c)
 async def one(sample):
  nonlocal active,maxactive
  async with sem:
   active+=1;maxactive=max(active,maxactive)
   try:return await request(session,sample,mode,phase,rep,c)
   finally:active-=1
 begin=time.perf_counter();rows=await asyncio.gather(*(one(s) for s in samples));elapsed=time.perf_counter()-begin
 newlog=log.read_text()[len(before):];tag=f'{phase}_c{c}_r{rep}_{mode}'
 (R/'logs'/f'{tag}.p_excerpt.log').write_text(newlog)
 batches=[]
 for line in newlog.splitlines():
  if 'Prefill batch,' not in line:continue
  m=re.search(r'#new-token: (\d+), #cached-token: (\d+)',line)
  if m:batches.append(dict(new_tokens=int(m[1]),cached_tokens=int(m[2])))
 audit=[]
 if routefile.exists():audit=[json.loads(x) for x in routefile.read_text().splitlines()[rstart:]]
 success=[x for x in rows if x['success']]
 result=dict(tag=tag,phase=phase,concurrency=c,rep=rep,mode=mode,success=len(success),failed=len(rows)-len(success),observed_client_max_inflight=maxactive,elapsed_s=elapsed,requests_per_s=len(success)/elapsed,output_tokens_per_s=sum(x['usage']['completion_tokens'] for x in success)/elapsed,prefill_batches=batches,total_new_tokens=sum(x['new_tokens'] for x in batches),total_cached_tokens=sum(x['cached_tokens'] for x in batches),route_count=len(audit),selected_calls=sum(x['selected'] for x in audit),multi_sequence_calls=sum(x['sequences']>1 for x in audit),metrics={m:stats([x[m] for x in success]) for m in ['ttft_ms','tpot_ms','e2e_ms']} if success else {})
 result['cpu_quota_delta']={k:v-cpu_before.get(k,0) for k,v in b.cpu_quota_stats().items()};append('batches.jsonl',result);print('BATCH_COMPLETE',json.dumps(result),flush=True)
 assert result['failed']==0,result
 assert maxactive==c,(maxactive,c)
 return result
async def ready(session):
 for i in range(600):
  try:
   statuses=[]
   for port in [18081,18082]:
    async with session.get(f'http://127.0.0.1:{port}/health') as r:statuses.append(r.status)
   async with session.get('http://127.0.0.1:18000/v1/models') as r:data=await r.json()
   if statuses==[200,200] and data.get('data'):return
  except (aiohttp.ClientError,ValueError):pass
  if i%15==0:print('WAIT_READY',i,flush=True)
  await asyncio.sleep(2)
 raise RuntimeError('not ready')
async def main():
 assert not (R/'results/requests.jsonl').exists(),'new campaign only'
 rt=b.DistributedRuntime(asyncio.get_running_loop(),'etcd','tcp',event_plane='zmq');samples=json.loads((R/'config/fixed10.json').read_text())
 try:
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
   await ready(session)
   for c in [2,4,8]:await batch(rt,session,samples,'verify_multi','shadow',0,c)
   checks=[json.loads(x) for x in (R/'evidence/real_checks.jsonl').read_text().splitlines()]
   assert checks and all(x['valid'] and all(v==0 for v in x['errors'].values()) for x in checks)
   print('SHADOW_COMPLETE',len(checks),flush=True)
   for variant,mode,reps in [('chunk8k_a1','chunk8k',[1]),('chunk16k','chunk16k',[1,2]),('chunk8k_a2','chunk8k',[2])]:
    f.restart_p(variant);await ready(session)
    for c in [2,4,8]:await batch(rt,session,samples,mode,'warm_'+variant,0,c)
    for rep in reps:
     for c in [2,4,8]:await batch(rt,session,samples,mode,'formal',rep,c)
   print('COMPLETE_CONCURRENT_ABBA',flush=True)
 finally:rt.shutdown()
if __name__=='__main__':
 try:asyncio.run(main())
 except BaseException:
  (R/'results/failure.txt').write_text(traceback.format_exc());raise
 finally:subprocess.run(['/opt/sglang/bin/python',str(R/'scripts/07_stop.py')],check=False)
