"""Fixed-ten streaming end-to-end A/B; sequential, reset once per batch then preserve cache between requests."""
import asyncio,json,pathlib,time,hashlib,os,sys
import aiohttp
import re

def cpu_quota_stats():
 p=pathlib.Path("/sys/fs/cgroup/cpu/cpu.stat")
 return {k:int(v) for k,v in (line.split() for line in p.read_text().splitlines())} if p.exists() else {}
R=pathlib.Path('/root/gdn35/20261024-gdn-concurrency24')
os.environ.update(DYN_NAMESPACE='gdn35-concurrency-1024',ETCD_ENDPOINTS='http://127.0.0.1:12379',DYN_DISCOVERY_BACKEND='etcd',DYN_REQUEST_PLANE='tcp',DYN_EVENT_PLANE='zmq',DYN_SYSTEM_PORT='-1')
from dynamo.runtime import DistributedRuntime

def setmode(mode):
 p=R/'config/mode.tmp';p.write_text(mode+'\n');p.replace(R/'config/mode.txt')

async def clear(runtime):
 records=[]
 for comp in ['prefill','backend']:
  client=await runtime.endpoint(f'gdn35-concurrency-1024.{comp}.clear_kv_blocks').client()
  ids=list(client.instance_ids())
  for _ in range(20):
   if len(ids)==1:break
   await asyncio.sleep(.25);ids=list(client.instance_ids())
  assert len(ids)==1,(comp,ids)
  stream=await client.direct('{}',ids[0]);responses=[]
  async for item in stream:
   raw=item.data();v=json.loads(raw) if isinstance(raw,str) else raw;responses.append(v)
  assert responses and all(isinstance(v,dict) and v.get('status')=='success' for v in responses),(comp,responses)
  records.append(dict(component=comp,responses=responses))
 return records

def route_counts(text):
 counts={}
 for line in text.splitlines():
  if 'GDN35_ROUTE ' in line:
   x=json.loads(line.split('GDN35_ROUTE ',1)[1]);counts[f"{x['mode']}:{x['selected']}"]=x['count']
 return counts

async def request(session,sample,mode,phase,rep):
 body=sample['payload'].copy();tag=f'{phase}_r{rep}_{mode}_{sample["sample_id"]}'
 before_log=(R/'logs/p_default.log').read_text();before_counts=route_counts(before_log)
 cpu_before=cpu_quota_stats()
 start=time.perf_counter();first=None;usage=None;events=[];done=False;finish=[];errors=[]
 async with session.post('http://127.0.0.1:18000/v1/chat/completions',json=body) as response:
  status=response.status
  if status!=200:raise RuntimeError(f'HTTP {status}: {(await response.text())[:1000]}')
  async for raw in response.content:
   line=raw.decode().strip()
   if not line.startswith('data:'):continue
   data=line[5:].strip();now=time.perf_counter()
   if data=='[DONE]':done=True;break
   x=json.loads(data);events.append(dict(elapsed_s=now-start,data=x))
   if x.get('error'):errors.append(x['error'])
   if x.get('usage'):usage=x['usage']
   for c in x.get('choices',[]):
    delta=c.get('delta',{})
    meaningful=bool(delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning') or any(t.get('function',{}).get('arguments') or t.get('function',{}).get('name') for t in delta.get('tool_calls',[])))
    if meaningful and first is None:first=now
    if c.get('finish_reason'):finish.append(c['finish_reason'])
 end=time.perf_counter()
 (R/'results'/f'{tag}.sse.json').write_text(json.dumps(events,ensure_ascii=False))
 assert done and first is not None and not errors and usage,(tag,done,first,usage,errors)
 n=usage['completion_tokens'];assert n>1,(tag,n)
 parts=[c.get('delta',{}) for e in events for c in e['data'].get('choices',[])]
 cpu_after=cpu_quota_stats()
 row=dict(cpu_throttle_delta={k:cpu_after[k]-cpu_before.get(k,0) for k in cpu_after},client_start_monotonic_s=start,client_end_monotonic_s=end,tag=tag,phase=phase,rep=rep,mode=mode,sample_id=sample['sample_id'],source_line=sample.get('source_line'),status=status,ttft_ms=(first-start)*1000,tpot_ms=(end-first)*1000/(n-1),e2e_ms=(end-start)*1000,usage=usage,finish_reasons=finish,stream_events=len(events),output_sha256=hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest())
 after_log=(R/'logs/p_default.log').read_text();after_counts=route_counts(after_log)
 row['gdn_route_delta']={k:after_counts[k]-before_counts.get(k,0) for k in after_counts if after_counts[k]!=before_counts.get(k,0)}
 batch_lines=[l for l in after_log[len(before_log):].splitlines() if 'Prefill batch,' in l]
 row['prefill_batches']=[]
 for line in batch_lines:
  match=re.search(r'#new-token: (\d+), #cached-token: (\d+)',line)
  if match:row['prefill_batches'].append(dict(new_tokens=int(match[1]),cached_tokens=int(match[2])))
 with (R/'results/requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print('REQUEST',json.dumps(row),flush=True)
 return row

async def main():
 samples=json.loads((R/'config/fixed10.json').read_text())
 assert not (R/'results/requests.jsonl').exists(),'Do not overwrite an existing run'
 runtime=DistributedRuntime(asyncio.get_running_loop(),'etcd','tcp',event_plane='zmq')
 try:
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
   # Startup readiness. No user requests enter the measurement until both workers are ready.
   for attempt in range(600):
    try:
     codes=[]
     for port in [18081,18082]:
      async with session.get(f'http://127.0.0.1:{port}/health') as response:codes.append(response.status)
     async with session.get('http://127.0.0.1:18000/v1/models') as response:models=await response.json()
     if codes==[200,200] and models.get('data'):break
    except (aiohttp.ClientError,ValueError):pass
    if attempt%15==0:print('WAIT_READY',attempt,flush=True)
    await asyncio.sleep(2)
   else:raise RuntimeError('Service not ready after 1200 seconds')
   print('READY',flush=True)
   for mode in ['baseline','optimized']:
    setmode(mode)
    for sample in [samples[0],samples[-1]]:
     await request(session,sample,mode,'jit_warmup',0)
   for rep,mode in [(1,'baseline'),(1,'optimized'),(2,'optimized'),(2,'baseline')]:
    # Clear once at batch start. No replay of test prompts before measurement.
    resets=await clear(runtime)
    with (R/'results/cache_resets.jsonl').open('a') as f:f.write(json.dumps(dict(phase='before_measured_batch',rep=rep,target_mode=mode,resets=resets))+'\n')
    setmode(mode)
    for sample in samples:await request(session,sample,mode,'measured',rep)
    print('BATCH_COMPLETE',rep,mode,flush=True)
   print('COMPLETED',flush=True)
 finally:runtime.shutdown()

if __name__=='__main__':asyncio.run(main())
