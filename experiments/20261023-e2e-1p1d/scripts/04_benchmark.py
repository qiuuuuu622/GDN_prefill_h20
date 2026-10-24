"""Fixed-ten streaming end-to-end A/B; sequential, cold cache per request."""
import asyncio,json,pathlib,time,hashlib,os,sys
import aiohttp
R=pathlib.Path('/root/gdn35/20261023-e2e-1p1d')
os.environ.update(DYN_NAMESPACE='gdn35-e2e-1023',ETCD_ENDPOINTS='http://127.0.0.1:12379',DYN_DISCOVERY_BACKEND='etcd',DYN_REQUEST_PLANE='tcp',DYN_EVENT_PLANE='zmq',DYN_SYSTEM_PORT='-1')
from dynamo.runtime import DistributedRuntime

def setmode(mode):
 p=R/'config/mode.tmp';p.write_text(mode+'\n');p.replace(R/'config/mode.txt')

async def clear(runtime):
 records=[]
 for comp in ['prefill','backend']:
  client=await runtime.endpoint(f'gdn35-e2e-1023.{comp}.clear_kv_blocks').client()
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

async def request(session,sample,mode,phase,rep):
 body=sample['payload'].copy();tag=f'{phase}_r{rep}_{mode}_{sample["sample_id"]}'
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
 row=dict(tag=tag,phase=phase,rep=rep,mode=mode,sample_id=sample['sample_id'],source_line=sample.get('source_line'),status=status,ttft_ms=(first-start)*1000,tpot_ms=(end-first)*1000/(n-1),e2e_ms=(end-start)*1000,usage=usage,finish_reasons=finish,stream_events=len(events),output_sha256=hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest())
 with (R/'results/requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print('REQUEST',json.dumps(row),flush=True)
 return row

async def main():
 phase=sys.argv[1] if len(sys.argv)>1 else 'smoke'
 samples=json.loads((R/'config/fixed10.json').read_text())
 runtime=DistributedRuntime(asyncio.get_running_loop(),'etcd','tcp',event_plane='zmq')
 try:
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
   if phase=='smoke':
    setmode('baseline');await clear(runtime)
    sample=dict(sample_id='smoke',payload=dict(model='gdn35-e2e',messages=[dict(role='user',content='请用两句话介绍杭州。')],temperature=0,max_tokens=64,seed=42,stream=True,stream_options={'include_usage':True}))
    await request(session,sample,'baseline','smoke',0)
   elif phase=='verify':
    setmode('verify');await clear(runtime);await request(session,samples[0],'verify','verify',0)
   elif phase=='bench':
    assert not any(json.loads(l)['phase']=='measured' for l in (R/'results/requests.jsonl').read_text().splitlines()),'Existing measured run: do not duplicate'
    for mode in ['baseline','optimized']:
     setmode(mode)
     for sample in samples:
      resets=await clear(runtime)
      with (R/'results/cache_resets.jsonl').open('a') as f:f.write(json.dumps(dict(phase='warmup',mode=mode,sample_id=sample['sample_id'],resets=resets))+'\n')
      await request(session,sample,mode,'warmup',0)
    for rep,order in enumerate([['baseline','optimized'],['optimized','baseline']],1):
     for mode in order:
      setmode(mode)
      for sample in samples:
       resets=await clear(runtime)
       with (R/'results/cache_resets.jsonl').open('a') as f:f.write(json.dumps(dict(phase='measured',rep=rep,mode=mode,sample_id=sample['sample_id'],resets=resets))+'\n')
       await request(session,sample,mode,'measured',rep)
      print('BATCH_COMPLETE',rep,mode,flush=True)
   print('COMPLETED',phase,flush=True)
 finally:runtime.shutdown()

if __name__=='__main__':asyncio.run(main())
