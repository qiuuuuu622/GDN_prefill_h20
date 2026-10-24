import asyncio,importlib.util,json,pathlib,aiohttp
R=pathlib.Path('/root/gdn35/20261024-gdn-ttft-investigation')
spec=importlib.util.spec_from_file_location('bench',R/'scripts/04_benchmark.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
async def main():
 rt=b.DistributedRuntime(asyncio.get_running_loop(),'etcd','tcp',event_plane='zmq')
 samples=json.loads((R/'config/fixed10.json').read_text())
 try:
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
   for attempt in range(600):
    try:
     async with session.get('http://127.0.0.1:18081/health') as resp:ready=resp.status==200
     if ready:break
    except aiohttp.ClientError:pass
    if attempt%15==0:print('WAIT_READY',attempt,flush=True)
    await asyncio.sleep(2)
   else:raise RuntimeError('P not ready')
   for mode in ['baseline','optimized']:
    await b.clear(rt);b.setmode(mode)
    for sample in samples:await b.request(session,sample,mode,'all_shape_warmup',0)
   for rep,mode in [(1,'baseline'),(1,'optimized'),(2,'optimized'),(2,'baseline')]:
    resets=await b.clear(rt)
    with (R/'results/all_shape_resets.jsonl').open('a') as f:f.write(json.dumps(dict(rep=rep,mode=mode,resets=resets))+'\n')
    b.setmode(mode)
    for sample in samples:await b.request(session,sample,mode,'all_shape_warmed_fullbatch',rep)
   print('COMPLETED',flush=True)
 finally:rt.shutdown()
if __name__=='__main__':asyncio.run(main())
