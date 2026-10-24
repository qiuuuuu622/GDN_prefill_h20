import asyncio,importlib.util,json,pathlib,aiohttp,time
R=pathlib.Path('/root/gdn35/20261024-gdn-ttft-investigation')
spec=importlib.util.spec_from_file_location('bench',R/'scripts/04_benchmark.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
async def profile(session,action,tag):
 body=dict(output_dir=str(R/'profiles'/tag),with_stack=True,record_shapes=True,activities=['CPU','GPU'],profile_by_stage=False,profile_prefix=tag,merge_profiles=False,detailed_annotations=True) if action=='start' else {}
 async with session.post(f'http://127.0.0.1:18081/engine/control/{action}_profile',json=body) as resp:
  text=await resp.text();print('PROFILE_CONTROL',action,resp.status,text,flush=True);assert resp.status==200,text
  result=json.loads(text);assert result.get('status')=='ok',result
async def main():
 samples=json.loads((R/'config/fixed10.json').read_text())
 rt=b.DistributedRuntime(asyncio.get_running_loop(),'etcd','tcp',event_plane='zmq')
 try:
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:
   for step in range(600):
    try:
     codes=[]
     for port in [18081,18082]:
      async with session.get(f'http://127.0.0.1:{port}/health') as resp:codes.append(resp.status)
     async with session.get('http://127.0.0.1:18000/v1/models') as resp:models=await resp.json()
     if codes==[200,200] and models.get('data'):break
    except (aiohttp.ClientError,ValueError):pass
    if step%15==0:print('WAIT_READY',step,flush=True)
    await asyncio.sleep(2)
   else:raise RuntimeError('not ready')
   for mode in ['baseline','optimized']:
    await b.clear(rt);b.setmode(mode)
    for sample in [samples[0],samples[9]]:await b.request(session,sample,mode,'jit_full',0)
   for rep,mode in [(1,'baseline'),(1,'optimized'),(2,'optimized'),(2,'baseline')]:
    await b.clear(rt);b.setmode('baseline')
    await b.request(session,samples[0],'baseline','history_'+mode,rep)
    b.setmode(mode);await b.request(session,samples[1],mode,'diagnostic_unprofiled',rep)
   for target in [1,3]:
    for mode in ['baseline','optimized']:
     tag=f'sample{target:02d}_{mode}'
     await b.clear(rt);b.setmode('baseline')
     for sample in samples[:target]:await b.request(session,sample,'baseline','profile_history_'+tag,0)
     b.setmode(mode);await profile(session,'start',tag)
     try:await b.request(session,samples[target],mode,'profile_'+tag,0)
     finally:await profile(session,'stop',tag)
     print('TRACE_COMPLETE',tag,flush=True)
   print('COMPLETED',flush=True)
 finally:rt.shutdown()
if __name__=='__main__':asyncio.run(main())
