import asyncio,importlib.util,json,pathlib,aiohttp
R=pathlib.Path('/root/gdn35/20261024-gdn-ttft-investigation')
spec=importlib.util.spec_from_file_location('bench',R/'scripts/04_benchmark.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
async def main():
 rt=b.DistributedRuntime(asyncio.get_running_loop(),'etcd','tcp',event_plane='zmq');samples=json.loads((R/'config/fixed10.json').read_text())
 try:
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:
   for mode in ['baseline','optimized']:
    await b.clear(rt);b.setmode(mode);tag='batch_'+mode
    async with session.post('http://127.0.0.1:18081/engine/control/start_profile',json=dict(output_dir=str(R/'profiles'/tag),with_stack=False,record_shapes=False,activities=['CPU','GPU'],profile_by_stage=False,profile_prefix=tag,detailed_annotations=True)) as resp:
     text=await resp.text();print('START',resp.status,text,flush=True);assert resp.status==200
    try:
     for sample in samples:await b.request(session,sample,mode,'batch_profile',0)
    finally:
     async with session.post('http://127.0.0.1:18081/engine/control/stop_profile',json={}) as resp:
      print('STOP',resp.status,await resp.text(),flush=True);assert resp.status==200
   print('COMPLETED',flush=True)
 finally:rt.shutdown()
if __name__=='__main__':asyncio.run(main())
