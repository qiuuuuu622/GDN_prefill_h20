import asyncio, importlib.util, json
import aiohttp
spec=importlib.util.spec_from_file_location('bench',__import__('pathlib').Path(__file__).with_name('04_benchmark.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
async def main():
    samples=json.loads((b.R/'config/fixed10.json').read_text())
    runtime=b.DistributedRuntime(asyncio.get_running_loop(),'etcd','tcp',event_plane='zmq')
    try:
      async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as s:
        for attempt in range(900):
          try:
            codes=[]
            for port in [18081,18082]:
              async with s.get(f'http://127.0.0.1:{port}/health') as r:codes.append(r.status)
            async with s.get('http://127.0.0.1:18000/v1/models') as r:models=await r.json()
            if codes==[200,200] and models.get('data'):break
          except (aiohttp.ClientError,ValueError):pass
          if attempt%15==0:print('WAIT_READY',attempt,flush=True)
          await asyncio.sleep(2)
        else:raise RuntimeError('not ready')
        await b.clear(runtime);b.setmode('audit')
        for x in samples:await b.request(s,x,'audit','audit',0)
        for mode in ['verify','verify_sparse']:
          await b.clear(runtime);b.setmode(mode)
          for x in samples:await b.request(s,x,mode,'real_verify',0)
        b.setmode('baseline');print('COMPLETE_AUDIT_VERIFY',flush=True)
    finally:runtime.shutdown()
if __name__=='__main__':asyncio.run(main())
