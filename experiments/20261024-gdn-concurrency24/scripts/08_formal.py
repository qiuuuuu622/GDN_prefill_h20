"""Stock vs fixed optimized deployment: AB/BA, warm shapes then flush each batch."""
import asyncio,importlib.util,json,pathlib,subprocess,signal,time,shutil,psutil
import aiohttp
R=pathlib.Path('/root/gdn35/20261024-gdn-concurrency24')
spec=importlib.util.spec_from_file_location('bench',R/'scripts/04_benchmark.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def restart_p(variant):
    pid=int((R/'logs/p_default.pid').read_text());root=psutil.Process(pid)
    assert str(R/'models/prefill_model') in root.cmdline(),root.cmdline()
    procs=root.children(recursive=True)+[root];root.send_signal(signal.SIGTERM)
    _,alive=psutil.wait_procs(procs,timeout=15)
    for p in alive:
        try:p.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(alive,timeout=5)
    for p in alive:
        try:p.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(alive,timeout=5)
    oldlog=R/'logs/p_default.log'
    if oldlog.is_symlink():oldlog.unlink()
    else:oldlog.rename(R/'logs/p_audit.log')
    (R/'logs/p_default.pid').unlink()
    subprocess.run(['/opt/sglang/bin/python',str(R/'scripts/03_launch.py'),'p',variant],check=True)
    oldlog.symlink_to(R/f'logs/p_{variant}.log')
    shutil.copy2(R/f'logs/p_{variant}.pid',R/'logs/p_default.pid')
    with (R/'evidence/restarts.jsonl').open('a') as f:f.write(json.dumps(dict(old_pid=pid,variant=variant,old_tree=[p.pid for p in procs],forced=[p.pid for p in alive]))+'\n')

async def ready(session):
    for i in range(600):
        try:
            async with session.get('http://127.0.0.1:18081/health') as r:
                if r.status==200:return
        except aiohttp.ClientError:pass
        if i%15==0:print('WAIT_P',i,flush=True)
        await asyncio.sleep(2)
    raise RuntimeError('P not ready')

async def main():
    rt=b.DistributedRuntime(asyncio.get_running_loop(),'etcd','tcp',event_plane='zmq');samples=json.loads((R/'config/fixed10.json').read_text())
    try:
      async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        for variant,mode,reps in [('stock_a1','baseline',[1]),('sparse_fixed','sparse',[1,2]),('stock_a2','baseline',[2])]:
            restart_p(variant);await ready(session);b.setmode(mode)
            await b.clear(rt)
            for sample in samples:await b.request(session,sample,mode,'formal_shape_warm_'+variant,0)
            for rep in reps:
                resets=await b.clear(rt)
                with (R/'results/cache_resets.jsonl').open('a') as f:f.write(json.dumps(dict(phase='formal_stock_abba',variant=variant,rep=rep,mode=mode,resets=resets))+'\n')
                rows=[]
                for sample in samples:rows.append(await b.request(session,sample,mode,'formal_stock_abba',rep))
                print('BATCH_COMPLETE',json.dumps(dict(variant=variant,mode=mode,rep=rep,count=len(rows),mean_ttft_ms=sum(x['ttft_ms'] for x in rows)/len(rows))),flush=True)
        print('COMPLETE_FORMAL_STOCK_ABBA',flush=True)
    finally:rt.shutdown()
if __name__=='__main__':asyncio.run(main())
