import os,pathlib,subprocess,json,sys
R=pathlib.Path('/root/gdn35/20261024-gdn-concurrency24');PY='/opt/sglang/bin/python'
(R/'topology').mkdir(exist_ok=True)
(R/'topology/host').write_text(__import__('socket').gethostname()+'\n')
role=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'default'
env=os.environ.copy();env.pop('GDN35_PATCH',None);env.pop('GDN35_VARIANT',None);env.update(DYN_NAMESPACE='gdn35-concurrency-1024',ETCD_ENDPOINTS='http://127.0.0.1:12379',DYN_DISCOVERY_BACKEND='etcd',DYN_REQUEST_PLANE='tcp',DYN_EVENT_PLANE='zmq',DYN_TOPOLOGY_ENABLED='true',DYN_TOPOLOGY_MOUNT_PATH=str(R/'topology'),DYN_KV_TRANSFER_DOMAIN='host',DYN_KV_TRANSFER_ENFORCEMENT='required',UCX_CUDA_IPC_CACHE='n',DYN_SYSTEM_STARTING_HEALTH_STATUS='notready',DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS='["generate"]',PYTHONPATH=str(R/'scripts'),PYTHONUNBUFFERED='1')
if role=='etcd':
 cmd=[str(R/'bin/etcd'),'--name','gdn35','--data-dir',str(R/'etcd-data'),'--listen-client-urls','http://127.0.0.1:12379','--advertise-client-urls','http://127.0.0.1:12379','--listen-peer-urls','http://127.0.0.1:12380','--initial-advertise-peer-urls','http://127.0.0.1:12380','--initial-cluster','gdn35=http://127.0.0.1:12380']
elif role=='frontend':
 env.update(CUDA_VISIBLE_DEVICES='',DYN_SYSTEM_PORT='18080');cmd=[PY,'-m','dynamo.frontend','--http-host','0.0.0.0','--http-port','18000','--router-mode','round-robin','--enforce-disagg']
else:
 assert role in ('p','d')
 env.update(CUDA_VISIBLE_DEVICES='6' if role=='p' else '7',DYN_SYSTEM_PORT='18081' if role=='p' else '18082')
 cmd=[PY,'-m','dynamo.sglang','--context-length','262144','--host','0.0.0.0','--tensor-parallel-size','1','--random-seed','42','--mem-fraction-static','0.80','--served-model-name','gdn35-e2e','--limit-mm-data-per-request','{"video":0}','--mm-process-config','{"image":{"min_pixels":3136,"max_pixels":2097152}}','--enable-multimodal','--page-size','16','--max-running-requests','16','--dyn-tool-call-parser','qwen3_coder','--dyn-reasoning-parser','deepseek_r1','--enable-metrics','--speculative-algorithm','DFLASH','--speculative-num-draft-tokens','8','--speculative-draft-model-path',str(R/'models/spec_model'),'--disaggregation-transfer-backend','nixl','--model-path',str(R/('models/prefill_model' if role=='p' else 'models/decode_model')),'--disaggregation-mode','prefill' if role=='p' else 'decode','--nccl-port','19506' if role=='p' else '19507','--kv-events-config','{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5560'+('6' if role=='p' else '7')+'"}']
 if role=='p' and not variant.startswith('stock'):env['GDN35_PATCH']='1'
 if role=='p' and variant=='sparse_fixed':env['GDN35_VARIANT']='sparse'
 if role=='p' and variant=='audit':env['PYTHONPATH']=str(R/'audit_scripts')
 if role=='p' and variant=='flashinfer':
  cmd+=['--linear-attn-prefill-backend','flashinfer'];env['GDN35_PATCH']='1'
key=f'{role}_{variant}';log=R/'logs'/f'{key}.log'
assert not (R/'logs'/f'{key}.pid').exists(),'PID record exists; check and archive before launching'
with log.open('w') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
(R/'logs'/f'{key}.pid').write_text(str(p.pid))
(R/'config'/f'{key}.json').write_text(json.dumps(dict(command=cmd,env={k:v for k,v in env.items() if k.startswith(('DYN_','ETCD_','UCX_','GDN35_','CUDA_VISIBLE','PYTHONPATH'))},pid=p.pid),indent=2))
print(json.dumps(dict(role=role,variant=variant,pid=p.pid,log=str(log),cmd=cmd)))
