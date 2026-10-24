"""Stop only process trees recorded by this experiment."""
import pathlib,psutil,signal,json,time
R=pathlib.Path('/root/gdn35/20261024-gdn-multiseq-opt');result=[]
for role in ['p','d','frontend','etcd']:
 pid=int((R/f'logs/{role}_default.pid').read_text())
 try:root=psutil.Process(pid)
 except psutil.NoSuchProcess:continue
 assert any(str(R) in a for a in root.cmdline()) or role=='frontend' and 'dynamo.frontend' in root.cmdline(),(pid,root.cmdline())
 procs=root.children(recursive=True)+[root]
 root.send_signal(signal.SIGTERM)
 gone,alive=psutil.wait_procs(procs,timeout=15)
 for p in alive:
  try:p.terminate()
  except psutil.NoSuchProcess:pass
 gone2,alive=psutil.wait_procs(alive,timeout=5)
 for p in alive:
  try:p.kill()
  except psutil.NoSuchProcess:pass
 result.append(dict(role=role,root_pid=pid,recorded_tree=[p.pid for p in procs],forced_kill=[p.pid for p in alive]))
(R/'evidence/cleanup.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
