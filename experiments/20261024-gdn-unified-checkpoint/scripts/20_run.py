"""Sequential owned-process runner; no serving launch/stop operations."""
import json,os,pathlib,subprocess,time
R=pathlib.Path(__file__).resolve().parents[1]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',PYTHONUNBUFFERED='1')
phases=[('correctness',['/opt/sglang/bin/python','-u','scripts/01_experiment.py','--mode','correctness']),('benchmark',['/opt/sglang/bin/python','-u','scripts/01_experiment.py','--mode','benchmark','--repeats','7','--replays','50'])]
for name,cmd in phases:
 with (R/'logs'/f'{name}.log').open('w') as out:
  p=subprocess.Popen(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT)
  (R/'config'/f'{name}_command.json').write_text(json.dumps(dict(command=cmd,CUDA_VISIBLE_DEVICES='6',pid=p.pid,started=time.time()),indent=2))
  print(name,'PID',p.pid,flush=True);code=p.wait()
  (R/'results'/f'{name}_status.json').write_text(json.dumps(dict(exit_code=code,ended=time.time())))
  print(name,'EXIT',code,flush=True)
  if code:raise SystemExit(code)
