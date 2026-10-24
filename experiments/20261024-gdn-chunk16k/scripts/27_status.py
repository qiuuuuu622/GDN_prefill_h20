import pathlib,json,re
r=pathlib.Path(__file__).resolve().parents[1]
for f in ['results/batches.jsonl','results/failure.txt']:
 p=r/f
 if p.exists():
  if f.endswith('.jsonl'):
   rows=[json.loads(x) for x in p.read_text().splitlines()]
   for x in rows[-4:]:print(x['tag'],x['success'],round(x['metrics']['ttft_ms']['mean'],1),round(x['requests_per_s'],3),x['total_new_tokens'])
  else:print(p.read_text()[-2000:])
p=r/'logs/concurrent.log'
if p.exists():
 lines=p.read_text().splitlines();stages=[x for x in lines if x.startswith(('WAIT_READY','SHADOW_COMPLETE','COMPLETE_CONCURRENT'))];print('stage',stages[-1:] if stages else '')
p=r/'logs/p_default.log'
if p.exists():
 lines=p.read_text().splitlines();clean=[x for x in lines if 'server_args=' not in x and '503 Service Unavailable' not in x and x.strip() and not x.startswith('\x1b[A')];print('P',p.resolve().name,'\n'.join(clean[-2:])[-700:])
p=r/'evidence/real_checks.jsonl'
if p.exists():
 rr=[json.loads(x) for x in p.read_text().splitlines()];print('shadow',len(rr),'valid',sum(x['valid'] for x in rr))
