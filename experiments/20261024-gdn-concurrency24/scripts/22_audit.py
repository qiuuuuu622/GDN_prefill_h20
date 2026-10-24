"""Audit measured source, payload, commands, and formal concurrent batches."""
import json,pathlib,hashlib
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads((R/p).read_text())
def main():
 configs=[load(f'config/p_{v}.json') for v in ['stock_a1','sparse_fixed','stock_a2']]
 assert configs[0]['command']==configs[1]['command']==configs[2]['command']
 assert configs[0]['env']==configs[2]['env']
 a,b=[x['env'] for x in configs[:2]];diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
 assert diff=={'GDN35_PATCH':[None,'1'],'GDN35_VARIANT':[None,'sparse']}
 for tag,expected in [('stock_a1',False),('stock_a2',False),('sparse_fixed',True)]:assert ('GDN35_CHECKPOINT_PATCH_INSTALLED' in (R/f'logs/p_{tag}.log').read_text())==expected
 assert sha(R/'config/fixed10.json')=='d5e8c10e433ea5952eaba44346489dd9161637bd1ed94060f45347f32717b30e'
 old=R.parent/'20261024-gdn-checkpoint-opt'
 names=['sitecustomize.py','service_patch.py','dvsplit.py','dvsplit_kernel.py']
 for n in names:
  original=(old/'scripts'/n).read_text().replace('/root/gdn35/20261024-gdn-checkpoint-opt','/root/gdn35/20261024-gdn-concurrency24')
  assert (R/'scripts'/n).read_text()==original,n
 bb=[json.loads(x) for x in (R/'results/batches.jsonl').read_text().splitlines()]
 for c in [2,4]:
  formal=[x for x in bb if x['phase']=='formal' and x['concurrency']==c]
  assert [(x['mode'],x['rep']) for x in formal]==[('baseline',1),('sparse',1),('sparse',2),('baseline',2)]
  assert all(x['success']==10 and x['failed']==0 and x['observed_client_max_inflight']==c for x in formal)
 summary=load('results/summary.json');assert summary['shadow_exact'] and summary['formal_requests']==80
 assert not (R/'results/failure.txt').exists()
 result={'status':'passed','same_p_commands':True,'environment_difference':diff,'stock_hook_absent':True,'formal_runtime_matches_previous_except_root':True,'dataset_sha256':sha(R/'config/fixed10.json'),'runtime_sha256':{n:sha(R/'scripts'/n) for n in names},'formal_requests':80,'all_requests':summary['successes'],'shadow_exact_calls':summary['shadow_checks'],'ordering':'A1 B1 B2 A2 for each concurrency','observed_client_concurrency':[2,4]}
 (R/'evidence/final_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for d in ['scripts','audit_scripts','config','results','evidence','logs'] for p in sorted((R/d).rglob('*')) if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json'}
 manifest['README.md']={'bytes':(R/'README.md').stat().st_size,'sha256':sha(R/'README.md')}
 (R/'evidence/artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
