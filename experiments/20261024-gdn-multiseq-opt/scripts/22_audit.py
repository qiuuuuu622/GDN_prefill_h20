"""Audit measured source, payload, commands, and formal concurrent batches."""
import json,pathlib,hashlib
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads((R/p).read_text())
def main():
 configs=[load(f'config/p_{v}.json') for v in ['stock_a1','multi_fixed','stock_a2']]
 assert configs[0]['command']==configs[1]['command']==configs[2]['command']
 assert configs[0]['env']==configs[2]['env']
 a,b=[x['env'] for x in configs[:2]];diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
 assert diff=={'GDN35_PATCH':[None,'1'],'GDN35_VARIANT':[None,'multi']}
 for tag,expected in [('stock_a1',False),('stock_a2',False),('multi_fixed',True)]:assert ('GDN35_MULTI_CHECKPOINT_PATCH_INSTALLED' in (R/f'logs/p_{tag}.log').read_text())==expected
 assert sha(R/'config/fixed10.json')=='d5e8c10e433ea5952eaba44346489dd9161637bd1ed94060f45347f32717b30e'
 hashes=load('evidence/runtime_hashes_at_launch.json')
 for n,h in hashes.items():assert sha(R/'scripts'/n)==h,n
 names=list(hashes)
 bb=[json.loads(x) for x in (R/'results/batches.jsonl').read_text().splitlines()]
 for c in [2,4,8]:
  formal=[x for x in bb if x['phase']=='formal' and x['concurrency']==c]
  assert [(x['mode'],x['rep']) for x in formal]==[('baseline',1),('multi',1),('multi',2),('baseline',2)]
  assert all(x['success']==10 and x['failed']==0 and x['observed_client_max_inflight']==c for x in formal)
 summary=load('results/summary.json');assert summary['shadow_exact'] and summary['formal_requests']==120
 assert len(load('results/multi_checks.json'))==72 and all(x['valid'] for x in load('results/multi_checks.json'))
 assert load('results/planner_checks.json')['planner_mapping_cases']==277 and load('results/planner_checks.json')['planner_all_valid']
 assert len(load('results/multi_memcheck.json'))==8 and 'ERROR SUMMARY: 0 errors' in (R/'logs/multi_memcheck.log').read_text()
 assert len(load('results/dispatch_sweep.json'))==19 and len(load('results/real_replay.json'))==6
 assert not (R/'results/failure.txt').exists()
 result={'status':'passed','same_p_commands':True,'environment_difference':diff,'stock_hook_absent':True,'runtime_matches_validated_launch_hashes':True,'dataset_sha256':sha(R/'config/fixed10.json'),'runtime_sha256':{n:sha(R/'scripts'/n) for n in names},'formal_requests':120,'all_requests':summary['successes'],'shadow_exact_calls':summary['shadow_checks'],'ordering':'A1 B1 B2 A2 for each concurrency','observed_client_concurrency':[2,4,8]}
 (R/'evidence/final_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for d in ['scripts','audit_scripts','config','results','evidence','logs','release'] for p in sorted((R/d).rglob('*')) if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json'}
 manifest['gdn35-multiseq-runtime.tar.gz']={'bytes':(R/'gdn35-multiseq-runtime.tar.gz').stat().st_size,'sha256':sha(R/'gdn35-multiseq-runtime.tar.gz')}
 manifest['README.md']={'bytes':(R/'README.md').stat().st_size,'sha256':sha(R/'README.md')}
 (R/'evidence/artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
