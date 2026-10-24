"""Read-only evidence audit and final local packaging."""
import hashlib,json,pathlib,tarfile,datetime
R=pathlib.Path(__file__).resolve().parents[1]
def load(p):return json.loads((R/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a,b,c=[load('config/'+n+'.json') for n in ['p_stock_a1','p_sparse_fixed','p_stock_a2']]
    assert a['command']==b['command']==c['command']
    diffs={k:[a['env'].get(k),b['env'].get(k)] for k in a['env'].keys()|b['env'].keys() if a['env'].get(k)!=b['env'].get(k)}
    assert diffs=={'GDN35_PATCH':[None,'1'],'GDN35_VARIANT':[None,'sparse']},diffs
    assert a['env']==c['env']
    for n,expected in [('p_stock_a1',False),('p_sparse_fixed',True),('p_stock_a2',False)]:
        assert ('GDN35_CHECKPOINT_PATCH_INSTALLED' in (R/'logs'/f'{n}.log').read_text())==expected
    hashes=load('evidence/runtime_hashes.json')
    for n,h in hashes.items():assert sha(R/'release'/n)==sha(R/'scripts'/n)==h,n
    assert sha(R/'config/fixed10.json')==load('evidence/environment.json')['fixed10_sha256']
    assert load('results/formal_summary.json')['requests']==40
    assert len(load('results/fallback_checks.json'))==3 and all(x['valid'] for x in load('results/fallback_checks.json'))
    checks=[json.loads(x) for x in (R/'evidence/real_checks.jsonl').read_text().splitlines()]
    assert len(checks)==720 and all(x['valid'] and all(v==0 for v in x['errors'].values()) for x in checks)
    audit={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formal_requests':40,'live_shadow_calls_exact':720,'fallback_cases':3,'p_commands_identical':True,'p_environment_diff':diffs,'stock_hook_absent':True,'release_runtime_hashes_match_measured':True,'status':'passed'}
    (R/'evidence/final_audit.json').write_text(json.dumps(audit,indent=2))
    with tarfile.open(R/'gdn35-checkpoint-runtime.tar.gz','w:gz') as t:
        for p in ['README.md','release','scripts/generate_dvsplit.py','evidence/runtime_hashes.json','evidence/environment.json','evidence/kernel_changes.diff','results/formal_summary.json']:
            t.add(R/p,arcname=p)
    files=[p for folder in ['scripts','release','config','results','profiles','evidence','logs'] for p in (R/folder).rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
    files += [R/'README.md',R/'WORKLOG.md',R/'gdn35-checkpoint-runtime.tar.gz']
    (R/'evidence/artifact_manifest.json').write_text(json.dumps({str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)},indent=2))
    print(json.dumps(audit,indent=2));print('Packaged',len(files),'artifacts')
if __name__=='__main__':main()
