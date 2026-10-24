"""CPU-only publication audit: provenance, syntax, generated data, links, payload exclusion."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
    for entry in manifest['files']:
        p=ROOT/entry['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['published_sha256'], f'Published hash mismatch: {p}'
    runtime=ROOT/'qwen35/runtime'
    archived=ROOT/'experiments/20261024-gdn-multiseq-opt/scripts'
    for p in runtime.glob('*.py'):
        assert p.read_bytes()==(archived/p.name).read_bytes(), f'Runtime drift from measured implementation: {p}'
    for p in ROOT.rglob('*.py'):
        if '.git' not in p.parts:ast.parse(p.read_text(),filename=str(p))
    # These checks deliberately do not import Torch/CUDA or execute archived launchers.
    subprocess.run([sys.executable,str(ROOT/'tools/regenerate_35b.py'),'--check'],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/build_tables.py'),'--check'],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/build_unified_tables.py'),'--check'],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/regenerate_kernels.py'),'--check'],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/build_entry_report.py'),'--check'],check=True)
    runtime_manifest=json.loads((ROOT/'experiments/20261024-unified-entry/evidence/runtime_source_manifest.json').read_text())
    for name,sha in runtime_manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha, f'Runtime drift from GPU-verified entry: {name}'
    new_docs=[ROOT/'README.md',ROOT/'qwen35/README.md',ROOT/'experiments/README.md']
    new_docs.extend(p for p in (ROOT/'docs').glob('*.md') if p.name!='122b-original-report.md')
    for p in new_docs:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in link or link.startswith('#'):continue
            assert (p.parent/link.split('#')[0]).exists(), f'Broken link in {p}: {link}'
    files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
    for p in files:
        assert p.name!='fixed10.json' and not p.name.endswith(('.sse.json','.pt','.trace.json','.trace.json.gz')), f'Private artifact: {p}'
        text=p.read_text()
        # No bearer values, GitHub access tokens, or OpenAI-style secret keys.
        for pattern in (r'gh[pousr]_[A-Za-z0-9]{20,}',r'Bearer\s+[A-Za-z0-9_-]{20,}',r'sk-[A-Za-z0-9_-]{30,}'):
            assert not re.search(pattern,text), f'Potential secret: {p}'
        if p.suffix not in ('.json','.jsonl'):continue
        values=[json.loads(line) for line in text.splitlines() if line.strip()] if p.suffix=='.jsonl' else [json.loads(text)]
        def scan(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key.lower() in ('messages','prompt','content','response','text','authorization','api_key','password'):
                        assert not isinstance(item,(str,list,dict)) or not item, f'Potential payload/secret field {key}: {p}'
                    scan(item)
            elif isinstance(value,list):
                for item in value:scan(item)
        for value in values:scan(value)
    print(f'PASS: {len(manifest["files"])} source hashes, runtime parity, syntax, generated tables, links and public data checks')

if __name__=='__main__':main()
