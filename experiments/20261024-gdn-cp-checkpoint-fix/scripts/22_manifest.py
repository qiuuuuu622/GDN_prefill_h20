"""Hash report, source, raw results and logs; exclude transient caches and self-manifest."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parents[1]
files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(r.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','.DS_Store')}
(r/'evidence/manifest.json').write_text(json.dumps(files,indent=2)+'\n');print('HASHED',len(files))
