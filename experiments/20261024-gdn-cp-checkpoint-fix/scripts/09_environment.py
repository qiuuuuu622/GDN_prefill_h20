import json,pathlib,hashlib,importlib.metadata,platform,torch
from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm90,delta_rule_sm90
R=pathlib.Path(__file__).resolve().parents[1]
versions={}
for p in ['torch','flashinfer-python','sglang','nvidia-cutlass-dsl']:
 try:versions[p]=importlib.metadata.version(p)
 except importlib.metadata.PackageNotFoundError:versions[p]='unavailable'
files=[pathlib.Path(delta_rule_cp_sm90.__file__),pathlib.Path(delta_rule_sm90.__file__)]
result=dict(python=platform.python_version(),versions=versions,gpu=str(torch.cuda.get_device_properties(0)),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((R/'scripts').glob('*.py'))})
(R/'evidence/environment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
