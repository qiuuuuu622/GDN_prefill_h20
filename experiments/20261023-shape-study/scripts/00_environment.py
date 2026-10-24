import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

root=pathlib.Path(__file__).resolve().parents[1]
model=json.loads(pathlib.Path('/prefill_model/config.json').read_text())
base=pathlib.Path('/opt/sglang/lib/python3.12/site-packages/flashinfer')
files=[base/'gdn_prefill.py',base/'utils.py',
       base/'gdn_kernels/delta_rule_dsl/delta_rule_sm90.py',
       base/'gdn_kernels/delta_rule_dsl/delta_rule_cp_sm90.py',
       base/'gdn_kernels/delta_rule_dsl/varlen_helper.py',
       pathlib.Path('/sgl-workspace/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py')]
data=dict(recorded_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
          target='h20-2',physical_gpu=4,python=sys.version,platform=platform.platform(),
          model='/prefill_model',tp=1,model_config=model,
          package_versions={n:importlib.metadata.version(n) for n in ['torch','sglang','flashinfer-python','nvidia-cutlass-dsl','transformers','triton']},
          source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
          sglang_commit=subprocess.check_output(['git','-C','/sgl-workspace/sglang','rev-parse','HEAD'],text=True).strip(),
          nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version,clocks.sm,clocks.mem,power.limit','--format=csv'],text=True),
          image_digest='not collected; existing managed environment')
(root/'evidence/environment.json').write_text(json.dumps(data,indent=2))
for p in files:
    (root/'evidence'/('source_'+p.name)).write_bytes(p.read_bytes())
print('Recorded environment and source snapshots; no package or service modifications.')
