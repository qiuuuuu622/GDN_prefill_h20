"""Fail before loading private kernels when the measured upstream implementation changed."""
from functools import cache
import hashlib
from pathlib import Path

EXPECTED = {'delta_rule_sm90.py': 'c72398f8f6e57ed2d708d364dd0299ce414e75a27d044ac83c6278c0954290a4', 'delta_rule_cp_sm90.py': '4b137abe0b810b9f74c85642fc10793dfb7156a934e00a8367857fe859385e7d'}

@cache
def verify_private_sources():
    from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_sm90, delta_rule_cp_sm90
    for module in (delta_rule_sm90, delta_rule_cp_sm90):
        path = Path(module.__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED[path.name]:
            raise RuntimeError(f"Unvalidated private FlashInfer source: {path.name}; use backend='original' or validate this baseline first")
