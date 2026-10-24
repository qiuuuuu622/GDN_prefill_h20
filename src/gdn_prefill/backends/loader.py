"""Load private CuTe modules under FlashInfer for their relative imports."""
from functools import cache
import importlib.util
from pathlib import Path
import sys


@cache
def load(kind):
    filename = {"full": "selected_kernel", "v2": "dvsplit_kernel", "cp_checkpoint": "cp_selected_kernel"}[kind]
    name = f"flashinfer.gdn_kernels.delta_rule_dsl.gdn_prefill_h20_{filename}"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "_kernels" / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
