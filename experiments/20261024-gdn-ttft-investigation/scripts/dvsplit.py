"""Local-only adapter; no changes to FlashInfer's public dispatcher."""
import importlib.util
import pathlib
import sys

name = "flashinfer.gdn_kernels.delta_rule_dsl.local_dvsplit_sm90"
spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name("dvsplit_kernel.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)


def run(*, q, k, v, g, beta, cu_seqlens, initial_state, output_state,
        output, max_seqlen=None):
    assert q.shape[1] == k.shape[1] and v.shape[1] >= q.shape[1]
    t, hv, dv = v.shape
    assert dv == 128
    return module.delta_rule_prefill_dsl_sm90(
        output.view(t, hv * 2, 64),
        output_state.view(-1, hv * 2, 64, 128),
        q, k, v.view(t, hv * 2, 64),
        None if initial_state is None else initial_state.view(-1, hv * 2, 64, 128),
        g, beta, cu_seqlens, 128 ** -0.5,
    )
