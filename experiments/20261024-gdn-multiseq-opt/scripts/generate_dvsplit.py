"""Generate an isolated experimental Dv=64 specialization from recorded source.

Physical [T, HV, 128] values become [T, HV*2, 64] views. Q/K and gates
are shared by paired virtual value heads; state/output writes are disjoint.
This reduces each CTA to one state warpgroup plus load/store and auxiliary
warpgroups, and removes the original two-state-warpgroup ordered barriers.
"""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
src = (root / "evidence/installed_delta_rule_sm90.py").read_text()

def replace(old, new, expected=None):
    global src
    n = src.count(old)
    assert n > 0 and (expected is None or n == expected), (old, n, expected)
    src = src.replace(old, new)

replace("_FullyFusedDeltaRuleSm90", "_DVSplitDeltaRuleSm90")
replace('"gdn_prefill_tensormaps"', '"gdn_dvsplit_prefill_tensormaps"', 1)
replace("    MATH_AUX = 3", "    MATH_AUX = 2", 1)
replace("        self.D = 128", "        self.D = 128\n        self.DV = 64", 1)
replace('            "D",', '            "D",\n            "DV",', 1)
replace("cute.make_layout((2, 1, 1))", "cute.make_layout((1, 1, 1))", 5)
replace("        d = cute.size(sQ_SD, mode=[1])", "        d = self.DV", 1)
replace("partition_shape_C((self.D, self.D))", "partition_shape_C((self.DV, self.D))", 1)
replace("(self.D, self.D, num_sab_heads,", "(self.D, self.DV, num_sab_heads,", 2)
replace("gV = cute.zipped_divide(mV, (self.D, self.BLK_KV))", "gV = cute.zipped_divide(mV, (self.DV, self.BLK_KV))", 1)
replace("(self.BLK_KV, self.D, self.v_stage)", "(self.BLK_KV, self.DV, self.v_stage)", 2)
replace("(self.D, self.BLK_Q, self.o_stage)", "(self.DV, self.BLK_Q, self.o_stage)", 2)
replace("tma_load_op, g_v, v_smem_layout, (self.D, self.BLK_KV)", "tma_load_op, g_v, v_smem_layout, (self.DV, self.BLK_KV)", 1)
replace("tma_store_op, g_o, o_smem_layout, (self.D, self.BLK_Q)", "tma_store_op, g_o, o_smem_layout, (self.DV, self.BLK_Q)", 1)
replace("CollectiveStoreTma(self.BLK_Q, self.D).run", "CollectiveStoreTma(self.BLK_Q, self.DV).run", 1)
replace("NUM_STATE_MMA_WARP_GROUPS = 2", "NUM_STATE_MMA_WARP_GROUPS = 1", 1)
replace("block=(512, 1, 1)", "block=(384, 1, 1)", 1)
replace("max_number_threads=(512, 1, 1)", "max_number_threads=(384, 1, 1)", 1)
replace("tok * num_sab_heads + sab_head_idx", "tok * (num_sab_heads // 2) + sab_head_idx // 2", 2)
src, n = re.subn(r"^\s*self\._math_order_(init|wait|notify)\(wg_idx\)\n", "\n", src, flags=re.M)
assert n == 11, n
replace("    D = q.shape[-1]", "    D = q.shape[-1]\n    DV = v.shape[-1]\n    assert D == 128 and DV == 64", 1)
replace("(D, total_seqlen, num_v_heads),\n        (1, num_v_heads * D, D)", "(DV, total_seqlen, num_v_heads),\n        (1, num_v_heads * DV, DV)", 1)
replace("(D, total_seqlen, num_o_heads),\n        (1, num_o_heads * D, D)", "(DV, total_seqlen, num_o_heads),\n        (1, num_o_heads * DV, DV)", 1)
# Private negative interval encodes one exact checkpoint token boundary.
# This stays inside the experiment adapter; FlashInfer public API never sees it.
replace("needs_checkpointing = checkpoint_every_n_tokens > 0", "needs_checkpointing = checkpoint_every_n_tokens != 0", 1)
replace("and block_end % checkpoint_every_n_tokens == cutlass.Int32(0)", "and ((checkpoint_every_n_tokens > 0 and block_end % checkpoint_every_n_tokens == cutlass.Int32(0)) or (checkpoint_every_n_tokens < 0 and block_end == -checkpoint_every_n_tokens))", 1)
replace("""                checkpoint_idx = (
                    cutlass.Int32(checkpoint_cu_starts[seq_idx])
                    + block_end // checkpoint_every_n_tokens
                    - cutlass.Int32(1)
                )""", """                checkpoint_idx = cutlass.Int32(checkpoint_cu_starts[seq_idx])
                if checkpoint_every_n_tokens > 0:
                    checkpoint_idx += block_end // checkpoint_every_n_tokens - cutlass.Int32(1)""", 1)
src = '# Experimental generated specialization. Regenerate with generate_dvsplit.py.\n' + src
target = root / "scripts/dvsplit_kernel.py"
target.write_text(src)
print(target)
