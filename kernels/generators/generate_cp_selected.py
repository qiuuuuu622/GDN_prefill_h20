"""Generate native CP prefill checkpoint writes, retaining CP arithmetic.

Private combined state buffer: [final states for N sequences, compact checkpoints].
Plan contains interleaved [checkpoint slot, target token] for each sequence.
"""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'evidence/delta_rule_cp_sm90.py').read_text()
start=s.index('class CPDeltaRulePrefillSm90(');end=s.index('\ndef cp_delta_rule_dsl_sm90(',start)
prefix=s[:start];src=s[start:end]
def replace(a,b,n=1):
 global src
 assert src.count(a)==n,(a,src.count(a),n)
 src=src.replace(a,b)
replace('CPDeltaRulePrefillSm90','CPDeltaRulePrefillSelectedSm90',5)
replace('def cp_delta_rule_prefill_dsl_sm90(','def cp_delta_rule_prefill_selected_sm90(')
replace('    def run_cp_state_math_role(\n','''    def maybe_store_selected(
        self,
        g_state: cute.Tensor,
        g_checkpoint_plan: cute.Tensor,
        state_fragment: cute.Tensor,
        tiled_mma,
        math_tidx: cutlass.Int32,
        public_seq_idx: cutlass.Int32,
        block_end: cutlass.Int32,
        num_seqs: cutlass.Int32,
        num_heads: cutlass.Int32,
        head_idx: cutlass.Int32,
    ):
        target = cutlass.Int32(g_checkpoint_plan[public_seq_idx * 2 + 1])
        if target > 0 and block_end == target:
            slot = num_seqs + cutlass.Int32(g_checkpoint_plan[public_seq_idx * 2])
            total_slots = cute.size(g_state) // (self.D * self.D * num_heads)
            layout = cute.make_ordered_layout(
                (self.D, self.D, num_heads, total_slots), order=(0, 1, 2, 3)
            )
            states = cute.make_tensor(g_state.iterator, layout)
            self.kv_store(state_fragment, states[None, None, head_idx, slot], tiled_mma, math_tidx)

    @cute.jit
    def run_cp_state_math_role(
''')
# Three original signatures: state math, __call__, and CUDA kernel. Helper already includes plan.
replace('        g_state: cute.Tensor,\n        g_fixed_state: cute.Tensor,','        g_state: cute.Tensor,\n        g_checkpoint_plan: cute.Tensor,\n        g_fixed_state: cute.Tensor,',3)
replace('        public_seq_idx: cutlass.Int32,\n        fixed_state_idx: cutlass.Int32,','        public_seq_idx: cutlass.Int32,\n        cp_token_start: cutlass.Int32,\n        fixed_state_idx: cutlass.Int32,')
replace('            g_state,\n            g_fixed_state,','            g_state,\n            g_checkpoint_plan,\n            g_fixed_state,')
replace('                        g_state,\n                        g_fixed_state,','                        g_state,\n                        g_checkpoint_plan,\n                        g_fixed_state,')
replace('                        seq_idx,\n                        fixed_state_idx,','                        seq_idx,\n                        chunk_idx_in_seq * chunk_len,\n                        fixed_state_idx,')
replace('''        for _ in cutlass.range(1, num_blocks - 1, 1, unroll=1):''','''        self.maybe_store_selected(g_state, g_checkpoint_plan, tKVrKV, kv_tiled_mma,
                                  math_tidx, public_seq_idx, cp_token_start + first_B,
                                  num_seqs, num_sab_heads, o_head_idx)
        for block_idx in cutlass.range(1, num_blocks - 1, 1, unroll=1):''')
replace('''        if num_blocks != 1:
            last_blk = num_blocks - 1''','''            self.maybe_store_selected(g_state, g_checkpoint_plan, tKVrKV, kv_tiled_mma,
                                      math_tidx, public_seq_idx, cp_token_start + (block_idx + 1) * self.BLK_KV,
                                      num_seqs, num_sab_heads, o_head_idx)

        if num_blocks != 1:
            last_blk = num_blocks - 1''')
replace('''        if cutlass.const_expr(self.store_final_state):
            if store_state:''','''            self.maybe_store_selected(g_state, g_checkpoint_plan, tKVrKV, kv_tiled_mma,
                                      math_tidx, public_seq_idx, cp_token_start + work_desc.seq_len,
                                      num_seqs, num_sab_heads, o_head_idx)
        if cutlass.const_expr(self.store_final_state):
            if store_state:''')
replace('''    *,
    _skip_check: bool = False,''','''    *,
    checkpoint_plan: torch.Tensor,
    _skip_check: bool = False,''')
replace('''            from_dlpack(state_arg.reshape(-1), assumed_align=16).mark_layout_dynamic(),''','''            from_dlpack(state_arg.reshape(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(checkpoint_plan.reshape(-1), assumed_align=8).mark_layout_dynamic(),''')
replace('''        state_arg.reshape(-1),
        fixed_state.reshape(-1),''','''        state_arg.reshape(-1),
        checkpoint_plan.reshape(-1),
        fixed_state.reshape(-1),''')
replace('"gdn_cp_prefill_tensormaps"','"gdn_cp_selected_prefill_tensormaps"')
# Public CP shape validator expects N state rows; this private wrapper requires skip_check.
replace('''    max_cp_chunks_per_seq = max_num_chunks_host(max_seqlen, cp_chunk_len)''','''    if not _skip_check:
        raise ValueError("Private combined-state adapter requires validated _skip_check=True")
    max_cp_chunks_per_seq = max_num_chunks_host(max_seqlen, cp_chunk_len)''')
(R/'scripts/cp_selected_kernel.py').write_text('# Generated by generate_cp_selected.py; do not hand-edit.\n'+prefix+src)
print('Generated native CP selected checkpoint kernel')
