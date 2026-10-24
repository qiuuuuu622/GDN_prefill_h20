"""Full-V selected-checkpoint local kernel adapter."""
import importlib.util,pathlib,sys
name='flashinfer.gdn_kernels.delta_rule_dsl.local_selected_sm90'
spec=importlib.util.spec_from_file_location(name,pathlib.Path(__file__).with_name('selected_kernel.py'))
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
def run(*,q,k,v,g,beta,cu_seqlens,initial_state,output_state,output,state_checkpoints=None,checkpoint_cu_starts=None,checkpoint_every_n_tokens=0,max_seqlen=None):
 return module.delta_rule_prefill_dsl_sm90(output,output_state,q,k,v,initial_state,g,beta,cu_seqlens,128**-.5,state_checkpoints,checkpoint_cu_starts,checkpoint_every_n_tokens)
