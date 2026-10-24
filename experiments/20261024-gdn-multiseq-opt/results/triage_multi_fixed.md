Triage View
Mode: single-trace
Framework: SGLang
Input traces: /local/gdn35-experiments/20261024-gdn-multiseq-opt/profiles/multi_fixed/multi_fixed-1792816095.6291938-TP-0.trace.json.gz

Kernel Table
##### extend/prefill
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<80>, cute::C<256> >, 256, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, false, true, true, true, false, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<256>, cute::C<80> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<128, 80, 256, 128, false, true, true, true, true, true> > > > | gemm | 2783.99 ms | 39.3% | 180 | unresolved | cudaLaunchKernelExC |
| void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 0u, 0u, 1u, 256u, 128u, 128u, 128u, 128u, 128u, 3u, 128u, 256u, 1u, false, 78u, (deep_gemm::GemmType)0, cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity> | gemm | 1850.72 ms | 26.1% | 2720 | unresolved | cudaGraphLaunch |
| fused_moe_kernel | moe | 1213.79 ms | 17.1% | 1440 | unresolved | cudaGraphLaunch |
| _causal_conv1d_fwd_kernel | hybrid_linear | 96.88 ms | 1.4% | 540 | unresolved | cuLaunchKernelEx |
| kernel_cutlass_kernel_flashinfergdn_kernelsdelta_rule_dsldelta_rule_cp_sm90CPDeltaRulePrefillSm90_object_at__tensorptrf32gmemalign16o1_CopyAtom_ThrID10_TVLayoutSrc1409601_TVLayoutDst14096_0 | hybrid_linear | 94.66 ms | 1.3% | 270 | unresolved | cudaLaunchKernelExC |
| kernel_cutlass_kernel_flashinfergdn_kernelsdelta_rule_dsldelta_rule_cp_sm90CPDeltaRuleMNPrecomputeSm90_object_at__tensorptrf32gmemalign16o1_CopyAtom_ThrID10_TVLayoutSrc1409601_TVLayoutDst_0 | hybrid_linear | 92.63 ms | 1.3% | 270 | unresolved | cudaLaunchKernelExC |
| kernel_cutlass_kernel_flashinfergdn_kernelsdelta_rule_dsllocal_dvsplit_sm90_DVSplitDeltaRuleSm90_object_at__tensorptrf32gmemalign16o1_tensorptrf32gmemalign16o1_CopyAtom_ThrID10_TVLayoutSr_0 | hybrid_linear | 88.61 ms | 1.2% | 210 | unresolved | cudaLaunchKernelExC |
| void sglang::per_token_group_quant_flat_kernel<sglang::QuantTrait<__nv_bfloat16, __nv_fp8_e4m3, 128u, false, false, true, false>, true> | quantize | 84.84 ms | 1.2% | 2880 | unresolved | cudaGraphLaunch |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
##### extend/prefill
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Confirmed | 203.12 ms | 2.9% | - | unresolved | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | Matches an open upstream path (2.9% related GPU time). Open SGLang PR replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused MoE activation + quant / re-quant | Confirmed | 2047.11 ms | 28.9% | void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 0u, 0u, 1u, 256u, 128u, 128u, 128u, 128u, 128u, 3u, 128u, 256u, 1u, false, 78u, (deep_gemm::GemmType)0, cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity> (26.1%)<br>void sglang::per_token_group_quant_flat_kernel<sglang::QuantTrait<__nv_bfloat16, __nv_fp8_e4m3, 128u, false, false, true, false>, true> (1.2%) | unresolved | python/sglang/srt/layers/moe/ep_moe/kernels.py<br>python/sglang/jit_kernel/nvfp4.py<br>python/sglang/srt/layers/moe/cutlass_w4a8_moe.py | Split kernels in this family take 28.9% of GPU time. This tree already has a matching path. Quantized MoE backends already fuse activation with re-quantization. |
