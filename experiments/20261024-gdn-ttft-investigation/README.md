> **历史实验记录（公开摘录）**：以下是当时的阶段报告，旧候选不代表当前推荐实现。提示词、SSE 输出、真实激活、完整服务日志与 profiler trace 未公开；文中这些路径指原始归档。当前入口见 [统一报告](../../README.md)，公开文件及源哈希见 [manifest](../../data/source_manifest.json)。

# 35B TP1：GDN 算子加速与 TTFT 负收益调查

- 日期：2026-10-24
- 运行结果：`completed`（完成调查与采样，不代表端到端性能验收通过）

## 结论与核心结果

**GDN GPU 加速真实存在，但目前这套接入没有证明稳定的端到端收益。不能把截图中的局部加速比解释成完整请求的 TTFT 加速比。**

1. 实际 Prefill 使用 BCG（Breakable CUDA Graph），不是关闭 CUDA Graph。A/B 代表请求均有 123 次 cudaGraphLaunch，GDN 在 BCG attention/mamba 边界处执行。
2. 代表请求 sample_01 的原版目标 GDN 路径合计 23.072 ms，Dv 拆分后 15.154 ms，实际加速 1.523×，共节省 7.918 ms。
3. 原版全部 GPU kernel 累计 817.588 ms，目标 GDN 只占 2.82%；因此即便该部分加速约 1.52×，也只能减少约 1% 的这一请求耗时。GPU kernel 总时间是累计值，不等于 TTFT，含并行执行的部分。
4. 批次轨迹抓到了数百毫秒的 CUDA API 阻塞与 P2P 传输阶段等待。这些停顿在原版也发生，足以超过 GDN 省下的几毫秒。其驱动/UCX/OS 内部根因仍未确定；不能把诊断轨迹中的全部停顿直接回填到此前未采集 trace 的某个请求。

### 代表请求：18,585 tokens，sample_01

| GPU 路径 | 原版 A（ms） | Dv 拆分 B（ms） | 含义 |
|---|---:|---:|---|
| 被替换的 GDN 路径，30 层调用 | 23.072 | 15.154 | 单请求仅省 7.918 ms |
| 其余 GDN checkpoint 路径，60 层调用 | 31.574 | 31.591 | 回退原版，未加速 |
| DeepGEMM 主 kernel | 258.495 | 257.257 | 主要耗时之一 |
| 全注意力 FA3 主 kernel | 209.706 | 209.729 | 主要耗时之一 |
| fused_moe_kernel | 171.873 | 171.726 | 主要耗时之一 |
| 全部 GPU kernel 累计 | 817.588 | 808.426 | 实际 GPU 计算减少 |
| GPU kernel 忙碌区间并集 | 796.778 | 787.747 | 去除 kernel 间重叠 |
| 该次带 profiler 的客户端 TTFT | 923.024 | 932.989 | 诊断样本，不作为正式性能排名 |

原版无 checkpoint 的 eligible 调用实际自动选择 FlashInfer CP pipeline（T/MN/fixup/prefill 四个 kernel）；B 用一个 Dv-split kernel 替换它。此处 CP pipeline 指 context parallel；trace 中 `cp=64` 标签指 checkpoint 间隔，两者不要混淆。

输入分成 8192+8192+2201 三个 chunk；共 90 次 GDN 层调用，仅 30 次满足本轮优化条件。整个固定 10 请求批次两轮共 1140 次 GDN 调用，B 仅替换 420 次（36.84%），其余 720 次保留原 checkpoint 路径。

截图中的 512-token 算子为 2.98×、8192-token 为 1.51×、32768-token 为 1.46×。本次线上 eligible 调用主要是 8192 tokens，不能用 512-token 的接近 3× 估算整体收益。按原微基准估算，平均每请求 21 次 eligible 层调用 × 0.2573 ms ≈5.4 ms；这只是量级估算，实际取决于调度、checkpoint 和重叠。

### 端到端复核：原始数据全部保留

每组仍为并发 1、固定去图 10 条、AB/BA 两轮。每批清理一次 P/D cache，请求之间保留 cache。没有删除慢请求，也没有把 profiler 结果混入正式比较。

| 组别 | A 平均 TTFT ms | B 平均 TTFT ms | B 相对 A |
|---|---:|---:|---:|
| 复核，profiler 关闭 | 713.716 | 761.670 | +6.72% |
| P 重启后加入 GC 观测 | 757.536 | 688.665 | -9.09% |
| 全部形状预热后，再清缓存正式测量 | 743.549 | 776.262 | +4.40% |

三组测试属于诊断步骤，服务进程历史不同；不把它们合并为一个大样本。全部形状预热后的正式组仍为负收益，不能宣传该配置已实现稳定端到端提速。最后一组 A/B 每条缓存量一致，命中率仍为 51.48%。

最后一组各分位数（TTFT/E2E 单位 ms，TPOT 单位 ms/token）：

| 指标 | 实现 | mean | P50 | P75 | P90 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|
| ttft_ms | baseline | 743.549 | 838.658 | 924.893 | 1266.918 | 1596.298 | 1630.067 |
| ttft_ms | optimized | 776.262 | 684.490 | 1046.212 | 1525.828 | 1545.647 | 1697.210 |
| tpot_ms | baseline | 1.849 | 1.817 | 1.948 | 2.115 | 2.160 | 2.481 |
| tpot_ms | optimized | 1.847 | 1.911 | 1.983 | 2.035 | 2.130 | 2.163 |
| e2e_ms | baseline | 1275.194 | 1251.517 | 1335.424 | 1720.520 | 2676.245 | 2720.642 |
| e2e_ms | optimized | 1289.222 | 1127.529 | 1401.930 | 2161.153 | 2537.246 | 2563.863 |

## 复现与调查方法

- 主机 h20-2，GPU 6=P，GPU 7=D+DFLASH；其余 GPU 未操作。服务使用与上一轮相同模型副本、TP1、1P1D、DFLASH 8、NIXL、BCG、mem-fraction-static=0.80。
- A/B 内核与上轮相同。仅增加 GDN `record_function` 标记、GC 耗时观察和客户端计时字段；没有更改 CUDA kernel、关闭 GC、锁频或修改 CPU 配额。
- 先复核 sample_01，使用与原测试相同的 sample_00 前置历史；未开 profiler 时 A 两次 930/877 ms，B 两次 866/885 ms，未稳定重现原先约 149 ms 的 P 端劣化。
- 从 P worker `/engine/control/start_profile`、`/engine/control/stop_profile` 采集；D worker未采 torch trace。先采 sample_01 A/B 与 sample_03 A/B，with_stack=True、record_shapes=True。再采 A/B 整批，with_stack=False、record_shapes=False，降低 Python 记录开销。
- 单请求 trace 的 CPU GDN adapter 调用总时间 A≈95.74 ms、B≈90.86 ms；目标路径 CPU scope A≈30.38 ms、B≈25.70 ms。这份 trace 没有显示 B 的 Python 接入产生 149 ms 额外开销。
- 补充完整形状预热：A、B 各执行全部 10 条后，再清空 P/D cache，正式执行 AB/BA；避免把代码/JIT 预热与 prefix cache 填充混淆。
- 共 188 条诊断/预热/测量请求，全部返回成功；其中 3 组各 40 条正式对照请求，另 20 条整批 profiler 请求与 4 条单请求 profiler 采样均单独标识。
- 客户端使用本机 loopback，TTFT=POST 至首个非空 content/reasoning/tool fragment，包含前端、调度、P/D交接；TPOT=(E2E−TTFT)/(completion_tokens−1)。

## 关键证据与定位边界

### CPU/CUDA/传输停顿

整批低开销 trace 中：

| 原版请求 | 客户端 TTFT | GPU kernel 忙碌并集 | 观测 |
|---|---:|---:|---|
| sample_04 | 1054.41 ms | 134.00 ms | scheduler.run_batch 约245ms，其后有大量 P2P copy 间隔；对应优化版 204.58/133.88ms，GPU工作量几乎相同 |
| sample_09 | 2580.13 ms | 1449.06 ms | aten::arange / aten::index 的 cudaLaunchKernel 分别阻塞428.91/479.52ms，且对应 GPU 空闲间隔约429.09/479.72ms |

这些长停顿未发生在 Dv-split GPU kernel 内。sample_09 的两次阻塞在原版上发生。kernel 更快与端到端更慢可以同时成立，因为非 kernel 等待超过局部收益。带 profiler 观测存在扰动；不能据此声称历史 149ms 差异的底层根因已被完全证明。

### 排查结果

- **BCG：确认启用。** A/B 代表请求均 123 次 cudaGraphLaunch；源码为 BreakableCudaGraphBackend，attention/mamba 在边界外动态执行。没有为本次诊断关闭 BCG。
- **优化是否生效：确认。** profiler 直接看到 DVSplitDeltaRuleSm90 30 次、合计15.15ms，原 CP pipeline 被替换；非 eligible checkpoint kernel 保持60次。
- **GDN kernel 是否反而变慢：该代表 trace 不支持。** 1.52×与原先8192 microbenchmark的1.51×一致。
- **Prefix cache 可比性：确认。** 三个正式诊断组各20对请求缓存量一致。
- **P scheduler Python GC：未支持为这些正式停顿的原因。** 启动时曾有472.57ms gen2 GC，但40条观测组中A/B的TTFT区间GC均值仅0.355/0.228ms，最大单次不足3ms；该观察不覆盖D worker与所有其他进程。
- **CPU配额节流：部分请求记录到，但不与所有慢请求对应。** 记录为整个容器的 cpu.stat 差值，包含E2E区间及同容器其他任务，不能直接作为某请求CPU等待时间。
- **GPU时钟：遥测以1980MHz为主，也包含345/420MHz的低频记录（含启动、重启和空闲阶段）。500ms采样不足以排除所有短时频率变化，不能把它当成已锁频的受控实验。
- **尚未确定：** CUDA API阻塞/P2P间隔的底层来源（驱动锁、UCX/NIXL进度、操作系统调度或其他资源争用）；原负收益测试没有trace，无法逐个追溯历史停顿。

下一步性能工作应优先覆盖 checkpoint GDN，并先定位传输与 CUDA API 长停顿。当前不把此 Dv-split 接入标记为端到端优化成功，也不通过全局禁用GC或删除慢请求来制造正收益。

## 限制、配置与文件

- /prefill_model FP8、/decode_model 4-bit、/spec_model；固定去图片数据与上一轮字节一致。
- 原版平均输出长度与优化版不同，E2E不能直接当作等输出工作量比较。各组输出均值见 diagnostic_summary.json。
- 当前调查不新增模型质量验收；此前真实激活检查以原 non-CP kernel 为数值参考，不能扩大解释成整个自动 CP 路径与模型输出逐位一致。
- 同主机 GPU0–3 有其他任务；未触碰。服务和遥测已停止，GPU6/7释放。
- infer-kb 历史记录格式校验问题仍存在，按既定限制未写知识库，本报告独立保存。

| 文件 | 用途 |
|---|---|
| [diagnostic_summary.json](results/diagnostic_summary.json) | 所有正式诊断组、分位数、缓存配对 |
| [key_findings.json](results/key_findings.json) | GDN/GPU核心实测值 |
| [requests.jsonl](results/requests.jsonl) | 全部188条请求及phase标识 |
| [batch_trace_attribution_baseline.json](results/batch_trace_attribution_baseline.json) | 原版逐请求GPU/API分解 |
| [batch_stall_gaps.json](results/batch_stall_gaps.json) | 两类长停顿的精确trace位置 |
| [gc_events.jsonl](evidence/gc_events.jsonl) | GC原始时序 |
| [previous_request_timing.json](evidence/previous_request_timing.json) | 上轮P端request handler与TTFT对齐 |
| [原版三表分析](results/triage_sample01_baseline.md) | kernel/overlap/fuse技能原始输出 |
| [优化版三表分析](results/triage_sample01_optimized.md) | kernel/overlap/fuse技能原始输出 |
| profiles/ | 6份P worker原始trace；4份单请求、2份整批 |
| evidence/sources/ | 已运行版本的源码快照 |
| config/p_trace.json、config/p_default.json、config/d_default.json | 初始P、GC观测P与D的实际启动参数 |
| scripts/08_diagnose.py 至 scripts/15_report.py | 采样与统计脚本 |

技能三表为 single-trace 模式，没有 graph-off mapping/formal 配对，未证明新的 overlap 优化机会。自动 fuse 表的目录匹配仅是线索，并非已验证的可用提速；其中 FA3 被自动分类器列入 gemm，本报告按 kernel 的真实功能归为全注意力。
