> **历史实验记录（公开摘录）**：以下是当时的阶段报告，旧候选不代表当前推荐实现。提示词、SSE 输出、真实激活、完整服务日志与 profiler trace 未公开；文中这些路径指原始归档。当前入口见 [统一报告](../../README.md)，公开文件及源哈希见 [manifest](../../data/source_manifest.json)。

# 35B TP1：checkpoint GDN 优化与端到端验收

- 日期：2026-10-24；运行结果：`completed`，不等于已通过模型质量或所有场景性能验收。
- h20-2；GPU6=P，GPU7=D+DFLASH；其余GPU未操作。
- 原版无 checkpoint 的自动 CP 路径保留；仅优化 checkpoint 路径。BCG、模型、TP1、1P1D、DFLASH8、NIXL 保持。

## 核心结果

固定10条、并发1、AB/BA两轮，平均TTFT：789.355 → 689.546 ms，变化 -12.64%。全部40条保留，未删除慢请求。

20对请求输入与缓存量一致：20/20；归一化输出文本一致：1/20。原版自身重复运行也存在文本变化，见 stock_a1_repeatability.json；本报告不宣称端到端输出逐位确定性。

## 实际优化内容

1. 为原有V维拆分添加完整checkpoint写出支持：[32,128,128]状态作为[64,64,128]计算分片，两个分片写不相交的V行，保持外部布局。
2. 审计缓存消费者后，只为单序列保存实际需要的一个中间状态；支持第128、第1280等前部边界，而非只优化尾部。最终状态仍正常保存。
3. 不适用时保留原路径；紧凑元数据下若kernel条件不满足，则重建原版dense调用并取回相同状态。多序列不进入此优化。
4. 最终B使用固定环境开关，无逐层模式文件读取、审计记录或shadow核对。A未安装GDN hook。

固定10条共570次GDN调用，其中360次checkpoint调用被覆盖，210次无checkpoint调用保留原版。每层原本生成914份中间状态，消费者只取12份；对应30层checkpoint写出数据量从57,503,907,840 bytes降到754,974,720 bytes，减少98.69%。这是整个批次的累计写出量，不是峰值显存；8192-token调用的checkpoint buffer从256MiB降到2MiB。这不代表模型总带宽或总耗时降低98.69%。

## 正确性与拒绝的候选

- dense checkpoint合成32组通过，包括短边界、长序列、变长/空序列及near-one gate；token输出、最终状态、全部checkpoint与原版逐位一致。
- 指定位置保存24组通过，包括早/中/末边界、零/非零初始状态、CUDA Graph和恢复续算。
- 紧凑元数据回退到原版dense的3个真实输入场景通过，覆盖8192的128/1280位置及3230的3200位置。
- 捕获6份真实输入回放；checkpoint输出与原版一致，实际选取位置及恢复续算一致。
- 服务内shadow核对720次：dense360次、指定位置360次；输出、最终状态、所需checkpoint最大差异0。
- Compute Sanitizer memcheck 4组，0 errors；关闭了CUDA API探测返回码报告，设备内存和stream-ordered race检查保留。这不等于运行了全部racecheck工具。
- 独立FP32递推：真实输入的短片段在预设阈值下有失败，原版有完全相同误差。保留失败标记，未放宽阈值；当前保证是对原版无新增数值差异，不宣称独立高精度参考全部通过。
- 拒绝旧的全路径DVSplit：真实auto-CP最终状态有9/524288元素超出atol/rtol=.02，最大违规绝对差约.034626。stage1日志保留。此前non-CP逐位一致不能覆盖auto-CP，因此最终候选不替换这条路径。

## 完整GDN extend组件

包括输入预处理、gate转换、状态池读写及两边相同的状态复位。下表为CUDA Graph微基准中位数，不是完整模型TTFT；eager计时亦保存在component.json。

| tokens | 原版 ms | dense DV ms | 指定状态 DV ms | 加速比 |
|---|---:|---:|---:|---:|
| 2239 | 0.2659 | 0.2029 | 0.1789 | 1.49× |
| 3230 | 0.3747 | 0.2833 | 0.2472 | 1.52× |
| 8192 | 0.8971 | 0.6675 | 0.5791 | 1.55× |

## 服务内真实轨迹

| 请求 | checkpoint原版 ms | 优化 ms | GPU kernel总时间 A/B ms | Graph launch A/B |
|---|---:|---:|---:|---:|
| sample01 | 31.614 | 19.706 | 818.107 / 805.969 | 123 / 123 |
| sample03 | 9.915 | 6.195 | 189.485 / 185.219 | 41 / 41 |

sample01输入18585 tokens，分8192+8192+2201，共90层调用：60次checkpoint被替换，另30次无checkpoint CP pipeline保留。sample03新增3230、缓存16512 tokens。GPU累计时间不等于TTFT，profiler请求不进入正式排名。

## 正式端到端数据

每个部署先执行固定10条完成代码/形状预热，再清P/D缓存测量；每批内逐条执行并保留前缀缓存。顺序A1、B1、B2、A2；A1/A2为未安装hook的原版P进程，B1/B2为固定优化P进程，D/frontend保持不变。每次P重启完成相同预热。两边服务random-seed=42，payload temperature=0、seed=42、max_tokens=512、自然停止。

| 指标 | 实现 | mean | P50 | P75 | P90 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|
| ttft_ms | baseline | 789.355 | 746.321 | 1150.884 | 1449.224 | 1564.396 | 1582.435 |
| ttft_ms | sparse | 689.546 | 694.255 | 907.401 | 1000.763 | 1552.571 | 1569.554 |
| tpot_ms | baseline | 1.805 | 1.841 | 1.876 | 1.957 | 2.071 | 2.110 |
| tpot_ms | sparse | 1.799 | 1.775 | 1.875 | 1.973 | 2.006 | 2.087 |
| e2e_ms | baseline | 1289.046 | 1247.734 | 1381.602 | 1828.951 | 2539.509 | 2553.359 |
| e2e_ms | sparse | 1193.715 | 1085.885 | 1291.363 | 1481.592 | 2451.540 | 2610.885 |

TTFT/E2E单位ms，TPOT单位ms/token。分位数采用线性插值；每臂仅20个固定样本，P95/P99不能代表生产总体。TTFT为POST至首个非空content/reasoning/tool片段。TPOT=(E2E−TTFT)/(输出tokens−1)。

| 批次 | 平均TTFT ms |
|---|---:|
| baseline 1 | 797.779 |
| sparse 1 | 688.770 |
| sparse 2 | 690.322 |
| baseline 2 | 780.930 |

## 收益归因与边界

20个配对中15个TTFT降低；配对节省均值99.81ms、中位数26.67ms。平均值受少数大差异影响。不能将12.64%的端到端观测改善全部归因于GPU kernel。

sample01轨迹直接确认checkpoint GDN节省11.91ms、全部GPU kernel累计节省12.14ms，而正式请求该样本节省约275/305ms。两者是不同测量运行；额外差异的因果来源尚未分解。不能据此宣称NIXL传输节省数百毫秒。

读取正式运行P端日志后，sample01的P handler为1136.3→861.2ms、1181.1→873.4ms，几乎覆盖其TTFT变化。定位范围是P handler内部，仍包含调度、CPU、GPU等待和状态处理；日志中的统计打印间隔不是GPU执行时间，不能把它命名为纯CUDA或纯传输等待。原始逐请求数据见results/formal_p_log_attribution.json。

E2E P99没有改善；每臂仅20条，不应给出生产尾延迟保证。保持BCG、缓存策略和解码配置后，本次覆盖的是这10条真实单并发请求的checkpoint路径，并非所有shape的普遍加速。

| 请求 | A1 TTFT ms | B1 TTFT ms | A2 TTFT ms | B2 TTFT ms |
|---|---:|---:|---:|---:|
| sample_00 | 843.19 | 843.31 | 855.78 | 835.59 |
| sample_01 | 1139.38 | 864.78 | 1185.40 | 880.72 |
| sample_02 | 1202.85 | 907.71 | 1436.56 | 907.30 |
| sample_03 | 649.45 | 257.85 | 255.14 | 256.11 |
| sample_04 | 233.22 | 197.61 | 196.43 | 230.98 |
| sample_05 | 378.23 | 369.01 | 378.92 | 368.10 |
| sample_06 | 414.43 | 403.48 | 423.24 | 402.82 |
| sample_07 | 539.68 | 552.92 | 553.87 | 520.96 |
| sample_08 | 990.41 | 939.58 | 960.75 | 926.84 |
| sample_09 | 1586.95 | 1551.45 | 1563.21 | 1573.80 |

输出tokens均值：A 279.10，B 281.55。因此E2E并非严格等输出工作量比较，TPOT变化也不直接归因于P端GDN。

## 复现与使用

完整命令见scripts/03_launch.py和config/p_stock_a1.json、p_sparse_fixed.json、p_stock_a2.json、d_default.json。固定数据SHA256：`d5e8c10e433ea5952eaba44346489dd9161637bd1ed94060f45347f32717b30e`。

P使用/prefill_model的本地副本(FP8)，D使用/decode_model副本(4-bit)，draft为/spec_model副本；context262144、mem.80、page16、max-running16（负载并发仍为1）、NIXL、DFLASH8。frontend18000，P18081，D18082，etcd12379。

运行时只需在P侧使用release中的四个Python文件，设置PYTHONPATH指向该目录，并设置GDN35_PATCH=1、GDN35_VARIANT=sparse；D不要启用该hook。回退时移除这两个环境变量并重启P。适用范围为本次H20、TP1、Hq16/Hv32、D128、BF16激活/FP32状态及已验证软件版本。

内部负checkpoint间隔仅用于实验适配器传递“指定保存的token位置”；它不是FlashInfer公共API，不能直接传给原版函数。源码快照、生成器和kernel diff一并保存，没有修改安装目录。

## 证据索引

- results/formal_summary.json、requests.jsonl：全部正式数据与其他phase；原始SSE逐请求保存。
- results/checkpoint_full.json、selected_checks.json、real_replay.json、component.json；evidence/real_checks.jsonl：正确性与组件证据。
- profiles/：4份原始P trace；results/trace_*.json：kernel/API统计。
- results/triage_sample01_*.md：技能原始kernel/overlap/fuse三表，single-trace模式，没有关闭BCG。自动分类器将FA3列为gemm，本报告按全注意力理解；fuse目录匹配不是已验证优化。
- evidence/stage1_logs/、logs/replay_initial_fp32_failure.log：保留的失败记录。
- evidence/capture_manifest.json：6份远程真实张量fixture的校验信息；原始fixture保留在远程captures/。
- evidence/gpu_telemetry.csv、cleanup.json、gpu_final.csv：运行环境与清理记录。

本次没有修改MoE、全注意力、通信参数、CPU配额、GPU频率或GC策略。同机GPU0–3有其他工作，未触碰；本次结果不排除共享资源干扰。服务与遥测在结束时停止。知识库历史格式校验仍失败，按技能要求未写入知识库，本报告独立保存。
