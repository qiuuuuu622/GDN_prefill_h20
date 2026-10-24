> **历史实验记录（公开摘录）**：以下是当时的阶段报告，旧候选不代表当前推荐实现。提示词、SSE 输出、真实激活、完整服务日志与 profiler trace 未公开；文中这些路径指原始归档。当前入口见 [统一报告](../../README.md)，公开文件及源哈希见 [manifest](../../data/source_manifest.json)。

# 35B TP1：1P1D + DFLASH，保留缓存的 GDN A/B

- 日期：`2026-10-23`
- 运行结果：`completed`

## 结论与核心结果

**并发 1、固定去图片 10 请求、AB/BA 两轮，共 40/40 成功。原版与优化版的平均 TTFT 为 145.77→136.55 ms，降低 6.33%；平均 E2E 为 649.17→662.54 ms，增加 2.06%。TTFT P95 未改善，不能概括为全面端到端提速。**

每批 10 条共 239,424 输入 tokens，其中 233,344 命中缓存，6,080 重新计算，命中率 **97.46%**。20 对 A/B 的缓存命中量逐条一致。

下表为每组 20 次正式请求的分布，NumPy linear percentile；TTFT/E2E 单位 ms，TPOT 单位 ms/token。

| 指标 | 实现 | mean | P50 | P75 | P90 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|
| ttft_ms | baseline | 145.774 | 118.099 | 145.250 | 207.733 | 318.280 | 322.723 |
| ttft_ms | optimized | 136.548 | 108.597 | 127.154 | 213.853 | 325.912 | 337.761 |
| tpot_ms | baseline | 1.803 | 1.796 | 1.848 | 2.052 | 2.065 | 2.101 |
| tpot_ms | optimized | 1.849 | 1.855 | 1.946 | 2.030 | 2.079 | 2.203 |
| e2e_ms | baseline | 649.167 | 622.397 | 723.553 | 795.017 | 1197.965 | 1203.260 |
| e2e_ms | optimized | 662.538 | 637.316 | 788.851 | 874.387 | 1137.296 | 1241.056 |

每批均值：

| 轮次/顺序 | 实现 | TTFT ms | TPOT ms/token | E2E ms |
|---|---|---:|---:|---:|
| 1 | baseline | 152.479 | 1.782 | 644.896 |
| 1 | optimized | 135.328 | 1.887 | 670.013 |
| 2 | optimized | 137.768 | 1.812 | 655.063 |
| 2 | baseline | 139.069 | 1.825 | 653.438 |

与上一轮冷缓存的原版均值相比，保留缓存使 TTFT 从 1256.61 ms 降至 145.77 ms，约降低 88.4%。这部分收益来自缓存，不能记为 GDN kernel 改动的收益。

## 复现

- 仍为物理 GPU 6 prefill、GPU 7 decode+DFLASH；并发 **1**，服务 max-running-requests=16 只是容量上限。
- 固定样本字节内容与冷缓存测试完全相同，图片内容块已删除；temperature=0、seed=42、max_tokens=512，自然停止。
- 服务启动后先用 A/B 各两条请求完成 JIT 预热，不计时。
- 每个正式批次之前，只在预热开始前清理一次 P/D cache；然后统一使用原版、相同顺序预热完整 10 条，再切换到待测 A 或 B。
- **正式批次内保留所有缓存，逐请求清缓存已移除。** 执行 A(10)→B(10)→B(10)→A(10)，正式请求共 40，缓存预热请求共 40，JIT 预热请求共 4。
- A 调用原 FlashInfer auto；B 只对合格单序列、无 checkpoint 的 GDN 调用使用 Dv 二分。P/D 其他实现和 CUDA Graph 保持一致。
- TTFT 为客户端 POST 至首个非空内容/推理/tool fragment；E2E 至 SSE DONE；TPOT=(E2E−TTFT)/(completion_tokens−1)。

```bash
cd /root/gdn35/20261023-e2e-warm-1p1d
printf "baseline\n" > config/mode.txt
/opt/sglang/bin/python scripts/03_launch.py etcd
/opt/sglang/bin/python scripts/03_launch.py frontend
/opt/sglang/bin/python scripts/03_launch.py p
/opt/sglang/bin/python scripts/03_launch.py d
/opt/sglang/bin/python scripts/04_benchmark.py
/opt/sglang/bin/python scripts/05_summarize.py
/opt/sglang/bin/python scripts/06_report.py
/opt/sglang/bin/python scripts/07_stop.py
```

复测应新建结果目录并更新脚本路径、namespace；脚本拒绝覆盖已有正式结果/PID。models/、bin/ 复用前一轮本机磁盘副本。

## 运行配置

- 模型：/prefill_model FP8、/decode_model 4-bit compressed-tensors、/spec_model DFLASH；模型本地副本与上轮相同。
- TP1、1P1D、DFLASH=8、context=262144、mem-fraction-static=0.80、page-size=16、max-running-requests=16。
- Dynamo 1.5.0、SGLang 0.5.20、FlashInfer 0.6.18；源码状态沿用前一轮，未升级包。镜像 digest 未暴露。
- GDN 实际自动选择：prefill=FlashInfer，decode/verify=Triton。
- namespace=gdn35-warm-1023；Frontend=18000，P/D health=18081/18082，etcd=12379。
- 固定数据 SHA256：`d5e8c10e433ea5952eaba44346489dd9161637bd1ed94060f45347f32717b30e`。

实际缓存分布（两轮、两组逐条一致）：

| 样本 | 输入 tokens | 缓存命中 | 新计算 | B 的 GDN 路径 |
|---|---:|---:|---:|---|
| sample_00 | 17925 | 17920 | 5 | Dv 二分 |
| sample_01 | 18585 | 18560 | 25 | Dv 二分 |
| sample_02 | 19167 | 19136 | 31 | Dv 二分 |
| sample_03 | 19742 | 18112 | 1630 | checkpoint 回退 |
| sample_04 | 20351 | 20288 | 63 | Dv 二分 |
| sample_05 | 21976 | 17792 | 4184 | checkpoint 回退 |
| sample_06 | 24059 | 24000 | 59 | Dv 二分 |
| sample_07 | 25865 | 25856 | 9 | Dv 二分 |
| sample_08 | 31777 | 31744 | 33 | Dv 二分 |
| sample_09 | 39977 | 39936 | 41 | Dv 二分 |

## 限制与证据

- 这是**同一批请求预热后重复回放**的 warm-cache 结果，97.46% 命中率不是生产多轮请求缓存命中率的估计。
- B 的 20 次请求合计 480 次 GDN 层调用命中优化、120 次因 checkpoint 回退；对应每批 8 条短尾请求优化、2 条较长尾请求回退。
- TTFT 均值两轮分别降低约 11.25%、0.94%，改善幅度不稳定；P95/P99 未改善。只有 10 个不同请求，小样本高分位数不代表生产尾延迟。
- 原版平均输出 279.4 tokens，优化版 283.6 tokens。20 对 A/B 中仅 5 对输出 token 数相同、1 对规范化输出相同；因此 E2E 不是等输出工作量比较，不能用它独立判定 kernel 性能。
- 本轮未新增真实激活逐位比较；kernel 与此前 60 次真实激活检查通过的实现相同。端到端模型输出仍未实现确定性，不宣称模型质量或逐位一致性验收。
- 同主机 GPU 0–3 有其他任务，未触碰；未锁频。
- 最初 benchmark 记录器的语法错误发生在执行请求前，已修复并保留日志；正式 40 请求无失败。
- infer-kb ensure 仍被历史结果格式错误阻塞，按技能要求停止知识库相关写入，本实验独立保存。

| 证据 | 路径 |
|---|---|
| 完整统计/配对输出 | [results/summary.json](results/summary.json) |
| 每请求指标、缓存与 GDN 路径 | [results/requests.jsonl](results/requests.jsonl) |
| 缓存可比性校验 | [results/cache_audit.json](results/cache_audit.json) |
| 缓存重置发生位置 | [results/cache_resets.jsonl](results/cache_resets.jsonl) |
| 固定去图样本 | [config/fixed10.json](config/fixed10.json) |
| 服务命令 | config/p_default.json、config/d_default.json、config/service-1p1d.yaml |
| 原始 SSE | results/*.sse.json |
| 服务与测试日志 | logs/ |
| metrics | evidence/prefill_metrics_after.txt、evidence/decode_metrics_after.txt |
