> **历史实验记录（公开摘录）**：以下是当时的阶段报告，旧候选不代表当前推荐实现。提示词、SSE 输出、真实激活、完整服务日志与 profiler trace 未公开；文中这些路径指原始归档。当前入口见 [统一报告](../../README.md)，公开文件及源哈希见 [manifest](../../data/source_manifest.json)。

# 35B TP1：每批清缓存一次、逐条执行的 1P1D GDN A/B

- 日期：`2026-10-24`
- 运行结果：`completed`

## 结论与核心结果

40/40 正式请求成功，GPU 6 prefill / GPU 7 decode+DFLASH，并发 1，AB、BA 两轮。平均 TTFT 710.383→761.444 ms，变化 +7.19%；本轮未观察到平均 TTFT 提升。

每组 20 次请求的分布；TTFT/E2E 单位 ms，TPOT 单位 ms/token。

| 指标 | 版本 | mean | P50 | P75 | P90 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|
| ttft_ms | baseline | 710.383 | 707.261 | 932.603 | 1040.205 | 1559.837 | 1587.048 |
| ttft_ms | optimized | 761.444 | 792.667 | 1034.579 | 1229.632 | 1538.768 | 1546.297 |
| tpot_ms | baseline | 1.814 | 1.813 | 1.949 | 2.016 | 2.022 | 2.060 |
| tpot_ms | optimized | 1.831 | 1.841 | 1.993 | 2.058 | 2.097 | 2.187 |
| e2e_ms | baseline | 1224.285 | 1184.322 | 1321.356 | 1486.030 | 2562.018 | 2612.957 |
| e2e_ms | optimized | 1285.913 | 1250.632 | 1379.422 | 1613.315 | 2552.999 | 2604.097 |

批次均值：

| 顺序 | TTFT ms | TPOT ms/token | E2E ms | 输入 tokens | 缓存命中 tokens |
|---|---:|---:|---:|---:|---:|
| baseline 1 | 721.329 | 1.827 | 1234.690 | 239424 | 123264 |
| optimized 1 | 736.501 | 1.833 | 1276.265 | 239424 | 123264 |
| optimized 2 | 786.386 | 1.829 | 1295.561 | 239424 | 123264 |
| baseline 2 | 699.437 | 1.801 | 1213.881 | 239424 | 123264 |

## 复现

- 每个正式批次前清理一次 P、D cache，然后固定 10 条按顺序、并发 1 执行。请求之间保留 cache。
- 没有在每批计时前回放测试请求来填缓存。启动 JIT 预热 4 条，随后先清缓存再开始 A1，预热不计时。
- 顺序 A1(10)→B1(10)→B2(10)→A2(10)，四次批次前重置均有服务端成功响应。
- A 为原 FlashInfer auto；B 为相同 Dv 二分实现，仅修改 prefill worker 的合格 GDN 调用。
- TTFT 从客户端 POST 到首个非空 content/reasoning/tool fragment；E2E 到 SSE DONE；TPOT=(E2E−TTFT)/(completion_tokens−1)。
- 客户端在服务主机通过 loopback HTTP 访问 Dynamo frontend，包含服务链路，不包含远程客户端网络延迟。
- [完整环境变量和启动命令](config/actual-start-commands.sh)，直接进程启动，无 service manager wrapper。
- [测试脚本](scripts/04_benchmark.py)，[汇总脚本](scripts/05_summarize.py)。重新运行应新建目录并更新根路径/namespace，不能覆盖此结果。

## 运行配置

- 固定去图片数据 SHA256：`d5e8c10e433ea5952eaba44346489dd9161637bd1ed94060f45347f32717b30e`；与前两轮相同。
- TP1、1P1D、DFLASH 8、context-length 262144、mem-fraction-static 0.80、page-size 16、max-running-requests 16（容量上限，实际并发 1）。
- /prefill_model FP8、/decode_model 4-bit、/spec_model 的相同本地副本；models/ 和 bin/ 复用 20261023 原始实验副本。
- temperature=0、seed=42、max_tokens=512、自然停止；没有图片输入。
- 使用 /opt/sglang/bin/python，继承同一服务软件环境；没有安装或修改依赖。
- 没有强制 Triton GDN prefill；prefill 自动 FlashInfer，decode/verify 原配置。

每条的缓存情况（第 1 轮，A/B）：

| 样本 | 输入 | A 缓存 | B 缓存 | A 新计算 | B 新计算 |
|---|---:|---:|---:|---:|---:|
| sample_00 | 17925 | 0 | 0 | 17925 | 17925 |
| sample_01 | 18585 | 0 | 0 | 18585 | 18585 |
| sample_02 | 19167 | 128 | 128 | 19039 | 19039 |
| sample_03 | 19742 | 16512 | 16512 | 3230 | 3230 |
| sample_04 | 20351 | 18112 | 18112 | 2239 | 2239 |
| sample_05 | 21976 | 16384 | 16384 | 5592 | 5592 |
| sample_06 | 24059 | 18112 | 18112 | 5947 | 5947 |
| sample_07 | 25865 | 17792 | 17792 | 8073 | 8073 |
| sample_08 | 31777 | 18112 | 18112 | 13665 | 13665 |
| sample_09 | 39977 | 18112 | 18112 | 21865 | 21865 |

## 限制与证据

- 20 对 A/B 中，缓存命中与新计算量相同：20/20。
- GDN 路径计数：`{'baseline': Counter({'baseline:False': 1140}), 'optimized': Counter({'optimized:False': 720, 'optimized:True': 420})}`。
- 输出 tokens 平均值 A=284.75，B=286.15。
- 20 对 A/B 中输出长度一致 3 对、规范化输出一致 0 对；E2E 不是等输出量比较。
- 只有 10 个不同请求、每个两次，高分位数不代表生产尾延迟。
- sample_03 在 B2 的 TTFT 为 748.5 ms，其余三次为 263.7–282.9 ms；该请求全部回退原 GDN，说明本轮存在不能直接归因于优化 kernel 的延迟波动。保留全部测量值，未剔除异常点。
- 同主机 GPU 0–3 有其他任务，未触碰；未锁频。
- 沿用已测试 kernel；本轮未增加真实激活正确性验证，不宣称端到端输出确定性或模型质量验收。
- infer-kb 既有结果格式校验失败尚未修复，依照技能限制未写知识库，此处独立保存。
- [逐请求结果](results/requests.jsonl)、[完整统计](results/summary.json)、[缓存审计](results/cache_audit.json)、[重置记录](results/cache_resets.jsonl)、logs/ 下服务日志。
