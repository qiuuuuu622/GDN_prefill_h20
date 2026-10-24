> **历史实验记录（公开摘录）**：以下是当时的阶段报告，旧候选不代表当前推荐实现。提示词、SSE 输出、真实激活、完整服务日志与 profiler trace 未公开；文中这些路径指原始归档。当前入口见 [统一报告](../../README.md)，公开文件及源哈希见 [manifest](../../data/source_manifest.json)。

# 35B TP1：1P1D + DFLASH，GDN 优化端到端 A/B

- 日期：`2026-10-23`
- 运行结果：`completed`

## 结论与核心结果

**40/40 正式请求完成；并发 1，固定 10 条去图片请求，执行 AB、BA 两轮。观察到平均 TTFT 降低 2.22%、E2E 降低 3.09%，TPOT 增加 0.93%；这两轮不足以证明稳定的端到端收益。**

第二轮 B 的平均 TTFT 反而比 A 高 0.72%。此外 B 平均输出 283.5 tokens，A 为 297.5 tokens，输出工作量不同；E2E 变化不能全部归因于 GDN。

以下为每种实现 20 次正式请求的分布，NumPy linear percentile；TTFT/E2E 单位 ms，TPOT 单位 ms/token。P95/P99 基于小样本，仅代表本次请求。

| 指标 | 实现 | mean | P50 | P75 | P90 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|
| ttft_ms | baseline | 1256.613 | 1059.449 | 1358.668 | 1876.179 | 2335.236 | 2344.138 |
| ttft_ms | optimized | 1228.730 | 1032.695 | 1396.140 | 1909.034 | 2298.329 | 2303.367 |
| tpot_ms | baseline | 1.836 | 1.870 | 1.942 | 1.983 | 2.063 | 2.115 |
| tpot_ms | optimized | 1.853 | 1.854 | 1.962 | 2.003 | 2.013 | 2.113 |
| e2e_ms | baseline | 1801.835 | 1652.473 | 2025.584 | 2394.692 | 3297.829 | 3343.910 |
| e2e_ms | optimized | 1746.247 | 1585.091 | 2042.086 | 2362.164 | 3245.691 | 3301.728 |

各批均值（10 请求/批）：

| 顺序 | 实现 | TTFT ms | TPOT ms/token | E2E ms | 输出 tokens mean |
|---|---|---:|---:|---:|---:|
| r1 | baseline | 1273.231 | 1.852 | 1813.631 | 292.1 |
| r1 | optimized | 1208.540 | 1.855 | 1737.515 | 287.4 |
| r2 | optimized | 1248.920 | 1.851 | 1754.980 | 279.6 |
| r2 | baseline | 1239.995 | 1.820 | 1790.039 | 302.9 |

## 复现

完整 worker 命令和环境：[config/p_default.json](config/p_default.json)、[config/d_default.json](config/d_default.json)。用户提供的 manager 格式已缩为仅 p0/d0：[config/service-1p1d.yaml](config/service-1p1d.yaml)。本次直接启动 worker/Frontend/etcd，没有启动 manager 包装层。

```bash
cd /root/gdn35/20261023-e2e-1p1d
# 使用 scripts/01_copy_models.py 复制并按相对文件名/大小验证模型
# scripts/02_select_data.py 负责去图和固定样本；复测应直接复用 config/fixed10.json
printf "baseline\n" > config/mode.txt
/opt/sglang/bin/python scripts/03_launch.py etcd
/opt/sglang/bin/python scripts/03_launch.py frontend
/opt/sglang/bin/python scripts/03_launch.py p
/opt/sglang/bin/python scripts/03_launch.py d
# 等待两个 worker ready 和 /v1/models 注册
/opt/sglang/bin/python scripts/04_benchmark.py smoke
/opt/sglang/bin/python scripts/04_benchmark.py verify
/opt/sglang/bin/python scripts/04_benchmark.py bench
/opt/sglang/bin/python scripts/05_summarize.py
```

复测须使用新的结果目录或归档旧结果/PID；脚本故意拒绝覆盖已存在的正式运行。

- A：原 FlashInfer public auto；B：仅替换 P worker 中无 checkpoint 的单序列 TP1 GDN prefill 为 Dv 二分。安装包未改动。
- 单进程通过模式文件切换。实际 breakable prefill graph 仍在运行期执行 GDN 调用，route 日志证实 A/B 分别命中对应实现；未关闭 CUDA Graph。
- P 的末尾 chunk 需要每 64 tokens 的状态 checkpoint，B 在这些调用上回退原实现；8192-token 中间 chunk 命中优化。
- 每组先预热完整 10 条，不计入分位数；再按 A(10)→B(10)→B(10)→A(10) 顺序测量。
- 客户端并发 **1**，服务 max-running-requests=16 仅为容量上限；一条完成后才发下一条。
- 每条请求前经 Dynamo clear_kv_blocks 清理 P 和 D，两侧返回 success；209 条 prefill batch 日志的 cached-token 全为 0（含启动/预热/验证）。
- 两组均 temperature=0、seed=42、max_tokens=512，保留自然停止。
- TTFT：客户端 POST 至首个非空 content/reasoning/tool fragment；E2E：至 SSE DONE；TPOT=(E2E−TTFT)/(completion_tokens−1)。它不是投机解码 SSE 事件之间的间隔。

## 运行配置

| 项目 | 实际值 |
|---|---|
| 主机/GPU | h20-2，物理 GPU 6 prefill、GPU 7 decode+draft，H20 96GB |
| 拓扑 | TP1、1P1D、NIXL，独立 namespace gdn35-e2e-1023 |
| 模型 | /prefill_model FP8；/decode_model 4-bit compressed-tensors；/spec_model DFLASH；实际从本次 models/ 磁盘副本加载 |
| 公共参数 | context=262144、mem-fraction-static=0.80、page-size=16、max-running-requests=16、DFLASH draft tokens=8 |
| 后端 | 自动选择 prefill=FlashInfer；decode/verify=Triton；P prefill graph=breakable，D verify graph=full |
| 端口 | Frontend 18000；P/D health 18081/18082；etcd 12379；NCCL 19506/19507；KV events 55606/55607 |
| 软件 | Dynamo 1.5.0，SGLang 0.5.20，FlashInfer 0.6.18，torch 2.13.0+cu130 |
| SGLang commit | 94602c9c2b7cbdb8efd5c52802dac6a1c180089e（工作树状态另存，非 clean checkout 保证） |
| 镜像 digest | 未暴露，未记录 |
| 数据 SHA256 | `d5e8c10e433ea5952eaba44346489dd9161637bd1ed94060f45347f32717b30e` |

数据源：`/private/benchmark-data/122b_20260914_2030_2130_clean.jsonl`。读取前 1000 行，先删除图片内容块/占位符，再跳过无法应用当前模型 chat template 的记录。142 条有效候选按输入长度分层取 10 条；选中请求共删除 11 个图片块。并非对整个 122B 数据集的随机代表性抽样。

固定样本：

| 样本 | 源行号 | 模板 token 估计 | 删除图片块 |
|---|---:|---:|---:|
| sample_00 | 581 | 17925 | 0 |
| sample_01 | 762 | 18585 | 0 |
| sample_02 | 805 | 19167 | 0 |
| sample_03 | 612 | 19742 | 0 |
| sample_04 | 246 | 20351 | 0 |
| sample_05 | 543 | 21976 | 0 |
| sample_06 | 525 | 24057 | 1 |
| sample_07 | 949 | 25865 | 0 |
| sample_08 | 921 | 31773 | 2 |
| sample_09 | 1000 | 39950 | 8 |

Dynamo 实际 prompt token 数略有模板差异，以 requests.jsonl 的 usage 为准；全部 20 对 A/B 的实际输入 token 数相同。

## 限制与证据

- 真实激活检查：sample_00 的两个完整 chunk、30 层/块，共 60 次 GDN 调用，Dv 二分与原 non-CP 的输出和 FP32 状态均完全一致。末尾需要 checkpoint 的 30 层调用回退，未计入这 60 项。
- **端到端输出并非逐位一致。** 20 对 A/B 中，3 对输出 token 数相同、1 对规范化输出相同；原版自身两次运行也只有 1/10 相同，优化版自身为 0/10。原因尚未定位，不能把这些差异直接归为新 kernel 错误，也不能声称完整模型正确性验收通过。
- B 输出 token 均值比 A 少约 4.71%，影响 E2E；即使 temperature=0 和 seed=42，本套服务仍未表现为确定性。
- 只有 10 个不同请求、两轮、并发 1；不能外推并发 10、吞吐、排队或生产流量。
- 同主机 GPU 0–3 运行其他任务；未触碰，未锁频。
- 首次 P 启动因缺失 topology/host 失败，补全后重启成功；失败日志保留。正式 40 请求中无失败。
- 知识库 ensure 被既有旧记录的格式错误阻塞，因此按 infer-kb 规定不写该知识库，本实验独立保存。

| 证据 | 路径 |
|---|---|
| 固定去图样本 | [config/fixed10.json](config/fixed10.json) |
| 选择/指纹 | [evidence/dataset_manifest.json](evidence/dataset_manifest.json) |
| 全部请求指标 | [results/requests.jsonl](results/requests.jsonl) |
| 完整分位数和配对结果 | [results/summary.json](results/summary.json) |
| 同实现重复稳定性 | [results/repeatability.json](results/repeatability.json) |
| 原始逐事件 SSE | results/*.sse.json |
| 清缓存回执 | [results/cache_resets.jsonl](results/cache_resets.jsonl) |
| 真实激活校验 | [evidence/real_activation_checks.jsonl](evidence/real_activation_checks.jsonl) |
| GDN 实际路径 | [evidence/route_events.json](evidence/route_events.json) |
| worker/Frontend/benchmark 日志 | logs/ |
| DFLASH metrics | evidence/decode_metrics_before.txt、decode_metrics_after.txt |
| 代码与命令 | scripts/、config/ |


### 为什么端到端收益较小

10 个样本每条平均 2.4 个完整 8192-token chunk，模型有 30 个 GDN 层，即平均 72 次 GDN 调用命中优化。前一轮独立算子测试该 shape 为 0.7633→0.5060 ms，每次节省 0.2573 ms；粗估每条请求节省 18.53 ms，相对于约 1256.6 ms TTFT 约为 1.47%。这是跨实验估计，不是本次 GPU trace 归因。

本轮实际 P handler 平均时长为 1252.20→1224.32 ms（含服务端非 GPU 工作），差值约 27.88 ms，与客户端 TTFT 差值一致。TTFT 的大部分时间在 P 侧，而被优化的 GDN 调用只占其中一小部分。详见 results/attribution_estimate.json。
