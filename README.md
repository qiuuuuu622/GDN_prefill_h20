# H20 GDN Prefill：122B TP4 与 35B TP1 统一优化记录

本仓库汇总 **Qwen3.5-MoE-122B FP8 / TP4** 和 **Qwen3.5 35B / TP1** 在 H20（SM90，78 SM）上的 GDN prefill 优化：核心代码、可重建内核、实验过程、正确性结果、算子数据和端到端数据。

**两种模型采用不同优化路线，不能把算子加速比当作 TTFT 加速比。** 122B 现有结果支持混合 CP 分流；35B 实测优先选择自适应 V 切分与选定 checkpoint 写出。35B 并发 8 仍有退化，尚未证明并发 2–8 的稳定普遍收益。

## 统一代码入口

现在可安装统一包：`pip install --no-deps .`，通过 `from gdn_prefill import GDNPrefill, Geometry, CheckpointRequest` 使用 `prepare()` / `run()`。核心代码在 [src/gdn_prefill](src/gdn_prefill/)，完整契约和用法见 [统一入口](docs/unified-entry.md)。已测私有内核固定 FlashInfer 0.6.18，并校验源码 hash。

默认 checkpoint 路径共享 full-V/V2，按实际 geometry 和长度分流；无 checkpoint 时122B保留CP/混合CP、35B保留原版行为。native CP checkpoint 的边界写出已修复，但 near-one 对 non-CP 回归仍失败，必须显式 experimental，不能自动启用。历史服务hook与新入口证据分开记录；新包尚未做完整服务AB。

## 从这里阅读

| 内容 | 入口 |
|---|---|
| 可安装公共入口与维护方式 | [统一入口](docs/unified-entry.md) / [源码](src/gdn_prefill/) |
| 新公共入口的独立GPU验证与完整入口计时 | [90项差分、46项memcheck、8组计时](experiments/20261024-unified-entry/README.md) |
| 同版本跨geometry对比、CP边界修复 | [新数据表](docs/unified-tables.md) / [CP修复报告](experiments/20261024-gdn-cp-checkpoint-fix/README.md) |
| checkpoint 差异与统一契约 | [checkpoint 统一方案](docs/checkpoint-contract.md) |
| 两种模型的机制、选择依据与 CP/V 关系 | [优化机制与取舍](docs/mechanisms.md) |
| 全部端到端分位数、CP/V 扫描、真实 shape 回放 | [自动生成的数据表](docs/tables.md) / [CSV](data/) |
| 122B 原始报告及限制 | [122B 历史报告](docs/122b-original-report.md) |
| 35B 当前核心代码、加载与撤销 | [35B 复现说明](qwen35/README.md) / [runtime](qwen35/runtime/) |
| 每轮实验的目的、演进和拒绝候选 | [实验索引](experiments/README.md) |
| 正确性、稳定性、缓存和归因边界 | [验证与测量协议](docs/validation.md) |
| 来源哈希、公开范围和缺失证据 | [证据说明](docs/evidence.md) / [manifest](data/source_manifest.json) |

## 历史优化路线与证据范围

| 项目 | 122B TP4 | 35B TP1 |
|---|---|---|
| 每 rank GDN geometry | Hq/Hk=4，Hv=16，Dk/Dv=128 | Hq/Hk=16，Hv=32，Dk/Dv=128 |
| 权重与激活 | FP8 权重目标；算子 BF16 I/O、FP32 状态 | P FP8 权重；算子 BF16 I/O、FP32 状态 |
| FlashInfer 基线 | `597e3a67c`，0.7.2-dev | 安装版 0.6.18；源文件随实验冻结 |
| 主要优化 | 最长序列 CP + 其余序列 non-CP，双 stream、零拷贝视图 | checkpoint 路径：V128/V64 自适应，按序列只写实际消费的中间状态 |
| 生效条件 | auto、CPU 长度前缀和、N≥2、无 checkpoint/indices | 已验证 H20/TP1 shape、N≤8、BF16 激活/FP32 状态 |
| 原实现未覆盖路径 | checkpoint；SGLang 端到端尚未接入验证 | 无 checkpoint auto-CP 保留原版；其他 shape 回退 |
| 证据级别 | 仓库原有算子与测试报告 | 合成、真实激活回放、在线 shadow、1P1D E2E |

**“122B 无 checkpoint”是原补丁的生效条件，不是模型不需要 checkpoint。** 122B 的真实服务 checkpoint 覆盖尚未审计；应统一保存/恢复语义和正确性验证，底层仍可分别选择 CP 或 V 切分。详见[统一方案](docs/checkpoint-contract.md)。

这里的 CP 指 **同一张 GPU 内沿 token 分段的并行算法**，不是多卡 context parallel；checkpoint 指供后续前缀状态恢复使用的中间状态，两者不同。

## 主要结果

### 122B TP4：混合 CP（原仓库报告）

| 序列长度 | 原版 ms | 混合 CP ms | 算子加速比 |
|---|---:|---:|---:|
| 8192 + 3×1024 | 0.753 | 0.519 | 1.45× |
| 16384 + 3×1024 | 1.500 | 0.884 | 1.70× |
| 32768 + 3×1024 | 2.999 | 1.605 | 1.87× |

CUDA Graph 设备时间，**不是端到端 TTFT**。这些数值来自原仓库提交 `9a3aaeab`；这张表保留原报告；新增实验另在 FlashInfer 0.6.18 对两种 rank-local geometry 比较 full-V/V2/CP，见[同版本新表](docs/unified-tables.md)。新增实验没有运行122B完整模型服务。[完整报告](docs/122b-original-report.md)。

### 35B TP1：最终多序列 checkpoint 路径

| 合成序列长度 | 分流 | 原版 ms | 优化 ms | 算子加速比 |
|---|---|---:|---:|---:|
| [8192] | V 切分 | 0.8293 | 0.5453 | 1.52× |
| [64, 8128] | V 切分 | 0.8235 | 0.5457 | 1.51× |
| [4096, 4096] | 完整 V | 0.4207 | 0.3813 | 1.10× |
| [2048, 2048, 2048, 2048] | 完整 V | 0.4237 | 0.3888 | 1.09× |

六组真实双序列回放为 **1.10–1.23×**。V 切分主要解决少量长序列的并行度；均衡合批主要受益于减少无用 checkpoint 写出。所有中间 token 仍参与计算。[完整 shape 数据](data/35b_checkpoint_shapes.csv)。

### 35B TP1：端到端，固定十条，AB / BA

GPU **6=P，7=D+DFLASH8**，1P1D、NIXL、BCG 保持。每部署先预热；每批开始清 P/D 前缀缓存，批内保留；闭环负载、自然输出，max_tokens=512。

| 实验 / 并发 | TTFT 原版→优化 ms | 变化 | 解释 |
|---|---:|---:|---|
| 单序列 checkpoint / C1 | 789.35 → 689.55 | −12.64% | 旧单序列实现；不能把全部差值归因于 GDN |
| 最终多序列 / C2 | 1250.29 → 1151.41 | −7.91% | 两轮均改善；仍非长期稳定性证明 |
| 最终多序列 / C4 | 2441.23 → 2317.85 | −5.05% | 两轮分别 +0.08%、−9.06%；缓存/调度有差异 |
| 最终多序列 / C8 | 3872.57 → 4087.16 | **+5.54%** | 总体退化，不能承诺高并发稳定收益 |

C1 与 C2/4/8 来自不同实验、不同候选，不是同一条性能扩展曲线。各指标 mean/P50/P75/P90/P95/P99、每轮数据、缓存差异见[统一数据表](docs/tables.md)及[最终多序列报告](experiments/20261024-gdn-multiseq-opt/README.md)。

另测两臂都启用新 GDN，只把 P chunk 从 8192 改为 16384：C2/C4/C8 平均 TTFT 分别 **+26.65% / +14.65% / +19.29%**。该候选拒绝，保留 8K；不能因减少 launch 数就判断端到端更快。

## 最新跨 geometry checkpoint 实验

| 输入8192、目标1280 | selected V2 ms | native CP checkpoint ms |
|---|---:|---:|
| 122B TP4 geometry | 0.5356 | 0.4136 |
| 35B TP1 geometry | 0.5442 | 0.7856 |

这是普通gate的合成原型设备计时，native CP仍是数值回归未全过的候选，不代表推荐切换。修复后的checkpoint相对原CP前缀状态逐位一致；相对non-CP在near-one下仍存在差异。详见[完整对比与失败记录](docs/unified-tables.md)。

## 核心实现与验证入口

- 122B：[分流器](flashinfer/gdn_split.py)、[上游补丁](patch/0001-perf-gdn-SM90-mixed-length-split-dispatch-for-GDN-pr.patch)、[测试](tests/test_prefill_sm90_split.py)。原路径保留，避免破坏既有使用方式。
- 35B：[SGLang adapter](qwen35/runtime/service_patch.py)、[shape 分流器](qwen35/runtime/dispatch.py)、[V64 adapter](qwen35/runtime/dvsplit.py)、[V128 adapter](qwen35/runtime/selected.py)。7 个 runtime 文件必须一起使用；不混用历史单序列协议。
- [生成器](qwen35/generators/) + [固定上游源码](qwen35/upstream/) 可逐字节重建两个内核；不要手工编辑生成文件。

新包使用 [kernels/generators](kernels/generators/) 和固定源码生成三个内核；旧 `qwen35/runtime` 保持历史复现，不作为新包运行依赖。

无需 GPU 的归档校验：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/unit -v
python3 tools/regenerate_kernels.py --check
python3 tools/regenerate_35b.py --check
python3 tools/build_tables.py --check
python3 tools/build_unified_tables.py --check
python3 tools/validate_archive.py
```

这些检查验证源码、数据和归档一致性，**不是重新运行 GPU 正确性或性能测试**。原有 GPU 测试结果和限制见[验证说明](docs/validation.md)。本仓库不包含模型权重、业务提示词、模型输出正文、真实激活或完整 profiler trace。
