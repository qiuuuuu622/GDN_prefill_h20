# 优化机制与选择依据

## 1. GDN 的两个不同并行维度

用数学布局 S∈R^(K×V) 表示状态，一个简化的 GDN 更新是：

```text
S_decay = alpha * S_previous
e       = v - S_decay.T @ k
S       = S_decay + beta * outer(k, e)
o       = S.T @ q
```

实际内核用 chunk 算法重排运算，物理状态布局可为 `[V,K]`；这里仅解释依赖关系。

**沿 V 切分**：每组 V 列依赖自己的旧状态和对应 v 分量，共享 q/k/gate。不同 V 分片不用互相归约。当前 35B 实现将 `[T,32,128]` 的 V/O 视图变为 `[T,64,64]`，状态 `[N,32,128,128]` 变为 `[N,64,64,128]`；分片读写地址不重叠，不复制整个激活。

**沿 token 切分（CP）**：后一段需要前一段的最终状态，不能直接当成独立序列。SM90 CP 用 T 预计算、MN 预计算、fixup、prefill 处理分段变换和入段状态，额外工作换取 token 方向并行。CP 与 V 数学上兼容，但实现和开销不同。

## 2. 122B TP4：为什么最长序列做 CP

每卡 Hv=16。原 non-CP 主要阶段是一条序列的每个 V head 一个 CTA；单条长序列只有 16 个主要计算 CTA，H20 有 78 SM，容易出现并行度不足。原报告的“每个活跃 SM 已较高效”是该基准的测量，不能推广到所有 GDN。

混合长度下，例如 `[8192,1024,1024,1024]`，短序列很快完成，之后主要由最长序列的 16 个 CTA 决定尾部时间。原 auto 按整个 batch 的 `N*Hv` 判定，无法充分表达这种长度不均衡。

实现将最长序列交给旁路 stream 做 CP，其余序列在当前 stream 走 non-CP。输入/输出和状态均为切片视图；当前 stream 等待旁路结束后再消费结果。双 stream 的并发不等于计算免费，分流器因此需要收益门槛。

```text
base  = max(L, total_tokens * Hv / SMs)
split = max((2.7 * L + rest_tokens) * Hv / (0.86 * SMs), second_longest)
accept when L >= 4096 and split * 1.1 < base
```

CP chunk 选使 `ceil(L/chunk)*Hv <= SMs` 的最小 `64×2^k`。`2.7`、`0.86` 是 H20 拟合系数，不是算法常数。

当前补丁只接受 auto、CPU 前缀和、至少两序列、无 state_indices、无 checkpoint；不新增逐层 GPU→CPU 长度同步。入口和条件见 [gdn_split.py](../flashinfer/gdn_split.py) 与 [patch](../patch/0001-perf-gdn-SM90-mixed-length-split-dispatch-for-GDN-pr.patch)。

## 3. 35B TP1：为什么不照搬混合 CP

35B 每卡 Hv=32，比 122B TP4 多一倍。下面仅是主要阶段的 CTA 数估算，不是实际 occupancy 或所有阶段的 launch 数：

| 单序列 | 122B TP4 | 35B TP1 |
|---|---:|---:|
| 完整 V | 16 | 32 |
| V 分为两份 | 32 | 64 |
| token 分成四段的主要阶段 | 64 | 128 |

35B V2 已产生 64 个主要 CTA，继续通过 CP 增加并行的边际价值更小，却仍要付出预计算和状态修正代价。原混合 CP 成本模型的长序列项系数在 Hv32 时为 `2.7*32/(0.86*78)≈1.288`，它本身就高于理想 non-CP 的 L 项。这个模型不证明 CP 永远无效；它解释原门控为何拒绝这些 shape。

实际扫描比模型推测更重要：[35B CP/V 表](../data/35b_cp_vs_v.csv)显示 8192 单序列 non-CP≈0.754 ms、最快扫描 CP≈0.762 ms、V2≈0.506 ms；32K 的 CP 已有小幅收益，仍慢于 V2；均衡双 8192 则完整 V 最快。CP 与 V 来自不同历史扫描，不能包装成一次同时计时实验。

512-token 的接近 3× 是相对原 auto，包含“绕开不划算的 CP”与“V 切分”两部分；不是 V 切分单独带来 3×。正式服务最终版本保留无 checkpoint auto-CP，并没有全面复用这个早期原型。

## 4. V 切分为什么也不能无条件启用

V64 内核把两组状态 warpgroup 改成一组，总线程数 512→384，减少每 CTA 状态工作并增加 CTA 数。Q/K、部分 QK/KK 辅助运算会重复，更多 CTA 还可能产生更多执行波次。

- 单条长序列：32→64 CTA，可减少并行度不足。
- 两条长度相近：完整 V 已约 64 CTA，V2 变约 128 CTA，重复工作与额外波次可能使其变慢。
- 两条长短悬殊：短序列退出后，完整 V 剩长序列的约 32 CTA；V2 可以缓解这个尾部。

最终 [dispatch.py](../qwen35/runtime/dispatch.py) 将各序列 `ceil(length/64)` 作为工作长度，分别模拟 32/64 个任务分配到 78 个槽位。单序列选 V2；多序列仅在 `0.67*makespan64 < 0.85*makespan32` 时选 V2，即保留约 15% 预测收益余量。按长度元组 LRU 缓存 1024 项，在 batch 元数据阶段选定，不在每个 GDN 层重新规划。系数只对当前 H20/shape 有实验依据；顺序、资源竞争和真实 occupancy 并未被完全建模。

## 5. checkpoint 精简：减少临时写出，不减少模型状态计算

原消费者已经指定所需中间状态：`track_ssm_h_src` 索引临时 checkpoint，再拷贝到 `track_ssm_h_dst` 对应的持久状态池。原 kernel 通常每 64 tokens 写出中间状态，消费者只选其中少数。

新版沿用原 mask、目标位置及边界取整，对当前已验证的每序列至多一个中间目标，构建 `[destination_index, target_token]` 元数据，仅在该边界写出。随后重映射临时源索引，持久目标不变；每个 token 的运算、最终状态与输出均保留。

例如单序列 8192 tokens 的临时 checkpoint：原来 128 份，每份 `32×128×128×4=2 MiB`，共 256 MiB；若消费者只需一个，则写出 2 MiB。这个比例是**该临时状态写出量**，不是整个模型显存、总带宽或端到端耗时比例。

多序列使用私有 interval −1（V2）/−2（完整 V）及紧凑元数据。它不是 FlashInfer 公共 API；planner、adapter、两个 kernel 必须作为整体使用。条件不满足时，必要时重建原 dense 输出再选取相同状态。合成检查覆盖 N≤8，真实服务 shadow 本轮实际覆盖 N=1/2/3；客户端并发 8 不等于算子 N=8。

## 6. CP、V 和 checkpoint 能否组合

“122B 无 checkpoint”仅描述现有补丁的覆盖限制，不代表122B服务不需要中间状态。两边应先统一保存/恢复契约，见[checkpoint统一方案](checkpoint-contract.md)。

理论上可以；当前代码不是开关组合。CP 内核接口约束 Q/K/V 的 D=128，而 V2 内核为 K128/V64；CP 与 checkpoint 当前也存在显式不支持路径。组合需要改 workspace、TMA、状态修正和最终 checkpoint 写出，并确保 checkpoint 是正确入段状态修正后的状态。

合理方向是复用 V 无关的 T/M，切分 V 相关的 N、状态和输出；盲目把整套 CP 预计算复制两遍可能得不偿失。这些组合尚未实现/验证，不能列为已有收益。

**新增同版本证据的选择**：无 checkpoint 时保留原有分流；需要 checkpoint 时，两种 geometry 共享 full-V/V2。新增0.6.18实验已对122B geometry做同条件比较：长单序列的native CP候选更快，均衡合批V2可能更快；CP checkpoint数值回归未全过，暂不进入默认策略。完整数据见[跨geometry表](unified-tables.md)。

## 7. 为什么 GDN 变快不代表 TTFT 同比例变快

TTFT 包含 frontend、排队、各模型层、状态处理、KV 交接及 D 端首 token。35B C4 原版诊断 trace 中，checkpoint GDN 主内核累计约 176.04 ms，无 checkpoint CP 四阶段约 184.54 ms；两类主路径共占 7188.94 ms kernel 累计的约 5.02%。这不包含所有卷积、norm 等 GDN 周边工作，也不是 TTFT 的严格上限。

GPU kernel 累计、GPU 时间并集和客户端墙钟不是同一种时间；高负载排队可能放大局部变化。A/B profile 的缓存与 batch 布局不同，所以不把两个 trace 总时间之比当作内核因果收益。累计 CUDA 同步等待约 99.6% 与观测 GPU 活动重叠，也不等于存在相同长度的可消除空闲或 NIXL 传输浪费。

提高并发后的缓存形成时机、合批和自然输出长度改变了实际工作量。因此本仓库同时保留同输入算子回放、在线正确性与真实闭环 E2E，分别回答不同问题。

## 统一后的核心路径与 CP 边界修复

当前正式公共入口在 `src/gdn_prefill`，CPU准备阶段决定算法、checkpoint目标与工作区，GPU执行阶段只执行已选择路径。head数由rank-local geometry提供，旧35B固定32/64 CTA估计改为Hv/2Hv；系数仍来自H20实测，不声称支持任意架构。详见[入口与资源生命周期](unified-entry.md)。

以精确算术描述CP分块，一个chunk可概括为 `S_out = M_chunk × S_in + N_chunk`。MN预计算和fixup把多个chunk组合，再把入段状态提供给最终prefill。但真实kernel会将状态转换成低精度MMA operand并在不同阶段舍入，因此fixup的边界workspace不保证等于最终prefill实际运行到该边界的accumulator。旧提取方法忽略了这个区别。

新native checkpoint在最终prefill每个64-token block计算完后检查目标位置，直接写出FP32 accumulator，不改变该prefill的运算，也不要求目标与CP chunk边界重合。目标1280因此无需把CP chunk降到256，减少了小chunk的预计算/fixup开销；目标4096原本就对齐，速度变化小，主要收益是修复状态来源。

这只去掉额外的“错误状态来源”偏差，没有消除原CP分块与non-CP之间的数值差异。near-one gate保留较长历史，原有差异更容易积累；当前仍有回归失败，auto分流拒绝这个实验候选。两种geometry的同条件数据和全部失败记录见[新对比表](unified-tables.md)。

本次统一还保留两层不同的验证：历史full/V2核心kernel的逐位测试，以及新prepare/run封装的差分与graph检查。二者都不能替代完整服务AB。新公共API为CP checkpoint增加caller-buffer scatter，其耗时不能直接套用无scatter原型数据。
