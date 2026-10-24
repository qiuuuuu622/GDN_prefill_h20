# checkpoint 的统一契约与实现状态

122B 与 35B 的 checkpoint 含义相同：处理完指定 token 前缀后的 GDN 递推状态。差异来自 kernel 优化覆盖范围、服务消费者和验证阶段。

现在仓库已提供 [统一入口](unified-entry.md)，共享 `CheckpointRequest`、目标校验、full-V/V2 selected kernel 和验证流程。每条序列可提出多个保存请求；当前单目标内核不支持时，完整 dense fallback 保存全部目标。服务优化不得自行删去消费者的目标。

## 原始覆盖与新增工作

| 项目 | 原 122B TP4 | 原 35B TP1 | 新统一包 |
|---|---|---|---|
| 无 checkpoint | CP/混合CP优化 | 保留原版auto | 按 geometry 保留对应行为 |
| selected checkpoint | 原补丁不覆盖 | 已审计真实consumer，full/V2 | 两种geometry共享full/V2；122B为算子证据 |
| CP checkpoint | 不覆盖 | 不启用 | 原型已修复边界写出，仍仅实验 |
| 端到端 | 尚未完成 | 历史1P1D C1/2/4/8 | 新包尚未进行服务替换AB |

## CP 边界修复

旧实验直接取 fixup workspace 作为checkpoint。fixup分块仿射组合与最终prefill实际递推，在低精度operand转换和舍入下并非逐位一致。新实现从最终prefill的FP32运行状态直接写出目标边界，并保留CP chunk=2048；因此token1280等chunk内部64对齐位置也可保存。

122B/35B、near-one gate、8192输入/4096checkpoint的同条件最大误差相对原版CP前缀状态分别从0.012661/0.014820降到0，输出与最终状态不变。112项边界检查、200次动态图回放和32项memcheck均通过。

**参考是原CP前缀，不是non-CP。** 普通gate对non-CP完整检查56/56通过；near-one仅19/56通过。强制FP32 fixup也没有解决全部失败。不能据此宣布CP/non-CP等价或模型质量通过，因此auto不会选native CP checkpoint。

完整原始数据、实现、失败项及计时见 [CP修复实验](../experiments/20261024-gdn-cp-checkpoint-fix/README.md) 和 [同版本对比表](unified-tables.md)。

## 消费者与执行语义

1. 入口使用extend相对token位置；服务adapter负责绝对前缀坐标转换。
2. 初始状态、序列范围、目标token和输出slot明确对应。target是处理完前target个token之后的状态。
3. 持久状态池映射和final-state消费者保持原义。临时布局可改变，不允许省略请求的状态。
4. 输出、final state、各checkpoint分别验证，再检查恢复后的输出与状态。
5. empty sequence、非整块尾部、多个目标、无目标和动态graph metadata分别覆盖。
6. 新包目前只支持明确列出的布局与dtype；不能把一个geometry的服务hook直接推广到其他consumer。

共享入口不意味着强制共享算法。CP checkpoint进入默认策略之前，仍需要解决数值验收并完成真实消费者和端到端回放；35B已有性能结果也不支持为了统一而改用CP。
