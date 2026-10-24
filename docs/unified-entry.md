# 统一入口、checkpoint 契约与代码维护

统一包为 `gdn-prefill-h20`，Python import 为 `gdn_prefill`。公共入口是 `GDNPrefill.prepare()` / `run()`。两个模型按 rank-local geometry 与 batch shape 分流，不按模型名称分流。默认策略 `stable` 表示不自动采用新的实验性 CP checkpoint，并不是模型质量、所有负载或完整服务已经认证的声明。

## 安装与兼容范围

```bash
pip install --no-deps .
PYTHONPATH=src python3 -m unittest discover -s tests/unit -v
python3 tools/regenerate_kernels.py --check
python3 tools/validate_archive.py
```

运行环境需自行安装 Torch/CUDA、FlashInfer 0.6.18 与 CuTe DSL。实测环境是 Torch 2.13.0+cu130、CuTe DSL 4.6.2、H20 SM90 78 SM；包不会自动安装或替换这些依赖。优化后端在进程内首次准备时校验 FlashInfer 私有源码 SHA256，避免相同版本号下源码不同却静默复用内核。原 122B 的 0.7.2dev 补丁仍独立保留在 `patch/`，不能和新包假定兼容。

正式运行代码只在 [src/gdn_prefill](../src/gdn_prefill/)；不会从实验目录导入实现。`qwen35/runtime/` 是历史实测代码的冻结副本，用于复现历史服务，不再是新包的修改入口。

## 使用示例

以下 q/k/v/g/beta、初始状态和输出由调用者提供；输出 buffer 必须提前分配，不能与输入共享 storage。

```python
from gdn_prefill import GDNPrefill, Geometry, CheckpointRequest

op = GDNPrefill(policy="stable")
plan = op.prepare(
    lengths=(4096, 4096),
    geometry=Geometry(hq=16, hv=32),  # 122B TP4 使用 hq=4, hv=16
    checkpoint_requests=(
        CheckpointRequest(sequence=0, token=2048, slot=0),
        CheckpointRequest(sequence=1, token=1280, slot=1),
    ),
    device=q.device,
)
op.run(
    plan=plan, q=q, k=k, v=v, g=g, beta=beta,
    initial_state=initial_state, output=output, output_state=output_state,
    checkpoint_states=checkpoint_states,
)
print(plan.backend, plan.spec.reason)
```

- Q/K: `[sum(lengths), Hq, 128]`，V/O: `[sum(lengths), Hv, 128]`，BF16 连续布局。
- g/beta: `[sum(lengths), Hv]`，FP32；g 是 **linear-space gate**，不是 log-space。
- initial/final state: `[N, Hv, 128, 128]`，FP32、V×K 排列。initial 可以为 None，final output 必须由调用者提供。
- checkpoint output: `[max_requested_slot+1, Hv, 128, 128]`，FP32；未请求槽位和空序列的 final state 保持不写。
- 输入应已完成模型要求的 Q/K normalization；本包不隐式加入 normalization，scale 固定为 `Dk**-0.5`。不支持 indexed state pool、自定义 scale 或非连续布局等未实现接口，不能用它替换所有 FlashInfer 调用。
- prefix sums 由 plan 持有，避免每层从 GPU 回读长度。plan 可在具有相同 geometry、长度和保存需求的层之间复用。

## 计划与回退

`CheckpointRequest(sequence, token, slot)` 中 token 是当前 extend 的相对位置，含义为“完成前 token 个输入后的状态”。只接受正的、64 对齐、范围内的位置。slot 必须非负且唯一，可以有空洞；多条请求可对应同一序列。

当前 selected full-V/V2/native-CP kernel 每序列至多写一个目标。多目标自动走 original dense-64 计算，再使用预分配的 gather/scatter buffer 写出所有请求目标。这可能增加显存和开销，但不会丢 checkpoint。非法位置直接报错；原始调用本身不支持的需求也不能被描述成已实现的回退。

| auto 情况 | 行为 |
|---|---|
| Hq4/Hv16，无 checkpoint，混合长短序列通过旧成本模型 | mixed CP；否则保留 upstream auto |
| Hq16/Hv32，无 checkpoint | 保留 upstream auto |
| 两种已测 geometry、N≤8、每序列至多一个 checkpoint | H20 参数化 CTA 估计选择 full-V / V2 |
| 多目标 checkpoint、N>8、未测 geometry | original；有 checkpoint 时 dense-64 + 精确目标 scatter |
| native CP checkpoint | auto 永不选择；必须指定 experimental policy 与 cp_checkpoint backend |

参数化估计复用实测系数 `.67` 和 15% 的选择余量，Hv 和 SM 数来自 geometry。它是 H20 上的经验模型，不是泛化到任意硬件的 autotuner。没有在线测速，也不读取 gate 内容决定分流。

```python
experimental = GDNPrefill(policy="experimental")
plan = experimental.prepare(
    lengths=(8192,), geometry=Geometry(4, 16),
    checkpoint_requests=(CheckpointRequest(0, 1280, 0),),
    device="cuda", backend="cp_checkpoint",
)
```

原型 native CP 使用合并状态 buffer；公共入口会把 final/checkpoint 复制到调用者 buffer，并在 GPU 上保留未请求槽位。旧实验的 0.4134 ms 是当时原型的调用方式，不包含新入口新增的外部 scatter，不能直接当成这个 API 的完整延迟。

## CUDA Graph 与资源生命周期

在 capture 外 prepare，使用实际输入 warm up 后再 capture run。首次 JIT/工作区初始化仍需预热；不能直接把第一次调用放入 capture。plan 持有元数据、dense fallback buffer、CP 中间缓冲引用和 side stream，必须存活到对应 graph 不再使用为止。

`plan.update_checkpoints(requests)` 可在 replay 前、同一有序 stream 上更新 selected full/V2 的目标及 slot，前提是每序列仍至多一个目标且不超过原容量。native CP 还要求 sequence→slot 映射保持不变，仅允许变更目标/停用目标。改变长度、后端、容量或从无 checkpoint 开启 checkpoint 时应重新 prepare/capture。

不同设备应使用各自的 plan。上游 CP 使用进程级 workspace cache，本版本只支持每设备按依赖有序调用；不支持多个独立线程/stream 同时重入 CP。mixed CP 只让 CP 长序列与 non-CP 短序列重叠，并显式建立 stream 依赖，不代表任意 CP 调用可并行。

## SGLang 接入

[integrations/sglang.py](../src/gdn_prefill/integrations/sglang.py) 提供显式 CPU metadata bridge `requests_from_tracking()`，沿用已审计 35B consumer 的 relative-length/interval 规则。它不在 import 时修改 SGLang，不写日志，也不读取历史配置路径。

接入时在 scheduler metadata 构造阶段转出请求并 prepare；各层运行后，按原 consumer 的 destination 映射复制到持久状态池。精确 interval 对齐的状态仍走原 final-state 消费流程。若 consumer 的含义不同、targets 非法或接口不匹配，保留原 planner/调用，不绕过检查。

**新公共入口及 bridge 已进行独立算子/metadata 检查，但本次没有把新包替换进完整服务，也没有新增122B或35B端到端结果。** 历史35B在线shadow/E2E属于原hook；122B真实消费者仍需独立接入验证。

## 维护和复现

- 修改 [kernels/generators](../kernels/generators/)，从固定 [upstream](../kernels/upstream/) 生成 [私有内核](../src/gdn_prefill/_kernels/)。禁止手改生成文件。
- `python3 tools/regenerate_kernels.py --check` 校验三个内核逐字节一致。
- `CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src python benchmarks/validate_entry.py --out validation.json` 运行新入口的 GPU 差分检查；只在指定卡空闲且有授权时执行。
- 公共入口 GPU 记录见 [统一入口验证](../experiments/20261024-unified-entry/README.md)；旧实验保持冻结，不把新代码冒充历史计时使用的版本。
