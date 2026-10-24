> 历史复现目录：新的可安装统一入口在 [src/gdn_prefill](../src/gdn_prefill/)，用法见 [统一入口](../docs/unified-entry.md)。本目录冻结实测版本，不作为新包依赖。

# 35B TP1 核心实现与复现

这里的 `runtime/` 是最终多序列 checkpoint 实验的 7 个原样运行文件，与 `20261024-gdn-multiseq-opt` 的 release 哈希一致。历史版本在 `experiments/`，不要混用。

## 文件职责

| 文件 | 职责 |
|---|---|
| `sitecustomize.py` | 在进程内拦截 SGLang GDN 模块导入，不修改安装目录 |
| `service_patch.py` | 原 planner 语义适配、紧凑 checkpoint、回退和可选 shadow |
| `dispatch.py` | H20 78 SM / 32 heads 的完整 V 与 V2 选择 |
| `dvsplit.py` | V64 零拷贝视图与内核调用 |
| `dvsplit_kernel.py` | 生成的 V64 kernel，不能手改 |
| `selected.py` | 完整 V 的选定 checkpoint 调用 |
| `selected_kernel.py` | 生成的 V128 kernel，不能手改 |

已验证环境：Linux/CUDA，H20 78 SM，Torch 2.13.0+cu130、CuTe DSL 4.6.2、FlashInfer 0.6.18、SGLang 0.5.20（实验记录 commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`）、Dynamo 1.5.0、NIXL 1.4.1。没有镜像 digest；不能据此宣称跨版本兼容。

只针对 `Hq=Hk=16,Hv=32,Dk=Dv=128`、BF16 Q/K/V、连续 FP32 初始状态、无 indices/内核 L2norm/自定义 scale 的已验证路径；N≤8。其他路径回退原版。不是 pip 包或自动探测版本的通用插件。

## 加载与撤销

在与实验相同的软件环境，**仅 P 进程**的现有启动命令前加入：

```bash
export GDN_RUNTIME=/absolute/path/to/GDN_prefill_h20/qwen35/runtime
PYTHONPATH="$GDN_RUNTIME${PYTHONPATH:+:$PYTHONPATH}" \
  GDN35_PATCH=1 GDN35_VARIANT=multi \
  python3 -m dynamo.sglang [原有的完整 P 参数]
```

`[原有的完整 P 参数]` 是占位说明，不能原样执行。实际参数数组见 [P 配置](../experiments/20261024-gdn-multiseq-opt/config/p_multi_fixed.json)；[原版 P](../experiments/20261024-gdn-multiseq-opt/config/p_stock_a1.json)、[D 配置](../experiments/20261024-gdn-multiseq-opt/config/d_default.json)、[完整启动器](../experiments/20261024-gdn-multiseq-opt/scripts/03_launch.py) 可逐项审计。

实验为 GPU6=P / GPU7=D，TP1、context262144、mem fraction0.80、page16、max-running16、DFLASH8、NIXL；max-running16 是容量，不是负载并发。P 模型是 FP8，D 模型是 4-bit，两边使用对应模型副本与同一 draft。没有强制替换为 Triton GDN；实际 P 的已验证路径为 FlashInfer，BCG 日志 backend=`breakable`。8K chunk 为该版本实际配置，后续 16K 实验已拒绝。

D 不加载 hook。原版 A 也不加载 hook；撤销时移除 `GDN35_PATCH` 和 `GDN35_VARIANT` 后重启 P。`multi` 固定模式不读取实验目录的 mode 文件，不写 shadow 日志。`verify_multi` 是历史审计模式，需要归档目录、captures 等配套设施，不能当成正式计时模式。

## 重建内核（无需 GPU）

```bash
python3 tools/regenerate_35b.py --check
# 要修改 kernel：编辑 generators 中的变换，再生成；不要手改生成文件。
python3 tools/regenerate_35b.py
```

包装器在临时目录运行历史生成器，使用 `upstream/installed_delta_rule_sm90.py` 固定输入。`--check` 不修改文件，并要求结果与实测 runtime 字节一致。

## 重现 GPU 检查与服务实验

原测试代码在 [多序列 scripts](../experiments/20261024-gdn-multiseq-opt/scripts/)：

- `02_multi_test.py`：72 组合成边界、状态、续算及 Graph 回放；其 memcheck 模式见脚本参数。
- `03_planner_check.py`：原版映射与紧凑映射、回退检查。
- `04_dispatch_sweep.py`：完整 V、V2、原版同输入计时。
- `05_real_replay.py`：捕获激活回放；原始 `.pt` 未公开。
- `20_concurrent.py`：C2/C4/C8、预热、清缓存、AB/BA 控制器。

这些是历史实验脚本，包含固定实验目录、模型路径及端口，**不是任意机器直接运行的启动器**。完整复现实验需要相同软件、可用 GPU、模型副本、独立端口及授权数据；先按实验目录结构配置路径。不要直接运行历史停机脚本去操作来源不明的 PID。旧 shape-study 使用 GPU4 是历史事实，不是当前 GPU 分配建议。

没有固定十条的私有 payload，不能重现完全相同的 E2E 输入；仓库提供哈希、选择规则、请求长度、数值结果及不依赖 payload 的汇总校验。也不能运行依赖私有 SSE 输出的历史全文一致性汇总；公开 `tools/build_tables.py` 可独立核对延迟表。

当前正确性保证是已测路径对原版无新增差异。C8 E2E 仍退化；不代表生产上线验收或长期稳定性认证。
