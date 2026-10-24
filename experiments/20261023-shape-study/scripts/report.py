"""Rebuild experiment report from saved evidence; standard-library only."""
from pathlib import Path
import collections
import difflib
import hashlib
import json

root=Path(__file__).resolve().parents[1]
ev=root/'evidence'
def rows(name):return [json.loads(x) for x in (ev/name).read_text().splitlines()]
def grouped(data):
    g=collections.defaultdict(dict)
    for r in data:g[tuple(r['lengths'])][r['strategy']]=r
    return g
def shape(ls):return '+'.join(map(str,ls))
full=rows('full.jsonl');dv=rows('dv_full.jsonl');hold=rows('holdout_slow.jsonl');near=rows('holdout.jsonl')
dg,hg=grouped(dv),grouped(hold)
validation=json.loads((ev/'validation.json').read_text())
candidate=[r for r in validation if r['method']!='noncp_vs_fp32_recurrence']
baseline=[r for r in validation if r['method']=='noncp_vs_fp32_recurrence']
assert all(r['valid'] and r['output_max_abs']==0 and r['output_state_max_abs']==0
           and r['graph_check']['output_max_abs']==0 and r['graph_check']['output_state_max_abs']==0 for r in candidate)
assert all(r['valid'] for r in full+dv+hold)
assert all(r['valid'] for r in near if r['strategy']!='public_auto')
integration=json.loads((ev/'integration.json').read_text())
assert len(integration)==5 and all(r['valid'] for r in integration)
mem=(ev/'memcheck_device.log').read_text() if (ev/'memcheck_device.log').exists() else 'pending'
summary={
    'full_rows_valid':len(full),'dv_rows_valid':len(dv),'holdout_shapes':len(hg),
    'holdout_rows_valid':len(hold),'candidate_checks_bitwise':len(candidate),
    'baseline_recurrence_checks':len(baseline),'baseline_recurrence_failures':[r for r in baseline if not r['valid']],
    'near_one_public_auto_failures':[r for r in near if not r['valid']],
    'holdout_selected':sum(d['adaptive']['dispatch_plan']['strategy']=='dvsplit' for d in hg.values()),
    'holdout_min_speedup':min(d['public_auto']['median_ms']/d['adaptive']['median_ms'] for d in hg.values()),
    'integration':integration,'device_memcheck_zero_errors':'ERROR SUMMARY: 0 errors' in mem,
}
(ev/'summary.json').write_text(json.dumps(summary,indent=2))

lines=['# Qwen3.5 35B TP1：H20 GDN prefill 调查与优化实验','',
'- 日期：`2026-10-23`（Asia/Shanghai）',
'- 运行结果：`completed`','',
'## 结论与核心结果','',
'`/prefill_model` 的 TP1 GDN shape 是 **Hq=Hk=16、Hv=32、Dk=Dv=128**。已实现隔离的 Dv=64 kernel 原型，把每个 value head 分成两个 CTA；单序列长 prefill 相对现有默认调度约 1.46–1.57×，需要分流以避免均衡 batch 退化。当前是可复现实验原型，未修改安装包、未发布、未接入运行中的服务。','',
'### 算子测量：CUDA Graph 设备时间','',
'以下来自 `dv_full.jsonl`，7 轮随机顺序，每轮每策略 100 次 graph replay，中位数。','',
'| token lengths | 默认 auto ms | 显式 non-CP ms | 全 batch Dv=64 ms | 相对默认加速 |',
'|---|---:|---:|---:|---:|']
for ls in [(512,),(1024,),(4096,),(8192,),(16384,),(32768,), (8192,1024,1024,1024),(8192,8192),(2048,)*4]:
    d=dg[ls];a=d['public_auto']['median_ms'];n=d['noncp']['median_ms'];c=d['dvsplit']['median_ms']
    lines.append(f'| {shape(ls)} | {a:.4f} | {n:.4f} | {c:.4f} | {a/c:.3f}× |')
lines+=['','### SGLang GDN extend 组件测量','',
'真实 `FlashInferGDNKernel.extend()` 的实例级适配，包含 Q/K 预处理、gate 转换、状态 gather/scatter 和两臂相同的初始状态池 reset copy；9 轮随机换序、每轮 100 次 graph replay。**不等于完整模型 TTFT 或请求吞吐。**','',
'| token lengths | 原组件 ms | 分流候选 ms | 加速 | 路由 |','|---|---:|---:|---:|---|']
for r in integration:
    a=r['median_ms']['baseline'];c=r['median_ms']['candidate']
    lines.append(f"| {shape(r['lengths'])} | {a:.4f} | {c:.4f} | {a/c:.3f}× | {r['plan']['strategy']} |")
lines+=['','### 关键观察','',
'- 原 CP split 的成本系数 `2.7*32/(0.86*78)=1.288`，仅长序列部分就超过普通路径下限；原规划器对这些 TP1 输入全部拒绝。强制扫描中多数 split 不获益，32K+3×1K 某个 chunk 仅约 1.06×。',
'- 现有 auto 对 32 heads 单序列选择 CP，短输入成本偏高；512 tokens 显式 non-CP 相对 auto 约 2.04×，这是调度收益。Dv=64 对显式 non-CP 的额外收益约 1.45–1.49×，应分开归因。',
f"- 保守分流在 26 个未用于拟合的 shape 上选择 {summary['holdout_selected']} 个 Dv=64，其余回退；最小观测加速比 {summary['holdout_min_speedup']:.4f}×，接近 1 的变化属于计时噪声。另测近 1 gate；候选仍与 non-CP 一致，两个原始 auto/CP 用例超出严格状态误差门槛而未计时。",'',
'## 复现','',
'本地目录与远程目录保留完整脚本。远程：`h20-2:/root/gdn35/20261023-shape-study`；物理 GPU 4。原始权重没有加载、复制或修改；输入是按模型配置生成的合成激活，不是实采模型激活。','',
'```bash',
'cd /root/gdn35/20261023-shape-study',
'bash scripts/01_pilot.sh',
'bash scripts/02_full.sh',
'bash scripts/03_dv_pilot.sh',
'bash scripts/04_dv_full.sh',
'bash scripts/05_validate.sh  # 默认严格基线参考检查会保留并报出两例误差',
'bash scripts/06_holdout.sh  # near_one gate',
'bash scripts/07_integration.sh',
'bash scripts/08_memcheck.sh # 保留 41 条 API 查询错误',
'bash scripts/09_holdout_slow.sh',
'bash scripts/10_memcheck_device.sh',
'```','',
'- 输入输出 BF16，状态和线性域 gate FP32；q/k L2 归一化。常规 gate=`exp(-0.05*softplus(N(0,1)))`，独立压力 gate=0.999；正确性另含 uniform gate。',
'- `benchmark.py` 的 seed、迭代次数、环境和源码 SHA256 记录在对应 `*.metadata.json`。',
'- 默认 3 次 eager warmup，先验证再捕获，交替多策略计时前显式保留所有 workspace 引用；首次 JIT 不计入 steady-state 时间。',
'- 重建实验 kernel：`python3 scripts/generate_dvsplit.py`；只从 `evidence/installed_delta_rule_sm90.py` 生成隔离文件，不覆盖安装包。',
'- `tp1_dispatch.plan()` 应每 batch 计算一次，跨 GDN 层复用，避免把 Python 规划开销放入每层热路径。这里的 graph 测量不包含 CPU 规划时间。','',
'### 实现原理','',
'V/O 由 `[T,32,128]` 视图变为 `[T,64,64]`；状态由 `[N,32,128,128]` 变为 `[N,64,64,128]`。Q/K 保持 `[T,16,128]`，成对虚拟 heads 复用原 gate，不需要复制。每 CTA 从 2 个状态 warpgroup 改成 1 个，保留 load/store 和辅助 warpgroup，总线程数 512→384；移除两个状态 warpgroup 间的有序 barrier。QK/KK 辅助工作会重复，故不能无条件对饱和 batch 启用。','',
'分流模拟 78 个 SM 的队列完成时间，分别放入每序列 32 和 64 个工作块；候选每块的有效成本系数取 0.68，要求预测加速 >1.1。它只针对当前 H20/TP1 geometry，未证明是通用最优策略。另测了 longest-only 双流版本，但当前保守分流没有纳入它。','',
'## 运行配置','',
'| 项目 | 实际值 |','|---|---|',
'| 主机/GPU | h20-2，物理 GPU 4，H20 97871 MiB，78 SM；其他任务占用 GPU 0–3 |',
'| 模型配置 | /prefill_model，Qwen3_5MoeForConditionalGeneration，FP8 权重配置；40 层，其中 30 个 linear-attention 层 |',
'| 目标 TP | 1；总 K heads=16，V heads=32，head dims=128 |',
'| 软件 | Torch 2.13.0+cu130，FlashInfer 0.6.18，CuTe DSL 4.6.2，SGLang 0.5.20，Triton 3.7.1 |',
'| SGLang commit | 94602c9c2b7cbdb8efd5c52802dac6a1c180089e |',
'| 驱动/频率快照 | 550.54.15；SM 1980 MHz，显存 2619 MHz；未锁频 |',
'| 镜像 digest | 未采集；沿用现有环境，没有升级安装包 |',
'| HTTP 端口/QPS/TTFT/TPOT/请求成功数 | N/A；未启动模型服务 |','',
'## 限制与证据','',
f"- 原版调度扫描 {len(full)} 个配置有效；Dv 扫描 {len(dv)} 个配置有效；常规 gate 独立验证 {len(hold)} 个配置有效。",
f"- 新 kernel 与 non-CP 的 {len(candidate)} 个等价检查（60 个全 batch、24 个最长序列）在 eager 和 graph 下输出与状态均逐位一致，包含零长度、非 64 对齐长度、零/非零初始状态和三种 gate。",
'- FP32 token recurrence 参考检查 34/36 通过；两例原 non-CP 在 257 tokens、gate=0.999 时状态最大绝对误差约 0.0114/0.0117，略超 0.01 门槛。候选与原 non-CP 完全一致。保留原始失败状态，不扩大容差掩盖。',
'- near_one holdout 中原 auto/CP 的 1536、3072 tokens 状态误差分别约 0.0112、0.0130，未进入这两个基线的性能计时；常规 gate 重跑所有基线均通过。',
f"- 设备 memcheck（关闭 API 返回码报告，开启 stream-ordered race 检查与 32-byte allocation padding）：{'ERROR SUMMARY: 0 errors' if summary['device_memcheck_zero_errors'] else '结果尚未确认，见日志'}。默认 memcheck 的 41 条错误全部是原始 kernel 编译时 cuGetProcAddress_v2 的 CUDA API 查询错误；原日志未删除。",
'- 首次调度 pilot 因测试 harness 未保留旧 CP workspace 引用而失败；已修复并重跑，失败日志留存。第一版组件测试也因未显式保留外部 pool 引用而作废，最终结果来自修复后的重跑。',
'- 原型只验证 BF16 I/O、FP32 packed state、无 checkpoint/无 state_indices 的 TP1 路径；组件适配对 checkpoint 回退原入口。尚未验证完整服务、真实激活、模型输出质量、并发跨 stream 的多请求调用、训练或其他 GPU。',
'- 未测 TTFT 或端到端吞吐；未升级到原仓库的 FlashInfer 0.7.2-dev 基线，因此不能把本结果称为那个版本上的上游补丁验证。',
'- infer-kb ensure 被既有 `results/sglang/pd-dflash-profile/20260907-tp4-1p1d/README.md` 缺少合法运行结果阻断；遵循失败停止写入约束，本目录是独立完整记录，未归档为知识库 Result/Case。','',
'### 证据索引','',
'| 文件 | 内容 |','|---|---|',
'| [environment.json](evidence/environment.json) | 模型配置、版本、GPU 快照、源码哈希 |',
'| [full.jsonl](evidence/full.jsonl) | 原调度、CP chunk、原 split 扫描 |',
'| [dv_full.jsonl](evidence/dv_full.jsonl) | Dv 全 batch 与 longest-only 对照 |',
'| [holdout_slow.jsonl](evidence/holdout_slow.jsonl) | 独立 shape 的分流性能 |',
'| [holdout.jsonl](evidence/holdout.jsonl) | 近 1 gate 压力检查 |',
'| [validation.json](evidence/validation.json) | 逐位等价与 FP32 参考全部原始结果 |',
'| [integration.json](evidence/integration.json) | 真实 SGLang extend 组件计时 |',
'| [memcheck_device.log](evidence/memcheck_device.log) | 设备内存与 stream race 检查 |',
'| [generate_dvsplit.py](scripts/generate_dvsplit.py) | 可审计、可重建的 kernel 变换 |',
'| [dvsplit_kernel.py](scripts/dvsplit_kernel.py) | 生成的实验 kernel |',
'| [dvsplit.py](scripts/dvsplit.py) | 零复制视图适配层 |',
'| [tp1_dispatch.py](scripts/tp1_dispatch.py) | 保守的 TP1 分流原型 |',
'| [dvsplit.patch](dvsplit.patch) | 相对已安装 kernel 源码的变更 |','']
(root/'README.md').write_text('\n'.join(lines))
old=(ev/'installed_delta_rule_sm90.py').read_text().splitlines(True)
new=(root/'scripts/dvsplit_kernel.py').read_text().splitlines(True)
(root/'dvsplit.patch').write_text(''.join(difflib.unified_diff(old,new,fromfile='installed_delta_rule_sm90.py',tofile='dvsplit_kernel.py')))
(root/'manifest.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n' for p in sorted((root/'scripts').glob('*')) if p.is_file()))
print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2))
