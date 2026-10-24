import json,pathlib,csv,hashlib
from collections import Counter
R=pathlib.Path(__file__).resolve().parents[1]
def read(name):return json.loads((R/'results'/name).read_text())
def csvfile(name,rows):
 with (R/'results'/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
correct=read('correctness.json');bench=read('benchmark.json');cp=read('cp_boundary.json');edge=read('edge_memcheck.json');prefix=read('cp_prefix_checks.json');dispatch=read('dispatch.json');empty=read('empty_plan.json')
assert len(empty)==4 and all(v['eager']['exact'] and v['graph']['exact'] for v in empty)
assert len(correct)==300 and len(bench)==16 and len(cp)==24 and len(edge['cases'])==16 and len(prefix)==8
exact=[r for r in correct if r['method']!='segmented_cp_selected'];seg=[r for r in correct if r['method']=='segmented_cp_selected']
assert len(exact)==240 and all(r['exact'] for r in exact)
assert 'ERROR SUMMARY: 0 errors' in (R/'logs/memcheck.log').read_text()
summary=dict(exact_cases=len(exact),exact_passed=sum(r['exact'] for r in exact),segmented_cp_cases=len(seg),segmented_cp_close=sum(r['close'] for r in seg),boundary_cases=len(cp)*2,boundary_close=sum(r['checks'][k]['close'] and all(z['close'] for z in r['resume'][k]) for r in cp for k in ['cp_boundary','mixed_boundary']),boundary_same_cp_output_final_exact=all(r['checks']['cp_boundary']['same_cp_output_exact'] and r['checks']['cp_boundary']['same_cp_final_exact'] for r in cp),edge_cases=len(edge['cases']),empty_plan_cases=len(empty),invalid_plans_rejected=edge['invalid_plans_rejected'],memcheck_errors=0,dispatch_best_observed=sum(r['over_best']==1 for r in dispatch),dispatch_cases=len(dispatch),benchmark_shapes=len(bench),e2e_tested=False,published=False)
(R/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
rows=[]
for r,d in zip(bench,dispatch):
 t=r['timing'];a=t['original_dense']['median_ms'];f=t['full_selected']['median_ms'];v=t['dv_selected']['median_ms']
 rows.append(dict(model=r['model'],lengths=str(r['lengths']),targets=str(r['targets']),original_dense_ms=a,full_selected_ms=f,dv_selected_ms=v,choice=d['choice'],chosen_ms=d['selected_ms'],speedup=d['speedup'],dense_checkpoint_bytes=r['dense_checkpoint_bytes'],selected_checkpoint_bytes=r['selected_checkpoint_bytes']))
csvfile('checkpoint_comparison.csv',rows)
cp_rows=[]
for r in cp:
 for name in ['cp_boundary','mixed_boundary']:
  c=r['checks'][name]
  cp_rows.append(dict(model=r['model'],gate=r['gate'],lengths=str(r['lengths']),targets=str(r['targets']),method=name,cp_chunk=r['cp_chunk'] if name=='cp_boundary' else r['mixed_cp_chunk'],close=c['close'],resume_close=all(z['close'] for z in r['resume'][name]),output_max_abs=c['output']['max_abs'],final_state_max_abs=c['output_state']['max_abs'],checkpoint_max_abs=c['state_checkpoints']['max_abs'],median_ms=r['timing'].get(name,{}).get('median_ms')))
csvfile('cp_boundary_comparison.csv',cp_rows)
text='''# 122B TP4 / 35B TP1：统一 checkpoint 的实测结论

运行结果：completed。2026-10-24，h20-2，仅GPU6；没有加载模型权重、启动服务或修改安装目录。本轮是每rank合成算子实验，不是122B TP4端到端，也没有新增35B E2E结论。仓库推送暂停，先交付本轮实验分析。

## 结论

1. **可以统一保存计划与完整V/V2 kernel。** 本轮直接复用35B已测kernel文件，分别输入Hq4/Hv16与Hq16/Hv32，两种geometry的240项完整/选定checkpoint检查全部逐位一致。统一的是消费者要求的保存语义，不能擅自改消费者目标。
2. **分流要带上每rank Hv，不能照搬35B的硬编码32 heads。** 两条4096，122B应选V2，35B应选完整V。将原估算器参数化Hv、SM数，保留原0.67/0.85门槛，在本轮16个shape上均选中完整V/V2中实测更快者；这是有限样本验证，不是通用最优证明。
3. **CP可以接同一个保存计划，但不能直接作为已验收后端。** 新增两个探针：在保存位置处分段调用CP；复用原CP fixup产生的已修正边界状态。前者60项中18项容差失败，后者两种后端48项中23项失败，均保留结果。near-one压力不能通过时，不扩大阈值来放行。
4. **checkpoint位置影响CP收益。** 对齐2048边界时122B CP可能更快；目标1280使对齐chunk缩至256后，CP反而慢于V2。因此将来CP选择器除了长度/Hv，还必须考虑目标位置和对齐代价。
5. 122B之前没有checkpoint优化，是补丁覆盖范围/实验阶段不同，不是122B模型不需要checkpoint。本轮支持“把精简写出与V2核心共享给122B”的方向；真实122B服务消费者、目标数量、接口版本和E2E覆盖仍待审计。

## 实验环境和公平性

- 两组统一使用安装版FlashInfer0.6.18，SGLang0.5.20、Torch2.13.0+cu130、CuTe DSL4.6.2，H20 78SM。原122B仓库基线0.7.2-dev不同，本轮不能冒充那个版本的直接补丁复现。
- 122B TP4每rank：Hq=Hk4/Hv16；35B TP1：Hq=Hk16/Hv32；Dk=Dv128，BF16 Q/K/V/O、FP32状态和linear gate。未运行FP8权重层。
- 固定seed20261024，q/k归一化；gate=exp(-0.05*softplus(N))与0.999压力；非零初态，额外零初态/空序列检查。
- checkpoint基线强制原non-CP dense、每64tokens写出；同输入比较full_dense/dv_dense/full_selected/dv_selected。无checkpoint CP/auto/mixed另列于原始JSON，不混作相同工作契约。
- CPU构建plan、分配输出和index张量不进入设备计时。3次预热；每shape的候选先校验再捕图；7轮固定随机换序，每轮50次CUDA Graph replay，报告中位数，保留所有样本和顺序。
- CP临时workspace保持引用，避免不同shape/chunk重分配导致图悬空；混合路径显式stream依赖。完整模型CPU规划成本未测。
- 原版等价路径要求atol=rtol=0；CP候选输出atol/rtol=.02，状态/checkpoint各.01。所有失败保留，失败候选不进入其对应性能计时。

## 同输入checkpoint对比

单位ms，speedup=原dense/选中候选。每条序列请求一个中部64对齐checkpoint；零目标/不齐长度见正确性及CP探针。完整V与V2均只保存实际指定状态。

| 模型 | lengths | 原dense | 选定完整V | 选定V2 | 分流 | 加速比 |
|---|---|---:|---:|---:|---|---:|
'''
for r in rows:text+=f"| {r['model']} | {r['lengths']} | {r['original_dense_ms']:.4f} | {r['full_selected_ms']:.4f} | {r['dv_selected_ms']:.4f} | {r['choice']} | {r['speedup']:.3f}× |\n"
text+='''
原版全量写出→选定完整V，隔离了checkpoint写出优化；再比较选定完整V→选定V2，才是V切分的额外收益。原non-CP与原auto并非同一基线，不能把短序列绕开CP的收益全部算给V切分。

## CP边界复用：可行性与代价

CP fixup输出的是各CP chunk末的已修正状态，下一段prefill会读取前一段对应状态。探针执行原四阶段pipeline，在后面用index_select将所需边界状态复制到输出槽位；该gather计入时间。没有修改CP算术。混合探针只让最长序列做CP，其余序列用选定完整V，避免短序列目标迫使整batch的CP chunk变小。

此方案只接受目标落在CP边界；未对齐时在实验中用gcd(2048,targets)缩小chunk。一般化不应盲目这样做：chunk过小可能显著增加工作量。它也不是任意64-token目标的原生CP checkpoint kernel。

以下只列常规gate；near-one失败结果全部保留于CSV/JSON。这些快的CP候选**尚未通过统一正确性验收**。

| 模型 | lengths / targets | 全V ms | V2 ms | 全batch CP ms | 混合CP ms | 全batch/长序列chunk |
|---|---|---:|---:|---:|---:|---|
'''
for r in cp:
 if r['gate']!='slow':continue
 t=r['timing'];text+=f"| {r['model']} | {r['lengths']} / {r['targets']} | {t['full_selected']['median_ms']:.4f} | {t['dv_selected']['median_ms']:.4f} | {t.get('cp_boundary',{}).get('median_ms',float('nan')):.4f} | {t.get('mixed_boundary',{}).get('median_ms',float('nan')):.4f} | {r['cp_chunk']}/{r['mixed_cp_chunk']} |\n"
text+='''
## 正确性与内存检查

- 240/240 full/V2组合检查逐位一致，覆盖两种geometry、slow/near-one、1/2/3/4/8序列、零长度、非64对齐尾部、混合保存目标、eager与Graph、同地址改变target后Graph重放及checkpoint续算。
- 分段CP 60项中42项容差通过、18项失败。它是参考实现，不是建议默认开启的后端。
- CP边界/混合边界探针48项中常规gate24项通过，near-one24项中23项未通过状态或续算容差。原完整CP与边界提取包装器的token输出/最终状态在全部24组输入逐位相同，说明包装器没有改变这些原CP输出；不能因此断言新增checkpoint与原dense等价。
- 额外16组零/非零初态、空序列state sentinel、full/V2/CP/mixed检查；Compute Sanitizer memcheck 0 errors。关闭API探测返回码报告，开启stream-ordered检查与padding32；不是完整racecheck/长时soak。
- 额外4项全零目标计划直接走无checkpoint接口，eager/Graph均逐位一致；该分支是主性能测量后的补充，未改变主测量中非空计划的行为。
- 7个非法保存计划被拒绝，包括多目标列表；当前共享协议是每序列0或1目标。没有 silently 丢弃额外目标。消费者若需要多个checkpoint，必须扩展协议，不能强制精简。
- 没有新增独立FP32高精度recurrence验收或真实122B激活/模型质量验证；仅能声称对已测原版non-CP路径无新增数值差异。

### CP prefix对照

为区分原CP数值差异与提取差异，单独从原初态运行到目标位置，对比CP prefix最终状态、fixup边界状态及原dense checkpoint。

| 模型 | gate | target | fixed vs CP-prefix maxabs / close | CP-prefix vs dense maxabs / close | fixed vs dense maxabs / close |
|---|---|---:|---|---|---|
'''
for r in prefix:
 cells=[]
 for k in ['fixed_vs_original_cp_prefix','original_cp_prefix_vs_dense','fixed_vs_dense']:
  v=r[k];cells.append(f"{v['max_abs']:.6f} / {v['close_01']}")
 text+=f"| {r['model']} | {r['gate']} | {r['target']} | "+' | '.join(cells)+' |\n'
text+='''
一个关键例子是122B、gate=0.999、target4096：原CP prefix最终状态对dense在既定容差内，但fixup提取状态对CP prefix及dense均失败。因此新增状态提取本身也有额外数值差异，不能简单用“原CP已有误差”解释。未来若继续，应在最终prefill正确边界写出状态并独立验收，而不是默认把fixup workspace作为可替代的持久checkpoint。

上述对照保留误差来源的区别：不能把所有CP偏差归咎于checkpoint提取，也不能以“原CP本来不同”替新增checkpoint放行。当前统一默认后端应保持non-CP完整V/V2，CP扩展继续作为研究候选。

## 建议的统一结构

```text
服务adapter（各版本审计消费者）
  -> 保存计划：lengths / target positions / output slots
  -> geometry：Hq,Hv,D,dtype,SM数
  -> 选择：full-V selected / V2 selected / 原版回退
  -> 相同输出、最终状态、checkpoint及续算验证

CP边界后端：实验分支；先通过状态数值验收，再参与选择。
```

本轮`unified.py`实现共享plan与参考CP分段接口，`dispatch_geometry.py`把原35B门控的Hv参数化；`selected*.py`、`dvsplit*.py`仍是同一套原kernel，没有分别为122B重写。没有把35B SGLang硬编码hook直接装到122B，也没有强制统一双方实际需要的checkpoint位置。

实施优先级：先共享保存计划/完整V/V2核心及测试；再审计122B真实服务consumer并做接入；最后单独解决CP checkpoint数值和对齐代价。本轮结果不支持为了代码统一而先把两边都改成CP。

## 复现与文件

- `config/contract.json`、`config/commands.json`：冻结条件与实际命令；`config/*_command.json`：主测试PID/启动记录。
- `scripts/01_experiment.py`：240精确项、60 CP项和16性能shape；`03_boundary_probe.py`：CP边界/混合探针。
- `scripts/04_edge_checks.py`：memcheck/空序列/零初态；`05_cp_prefix_check.py`：CP prefix误差归因。
- `scripts/21_report.py`：重建本报告、CSV与汇总；`results/*.json`保存样本、误差、Graph重复顺序，不删失败。
- `evidence/`冻结FlashInfer源码和版本；`logs/`保存完整合成实验日志，无业务payload。
- GPU6进程正常退出后释放；GPU7未使用，其他任务未操作。共享主机其他GPU上的工作仍可能造成CPU等干扰；未锁频。
- 知识库旧记录格式校验仍失败，因此独立保存，不写知识库。没有服务端口、TTFT/TPOT/E2E或请求吞吐数据。
'''
(R/'README.md').write_text(text)
print(json.dumps(summary,indent=2))
