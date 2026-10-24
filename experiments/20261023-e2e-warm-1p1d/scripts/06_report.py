import pathlib,json,statistics,hashlib,collections,yaml
R=pathlib.Path('/root/gdn35/20261023-e2e-warm-1p1d')
s=json.loads((R/'results/summary.json').read_text());rows=json.loads((R/'results/measured_normalized.json').read_text());data=json.loads((R/'evidence/dataset_manifest.json').read_text());resets=[json.loads(l) for l in (R/'results/cache_resets.jsonl').read_text().splitlines()]
assert len(rows)==40 and all(x['status']==200 for x in rows)
assert len(resets)==4 and all(x['phase']=='before_cache_warmup' for x in resets)
assert all(v['status']=='success' for x in resets for c in x['resets'] for v in c['responses'])
assert hashlib.sha256((R/'config/fixed10.json').read_bytes()).hexdigest()==data['sha256']
pairs=[]
for a in rows:
 assert len(a['prefill_batches'])==1
 assert a['usage']['prompt_tokens_details']['cached_tokens']==a['prefill_batches'][0]['cached_tokens']
 assert a['usage']['prompt_tokens']==a['prefill_batches'][0]['new_tokens']+a['prefill_batches'][0]['cached_tokens']
 if a['mode']=='baseline':
  b=next(x for x in rows if x['mode']=='optimized' and x['sample_id']==a['sample_id'] and x['rep']==a['rep'])
  pairs.append(dict(sample_id=a['sample_id'],rep=a['rep'],cache_equal=a['prefill_batches']==b['prefill_batches']))
assert all(x['cache_equal'] for x in pairs)
route={m:dict(sum((collections.Counter(x['gdn_route_delta']) for x in rows if x['mode']==m),collections.Counter())) for m in ['baseline','optimized']}
a=[x for x in rows if x['mode']=='baseline' and x['rep']==1];cached=sum(x['prefill_batches'][0]['cached_tokens'] for x in a);new=sum(x['prefill_batches'][0]['new_tokens'] for x in a)
audit=dict(successful_requests=40,paired_cache_checks=pairs,cached_tokens_per_batch=cached,new_tokens_per_batch=new,token_cache_hit_rate=cached/(cached+new),gdn_route_totals=route,cache_clear_calls_before_warmup=4,cache_clear_calls_during_measurement=0)
(R/'results/cache_audit.json').write_text(json.dumps(audit,indent=2))
lines=['# 35B TP1：1P1D + DFLASH，保留缓存的 GDN A/B','', '- 日期：`2026-10-23`','- 运行结果：`completed`','','## 结论与核心结果','','**并发 1、固定去图片 10 请求、AB/BA 两轮，共 40/40 成功。原版与优化版的平均 TTFT 为 145.77→136.55 ms，降低 6.33%；平均 E2E 为 649.17→662.54 ms，增加 2.06%。TTFT P95 未改善，不能概括为全面端到端提速。**','',f'每批 10 条共 {cached+new:,} 输入 tokens，其中 {cached:,} 命中缓存，{new:,} 重新计算，命中率 **{cached/(cached+new):.2%}**。20 对 A/B 的缓存命中量逐条一致。','', '下表为每组 20 次正式请求的分布，NumPy linear percentile；TTFT/E2E 单位 ms，TPOT 单位 ms/token。','','| 指标 | 实现 | mean | P50 | P75 | P90 | P95 | P99 |','|---|---|---:|---:|---:|---:|---:|---:|']
for k in ['ttft_ms','tpot_ms','e2e_ms']:
 for mode in ['baseline','optimized']:
  v=s[mode]['pooled_20'][k];lines.append('| '+k+' | '+mode+' | '+' | '.join(f'{v[p]:.3f}' for p in ['mean','p50','p75','p90','p95','p99'])+' |')
lines+=['','每批均值：','','| 轮次/顺序 | 实现 | TTFT ms | TPOT ms/token | E2E ms |','|---|---|---:|---:|---:|']
for rep,mode in [(1,'baseline'),(1,'optimized'),(2,'optimized'),(2,'baseline')]:
 rr=[x for x in rows if x['rep']==rep and x['mode']==mode]
 lines.append(f'| {rep} | {mode} | '+' | '.join(f'{statistics.mean(x[k] for x in rr):.3f}' for k in ['ttft_ms','tpot_ms','e2e_ms'])+' |')
lines+=['','与上一轮冷缓存的原版均值相比，保留缓存使 TTFT 从 1256.61 ms 降至 145.77 ms，约降低 88.4%。这部分收益来自缓存，不能记为 GDN kernel 改动的收益。','','## 复现','','- 仍为物理 GPU 6 prefill、GPU 7 decode+DFLASH；并发 **1**，服务 max-running-requests=16 只是容量上限。','- 固定样本字节内容与冷缓存测试完全相同，图片内容块已删除；temperature=0、seed=42、max_tokens=512，自然停止。','- 服务启动后先用 A/B 各两条请求完成 JIT 预热，不计时。','- 每个正式批次之前，只在预热开始前清理一次 P/D cache；然后统一使用原版、相同顺序预热完整 10 条，再切换到待测 A 或 B。','- **正式批次内保留所有缓存，逐请求清缓存已移除。** 执行 A(10)→B(10)→B(10)→A(10)，正式请求共 40，缓存预热请求共 40，JIT 预热请求共 4。','- A 调用原 FlashInfer auto；B 只对合格单序列、无 checkpoint 的 GDN 调用使用 Dv 二分。P/D 其他实现和 CUDA Graph 保持一致。','- TTFT 为客户端 POST 至首个非空内容/推理/tool fragment；E2E 至 SSE DONE；TPOT=(E2E−TTFT)/(completion_tokens−1)。','','```bash','cd /root/gdn35/20261023-e2e-warm-1p1d','printf "baseline\\n" > config/mode.txt','/opt/sglang/bin/python scripts/03_launch.py etcd','/opt/sglang/bin/python scripts/03_launch.py frontend','/opt/sglang/bin/python scripts/03_launch.py p','/opt/sglang/bin/python scripts/03_launch.py d','/opt/sglang/bin/python scripts/04_benchmark.py','/opt/sglang/bin/python scripts/05_summarize.py','/opt/sglang/bin/python scripts/06_report.py','/opt/sglang/bin/python scripts/07_stop.py','```','','复测应新建结果目录并更新脚本路径、namespace；脚本拒绝覆盖已有正式结果/PID。models/、bin/ 复用前一轮本机磁盘副本。','','## 运行配置','','- 模型：/prefill_model FP8、/decode_model 4-bit compressed-tensors、/spec_model DFLASH；模型本地副本与上轮相同。','- TP1、1P1D、DFLASH=8、context=262144、mem-fraction-static=0.80、page-size=16、max-running-requests=16。','- Dynamo 1.5.0、SGLang 0.5.20、FlashInfer 0.6.18；源码状态沿用前一轮，未升级包。镜像 digest 未暴露。','- GDN 实际自动选择：prefill=FlashInfer，decode/verify=Triton。','- namespace=gdn35-warm-1023；Frontend=18000，P/D health=18081/18082，etcd=12379。',f"- 固定数据 SHA256：`{data['sha256']}`。",'','实际缓存分布（两轮、两组逐条一致）：','','| 样本 | 输入 tokens | 缓存命中 | 新计算 | B 的 GDN 路径 |','|---|---:|---:|---:|---|']
for x in a:
 b=next(y for y in rows if y['sample_id']==x['sample_id'] and y['rep']==1 and y['mode']=='optimized');v=x['prefill_batches'][0]
 lines.append(f"| {x['sample_id']} | {x['usage']['prompt_tokens']} | {v['cached_tokens']} | {v['new_tokens']} | "+('Dv 二分' if b['gdn_route_delta'].get('optimized:True',0) else 'checkpoint 回退')+' |')
lines+=['','## 限制与证据','','- 这是**同一批请求预热后重复回放**的 warm-cache 结果，97.46% 命中率不是生产多轮请求缓存命中率的估计。','- B 的 20 次请求合计 480 次 GDN 层调用命中优化、120 次因 checkpoint 回退；对应每批 8 条短尾请求优化、2 条较长尾请求回退。','- TTFT 均值两轮分别降低约 11.25%、0.94%，改善幅度不稳定；P95/P99 未改善。只有 10 个不同请求，小样本高分位数不代表生产尾延迟。','- 原版平均输出 279.4 tokens，优化版 283.6 tokens。20 对 A/B 中仅 5 对输出 token 数相同、1 对规范化输出相同；因此 E2E 不是等输出工作量比较，不能用它独立判定 kernel 性能。','- 本轮未新增真实激活逐位比较；kernel 与此前 60 次真实激活检查通过的实现相同。端到端模型输出仍未实现确定性，不宣称模型质量或逐位一致性验收。','- 同主机 GPU 0–3 有其他任务，未触碰；未锁频。','- 最初 benchmark 记录器的语法错误发生在执行请求前，已修复并保留日志；正式 40 请求无失败。','- infer-kb ensure 仍被历史结果格式错误阻塞，按技能要求停止知识库相关写入，本实验独立保存。','','| 证据 | 路径 |','|---|---|','| 完整统计/配对输出 | [results/summary.json](results/summary.json) |','| 每请求指标、缓存与 GDN 路径 | [results/requests.jsonl](results/requests.jsonl) |','| 缓存可比性校验 | [results/cache_audit.json](results/cache_audit.json) |','| 缓存重置发生位置 | [results/cache_resets.jsonl](results/cache_resets.jsonl) |','| 固定去图样本 | [config/fixed10.json](config/fixed10.json) |','| 服务命令 | config/p_default.json、config/d_default.json、config/service-1p1d.yaml |','| 原始 SSE | results/*.sse.json |','| 服务与测试日志 | logs/ |','| metrics | evidence/prefill_metrics_after.txt、evidence/decode_metrics_after.txt |','']
(R/'README.md').write_text('\n'.join(lines))
p=json.loads((R/'config/p_default.json').read_text());d=json.loads((R/'config/d_default.json').read_text());cut=p['command'].index('--model-path')
cfg={'service':{'host':'0.0.0.0','port':19000},'worker':{'env':{k:v for k,v in p['env'].items() if k not in ['CUDA_VISIBLE_DEVICES','DYN_SYSTEM_PORT','GDN35_PATCH']},'command':p['command'][:3],'args':p['command'][3:cut],'workers':[{'name':'p0','env':{'CUDA_VISIBLE_DEVICES':'6','DYN_SYSTEM_PORT':'18081','GDN35_PATCH':'1'},'args':p['command'][cut:]},{'name':'d0','env':{'CUDA_VISIBLE_DEVICES':'7','DYN_SYSTEM_PORT':'18082'},'args':d['command'][cut:]}]}}
cfg['service']['policy']=dict(interval=10,probe_timeout=5,failure_threshold=3,startup_timeout=3600,all_down_timeout=120,stop_timeout=45,restart_backoff=5)
for w,port in zip(cfg['worker']['workers'],[18081,18082]):w['health_url']=f'http://127.0.0.1:{port}/health'
(R/'config/service-1p1d.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['scripts','config'] for p in (R/folder).glob('*') if p.is_file()}
(R/'evidence/artifact_sha256.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(audit))
