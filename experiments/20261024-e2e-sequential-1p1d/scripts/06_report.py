from pathlib import Path
import json,collections,statistics,hashlib,shlex
R=Path('/root/gdn35/20261024-e2e-sequential-1p1d')
rows=json.loads((R/'results/measured_normalized.json').read_text())
s=json.loads((R/'results/summary.json').read_text())
resets=[json.loads(l) for l in (R/'results/cache_resets.jsonl').read_text().splitlines()]
allrows=[json.loads(l) for l in (R/'results/requests.jsonl').read_text().splitlines()]
assert len(rows)==40 and len(allrows)==44 and len(resets)==4
assert all(x['phase']=='before_measured_batch' for x in resets)
assert all(x['phase'] in ['measured','jit_warmup'] for x in allrows)
assert [(x['rep'],x['target_mode']) for x in resets]==[(1,'baseline'),(1,'optimized'),(2,'optimized'),(2,'baseline')]
assert [(r['rep'],r['mode'],r['sample_id']) for r in rows]==[(rep,mode,f'sample_{i:02d}') for rep,mode in [(1,'baseline'),(1,'optimized'),(2,'optimized'),(2,'baseline')] for i in range(10)]
cache=[];route=collections.defaultdict(collections.Counter)
for r in rows:
 assert r['status']==200 and r['prefill_batches']
 c=r['usage'].get('prompt_tokens_details',{}).get('cached_tokens')
 # Use backend logs for the complete per-request prefill work and usage for prefix-cache hits.
 new=sum(x['new_tokens'] for x in r['prefill_batches'])
 if c is None:c=r['usage']['prompt_tokens']-new
 assert c==r['usage']['prompt_tokens']-new and c>=0,(r['tag'],c,new)
 cache.append(dict(rep=r['rep'],mode=r['mode'],sample_id=r['sample_id'],prompt=r['usage']['prompt_tokens'],cached=c,new=new,batches=r['prefill_batches']))
 route[r['mode']].update(r['gdn_route_delta'])
pairs=[]
for rep in [1,2]:
 for i in range(10):
  rr=[r for r in cache if r['rep']==rep and r['sample_id']==f'sample_{i:02d}']
  pairs.append(dict(rep=rep,sample_id=f'sample_{i:02d}',equal=rr[0]['cached']==rr[1]['cached'] and rr[0]['new']==rr[1]['new']))
audit=dict(successful_requests=len(rows),total_requests=len(allrows),reset_calls=4,resets_during_batch=0,cache_pairs=pairs,requests=cache,route_totals={k:dict(v) for k,v in route.items()})
(R/'results/cache_audit.json').write_text(json.dumps(audit,indent=2))
# Render the exact argv and explicit environment recorded by the launcher; no service wrapper involved.
commands=['#!/usr/bin/env bash','# Recorded startup commands; review ports and existing processes before running.','set -euo pipefail',f'mkdir -p {shlex.quote(str(R/"topology"))}',f'hostname > {shlex.quote(str(R/"topology/host"))}']
for role in ['etcd','frontend','p','d']:
 c=json.loads((R/f'config/{role}_default.json').read_text())
 env=dict(c['env'],PYTHONUNBUFFERED='1')
 commands.append('\n# '+role+'\n'+'env '+' \\\n  '.join(shlex.quote(k+'='+v) for k,v in env.items())+' \\\n  '+shlex.join(c['command'])+' > '+shlex.quote(str(R/f'logs/{role}_relaunch.log'))+' 2>&1 &')
(R/'config/actual-start-commands.sh').write_text('\n'.join(commands)+'\n')
a=s['baseline']['pooled_20'];b=s['optimized']['pooled_20']
lines=['# 35B TP1：每批清缓存一次、逐条执行的 1P1D GDN A/B','', '- 日期：`2026-10-24`','- 运行结果：`completed`','','## 结论与核心结果','',f'40/40 正式请求成功，GPU 6 prefill / GPU 7 decode+DFLASH，并发 1，AB、BA 两轮。平均 TTFT {a["ttft_ms"]["mean"]:.3f}→{b["ttft_ms"]["mean"]:.3f} ms，变化 {100*(b["ttft_ms"]["mean"]/a["ttft_ms"]["mean"]-1):+.2f}%；本轮未观察到平均 TTFT 提升。','', '每组 20 次请求的分布；TTFT/E2E 单位 ms，TPOT 单位 ms/token。','', '| 指标 | 版本 | mean | P50 | P75 | P90 | P95 | P99 |','|---|---|---:|---:|---:|---:|---:|---:|']
for k in ['ttft_ms','tpot_ms','e2e_ms']:
 for mode in ['baseline','optimized']:
  values=s[mode]['pooled_20'][k]
  lines.append('| '+k+' | '+mode+' | '+' | '.join(f'{values[p]:.3f}' for p in ['mean','p50','p75','p90','p95','p99'])+' |')
lines+=['','批次均值：','','| 顺序 | TTFT ms | TPOT ms/token | E2E ms | 输入 tokens | 缓存命中 tokens |','|---|---:|---:|---:|---:|---:|']
for rep,mode in [(1,'baseline'),(1,'optimized'),(2,'optimized'),(2,'baseline')]:
 rr=[r for r in rows if r['rep']==rep and r['mode']==mode];cc=[r for r in cache if r['rep']==rep and r['mode']==mode]
 lines.append('| '+f'{mode} {rep}'+' | '+' | '.join(f'{statistics.mean(r[k] for r in rr):.3f}' for k in ['ttft_ms','tpot_ms','e2e_ms'])+f' | {sum(r["prompt"] for r in cc)} | {sum(r["cached"] or 0 for r in cc)} |')
lines+=['','## 复现','','- 每个正式批次前清理一次 P、D cache，然后固定 10 条按顺序、并发 1 执行。请求之间保留 cache。','- 没有在每批计时前回放测试请求来填缓存。启动 JIT 预热 4 条，随后先清缓存再开始 A1，预热不计时。','- 顺序 A1(10)→B1(10)→B2(10)→A2(10)，四次批次前重置均有服务端成功响应。','- A 为原 FlashInfer auto；B 为相同 Dv 二分实现，仅修改 prefill worker 的合格 GDN 调用。','- TTFT 从客户端 POST 到首个非空 content/reasoning/tool fragment；E2E 到 SSE DONE；TPOT=(E2E−TTFT)/(completion_tokens−1)。','- 客户端在服务主机通过 loopback HTTP 访问 Dynamo frontend，包含服务链路，不包含远程客户端网络延迟。','- [完整环境变量和启动命令](config/actual-start-commands.sh)，直接进程启动，无 service manager wrapper。','- [测试脚本](scripts/04_benchmark.py)，[汇总脚本](scripts/05_summarize.py)。重新运行应新建目录并更新根路径/namespace，不能覆盖此结果。','','## 运行配置','','- 固定去图片数据 SHA256：`'+s['dataset_sha256']+'`；与前两轮相同。','- TP1、1P1D、DFLASH 8、context-length 262144、mem-fraction-static 0.80、page-size 16、max-running-requests 16（容量上限，实际并发 1）。','- /prefill_model FP8、/decode_model 4-bit、/spec_model 的相同本地副本；models/ 和 bin/ 复用 20261023 原始实验副本。','- temperature=0、seed=42、max_tokens=512、自然停止；没有图片输入。','- 使用 /opt/sglang/bin/python，继承同一服务软件环境；没有安装或修改依赖。','- 没有强制 Triton GDN prefill；prefill 自动 FlashInfer，decode/verify 原配置。','','每条的缓存情况（第 1 轮，A/B）：','','| 样本 | 输入 | A 缓存 | B 缓存 | A 新计算 | B 新计算 |','|---|---:|---:|---:|---:|---:|']
for i in range(10):
 aa=next(x for x in cache if x['rep']==1 and x['mode']=='baseline' and x['sample_id']==f'sample_{i:02d}');bb=next(x for x in cache if x['rep']==1 and x['mode']=='optimized' and x['sample_id']==f'sample_{i:02d}')
 lines.append(f'| sample_{i:02d} | {aa["prompt"]} | {aa["cached"]} | {bb["cached"]} | {aa["new"]} | {bb["new"]} |')
lines+=['','## 限制与证据','',f'- 20 对 A/B 中，缓存命中与新计算量相同：{sum(p["equal"] for p in pairs)}/20。',f'- GDN 路径计数：`{dict(route)}`。',f'- 输出 tokens 平均值 A={statistics.mean(r["usage"]["completion_tokens"] for r in rows if r["mode"]=="baseline"):.2f}，B={statistics.mean(r["usage"]["completion_tokens"] for r in rows if r["mode"]=="optimized"):.2f}。',f'- 20 对 A/B 中输出长度一致 {sum(p["output_tokens_equal"] for p in s["pairs"])} 对、规范化输出一致 {sum(p["output_equal"] for p in s["pairs"])} 对；E2E 不是等输出量比较。','- 只有 10 个不同请求、每个两次，高分位数不代表生产尾延迟。', '- sample_03 在 B2 的 TTFT 为 748.5 ms，其余三次为 263.7–282.9 ms；该请求全部回退原 GDN，说明本轮存在不能直接归因于优化 kernel 的延迟波动。保留全部测量值，未剔除异常点。','- 同主机 GPU 0–3 有其他任务，未触碰；未锁频。','- 沿用已测试 kernel；本轮未增加真实激活正确性验证，不宣称端到端输出确定性或模型质量验收。','- infer-kb 既有结果格式校验失败尚未修复，依照技能限制未写知识库，此处独立保存。','- [逐请求结果](results/requests.jsonl)、[完整统计](results/summary.json)、[缓存审计](results/cache_audit.json)、[重置记录](results/cache_resets.jsonl)、logs/ 下服务日志。']
(R/'README.md').write_text('\n'.join(lines)+'\n')
manifest={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for d in ['scripts','config'] for f in (R/d).glob('*') if f.is_file()}
(R/'evidence/artifact_sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'measured':len(rows),'cache_pairs_equal':sum(p['equal'] for p in pairs),'route_totals':audit['route_totals']}))
