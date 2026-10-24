"""Report every concurrent request, without outlier removal or cache equivalence assumptions."""
import json,pathlib,statistics,csv,hashlib,collections
R=pathlib.Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(x) for x in (R/p).read_text().splitlines()]
def pct(xs,p):
 a=sorted(xs);i=(len(a)-1)*p/100;j=int(i);return a[j]+(a[min(j+1,len(a)-1)]-a[j])*(i-j)
def stats(xs):return dict(mean=statistics.mean(xs),**{f'p{p}':pct(xs,p) for p in [50,75,90,95,99]})
def text_hash(row):
 events=json.loads((R/'results'/f"{row['tag']}.sse.json").read_text());body={'content':'','reasoning':'','tools':{}}
 for e in events:
  for choice in e['data'].get('choices',[]):
   d=choice.get('delta',{});body['content']+=d.get('content') or '';body['reasoning']+=d.get('reasoning_content') or d.get('reasoning') or ''
   for t in d.get('tool_calls',[]):
    z=body['tools'].setdefault(str(t.get('index',0)),{'name':'','arguments':''});f=t.get('function',{});z['name']+=f.get('name') or '';z['arguments']+=f.get('arguments') or ''
 return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
def main():
 allrows=rows('results/requests.jsonl');formal=[r for r in allrows if r['phase']=='formal'];batches=rows('results/batches.jsonl');checks=rows('evidence/real_checks.jsonl');routes=rows('evidence/concurrent_routes.jsonl')
 assert len(formal)==80 and all(r['success'] for r in formal)
 assert len(allrows)==160 and all(r['success'] for r in allrows)
 result={'formal_requests':80,'total_requests_including_shadow_and_warm':len(allrows),'successes':sum(x['success'] for x in allrows),'shadow_checks':len(checks),'shadow_exact':all(x['valid'] and all(v==0 for v in x['errors'].values()) for x in checks),'arms':{},'cache_pairs':[],'paired_ttft':{}}
 for c in [2,4]:
  result['arms'][str(c)]={}
  for mode in ['baseline','sparse']:
   rr=[r for r in formal if r['concurrency']==c and r['mode']==mode];assert len(rr)==20
   bb=[x for x in batches if x['phase']=='formal' and x['concurrency']==c and x['mode']==mode]
   result['arms'][str(c)][mode]={m:stats([r[m] for r in rr]) for m in ['ttft_ms','tpot_ms','e2e_ms']}
   result['arms'][str(c)][mode].update(mean_output_tokens=statistics.mean(r['usage']['completion_tokens'] for r in rr),requests_per_s=20/sum(x['elapsed_s'] for x in bb),total_new_tokens=sum(x['total_new_tokens'] for x in bb),total_cached_tokens=sum(x['total_cached_tokens'] for x in bb))
  pairs=[]
  for rep in [1,2]:
   aa,bb=[next(x for x in batches if x['phase']=='formal' and x['concurrency']==c and x['mode']==m and x['rep']==rep) for m in ['baseline','sparse']]
   result['cache_pairs'].append(dict(concurrency=c,rep=rep,a_new=aa['total_new_tokens'],b_new=bb['total_new_tokens'],a_cached=aa['total_cached_tokens'],b_cached=bb['total_cached_tokens'],total_equal=aa['total_new_tokens']==bb['total_new_tokens'] and aa['total_cached_tokens']==bb['total_cached_tokens'],batch_layout_equal=aa['prefill_batches']==bb['prefill_batches']))
   for i in range(10):
    a,b=[next(x for x in formal if x['concurrency']==c and x['rep']==rep and x['mode']==m and x['sample_id']==f'sample_{i:02d}') for m in ['baseline','sparse']]
    assert a['usage']['prompt_tokens']==b['usage']['prompt_tokens']
    ac=(a['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0;bc=(b['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0
    pairs.append(dict(sample_id=a['sample_id'],rep=rep,a_ttft_ms=a['ttft_ms'],b_ttft_ms=b['ttft_ms'],saved_ms=a['ttft_ms']-b['ttft_ms'],text_equal=text_hash(a)==text_hash(b),a_cached_tokens=ac,b_cached_tokens=bc,api_cache_equal=ac==bc))
  result['paired_ttft'][str(c)]={'pairs':pairs,'saved_ms':stats([p['saved_ms'] for p in pairs]),'wins':sum(p['saved_ms']>0 for p in pairs),'text_equal':sum(p['text_equal'] for p in pairs),'api_cache_equal_pairs':sum(p['api_cache_equal'] for p in pairs)}
 (R/'results/summary.json').write_text(json.dumps(result,indent=2))
 lines=['# 35B TP1 GDN：并发2/4端到端验证','','运行结果：completed。此状态表示本轮执行完成，不表示长期稳定性或模型质量认证。','',
 'h20-2 GPU6=P、GPU7=D+DFLASH，TP1，保留BCG、NIXL、DFLASH8。SGLang0.5.20、FlashInfer0.6.18、Dynamo1.5.0。模型为此前同一组本地副本：P FP8、D 4-bit，draft/spec_model。', '',
 '固定去图10条，每个并发分别AB、BA两轮，共80次正式请求。每个P部署先用相同10条按并发2和4预热，清P/D缓存后测量；批内保留前缀缓存。闭环并发，完成一条补下一条，最多2/4条在途，客户端测时从真正提交HTTP开始。输入顺序固定，但完成顺序由运行时决定。max-running-requests=16是服务容量，不是本次负载并发。', '',
 'P部署顺序A1、B、A2，B上连续完成B1/B2；D/frontend全程保持。原版A不加载hook，正式B使用上一轮固定实现；shadow单独运行并记录路径，不进入正式计时。完整参数见config/*.json和scripts/03_launch.py。', '',
 '## 端到端各分位数','', '| 并发 | 指标 | 实现 | mean | P50 | P75 | P90 | P95 | P99 |','|---:|---|---|---:|---:|---:|---:|---:|---:|']
 for c in [2,4]:
  for m in ['ttft_ms','tpot_ms','e2e_ms']:
   for mode in ['baseline','sparse']:
    x=result['arms'][str(c)][mode][m];lines.append(f'| {c} | {m} | {mode} | '+' | '.join(f'{x[k]:.3f}' for k in ['mean','p50','p75','p90','p95','p99'])+' |')
 lines+=['','TTFT/E2E单位ms，TPOT单位ms/token。TTFT为POST到首个非空content/reasoning/tool片段，TPOT=(E2E−TTFT)/(输出tokens−1)。分位数线性插值，每臂每并发20条，不代表生产尾部。自然结束，max_tokens512，temperature0、seed42。','',
 '| 并发 | 指标 | A | B | B相对变化 |','|---:|---|---:|---:|---:|']
 for c in [2,4]:
  a,b=[result['arms'][str(c)][m] for m in ['baseline','sparse']]
  for m in ['ttft_ms','tpot_ms','e2e_ms','requests_per_s','mean_output_tokens']:
   av=a[m]['mean'] if isinstance(a[m],dict) else a[m];bv=b[m]['mean'] if isinstance(b[m],dict) else b[m]
   lines.append(f'| {c} | {m} | {av:.3f} | {bv:.3f} | {(bv/av-1)*100:+.2f}% |')
 lines+=['','## 每批结果和缓存审计','','| 并发 | 轮 | 实现 | 成功 | req/s | TTFT mean ms | TPOT mean ms/token | E2E mean ms | 新tokens | 缓存tokens |','|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
 for x in batches:
  if x['phase']!='formal':continue
  lines.append(f"| {x['concurrency']} | {x['rep']} | {x['mode']} | {x['success']}/10 | {x['requests_per_s']:.3f} | {x['metrics']['ttft_ms']['mean']:.3f} | {x['metrics']['tpot_ms']['mean']:.3f} | {x['metrics']['e2e_ms']['mean']:.3f} | {x['total_new_tokens']} | {x['total_cached_tokens']} |")
 lines+=['','缓存从P的Prefill batch日志逐批求和，不能作为每请求精确缓存归因。日志记录的chunk缓存可能随批处理变化，原始batch序列一并保存。缓存总量/批布局相同也不足以证明所有调度一致。','']
 for x in result['cache_pairs']:lines.append(f"- C{x['concurrency']} 第{x['rep']}轮：缓存/新token总量相同={x['total_equal']}；batch布局完全相同={x['batch_layout_equal']}。")
 lines+=['','## 如何解读变化','', '均值、P50、P90、P99分别描述延迟分布的不同位置，不要求下降比例相同。前轮单并发中sample01/02贡献约70.3%的总配对TTFT节省，而最慢请求几乎不变，因此P90下降大、P99下降小。20条样本的P99接近最大值，容易被单条波动左右。', '', '并发测试还引入请求排队、合批、完成顺序与前缀可用时机变化。闭环端到端结果描述这套实际运行方式；缓存/批布局不一致时，无法把差值隔离归因于GDN算子。即使总缓存相同，单请求分配和GPU调度也可能不同。']
 for c in [2,4]:
  changes=[]
  for rep in [1,2]:
   a,b=[next(x for x in batches if x['phase']=='formal' and x['concurrency']==c and x['rep']==rep and x['mode']==m) for m in ['baseline','sparse']]
   changes.append((b['metrics']['ttft_ms']['mean']/a['metrics']['ttft_ms']['mean']-1)*100)
  lines.append(f"- C{c}：两轮平均TTFT相对变化分别{changes[0]:+.2f}%、{changes[1]:+.2f}%。仅两轮，不能作为长期稳定收益的证据。")
 lines+=['','## 正确性、路径与稳定性','',f"- 80/80正式请求成功；连同预热和shadow，共{len(allrows)}/{len(allrows)}请求成功。",f"- 本轮在线shadow：{len(checks)}次优化调用，输出、最终状态、选定checkpoint均逐位一致={result['shadow_exact']}。核对基准为相同实时输入上的原版。", '- 多序列或不支持shape保留原路径。shadow统计不能冒充无审计正式批次的精确路径分布。']
 for x in batches:
  if x['phase']=='shadow':lines.append(f"- C{x['concurrency']}诊断10条：{x['route_count']}次GDN调用，{x['selected_calls']}次选中优化，{x['multi_sequence_calls']}次多序列调用。")
 lines+=['- 进一步分解诊断checkpoint调用：C2为300次，其中单序列210次优化、双序列90次回退；C4为240次，其中单序列30次优化、双序列180次和三序列30次回退。checkpoint调用覆盖率分别70%和12.5%。另有无checkpoint调用保留原版。详见results/shadow_route_breakdown.json；这不是按耗时加权的覆盖率。']
 for c in [2,4]:
  x=result['paired_ttft'][str(c)];lines.append(f"- C{c}：TTFT改善{x['wins']}/20对，配对节省中位数{x['saved_ms']['p50']:.3f}ms；归一化输出文本一致{x['text_equal']}/20；API返回cached_tokens相同{x['api_cache_equal_pairs']}/20（未返回此字段按0记录）。")
 lines+=['- 前轮合成、真实张量、恢复续算、fallback、memcheck证据见../20261024-gdn-checkpoint-opt/README.md。独立FP32参考有原版同样的阈值失败；旧全路径auto-CP替换候选已拒绝。本轮没有重新替换该路径。', '- 本轮是短时并发冒烟和ABBA性能验证，不是长时间soak、故障恢复或所有batch/shape的稳定性认证。输出长度和文本可能不同，E2E不是严格等输出工作量比较，不能把变化全部归因于P GDN。', '',
 '## 文件与复现','', '- scripts/20_concurrent.py：完整负载/预热/清缓存/ABBA控制器；03_launch.py：启动；07_stop.py：只清理本实验PID。', '- results/requests.jsonl、batches.jsonl、cache_resets.jsonl及*.sse.json：原始请求与流式事件。', '- results/summary.json、metrics.csv：汇总；evidence/concurrent_routes.jsonl、real_checks.jsonl：并发路径及在线正确性。', '- config/fixed10.json沿用前轮同一SHA256；config/*.json记录实际命令、环境和PID；logs/保存每批P日志片段。', '- 未修改安装目录，结束停止本轮P/D/frontend/etcd，GPU释放状态见evidence/gpu_final.csv。', '- 同机GPU0–3有其他任务，未操作；不能排除共享CPU/内存带宽干扰。知识库旧记录格式校验失败，本轮独立保存，未写知识库。','']
 (R/'README.md').write_text('\n'.join(lines))
 with (R/'results/metrics.csv').open('w') as h:
  w=csv.writer(h);w.writerow(['concurrency','mode','metric','mean','p50','p75','p90','p95','p99'])
  for c in [2,4]:
   for mode in ['baseline','sparse']:
    for m in ['ttft_ms','tpot_ms','e2e_ms']:
     x=result['arms'][str(c)][mode][m];w.writerow([c,mode,m]+[x[k] for k in ['mean','p50','p75','p90','p95','p99']])
 print(json.dumps({k:v for k,v in result.items() if k!='paired_ttft'},indent=2))
if __name__=='__main__':main()
