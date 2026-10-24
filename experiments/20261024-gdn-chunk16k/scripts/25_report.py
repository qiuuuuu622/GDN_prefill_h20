"""Audit chunk-only ABBA and report all outcomes without excluding outliers."""
import json,pathlib,hashlib,statistics,csv,re
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(f):return [json.loads(x) for x in (R/f).read_text().splitlines()]
def pct(a,p):
 a=sorted(a);i=(len(a)-1)*p/100;j=int(i);return a[j]+(a[min(j+1,len(a)-1)]-a[j])*(i-j)
def stats(a):return dict(mean=statistics.mean(a),**{f'p{p}':pct(a,p) for p in [50,75,90,95,99]})
def load(f):return json.loads((R/f).read_text())
def main():
 rr=rows('results/requests.jsonl');bb=rows('results/batches.jsonl');formal=[x for x in rr if x['phase']=='formal'];checks=rows('evidence/real_checks.jsonl')
 assert len(rr)==240 and len(formal)==120 and all(x['success'] for x in rr)
 assert all(x['valid'] and all(v==0 for v in x['errors'].values()) for x in checks)
 assert not (R/'results/failure.txt').exists()
 commands=[load(f'config/p_{v}.json') for v in ['chunk8k_a1','chunk16k','chunk8k_a2']]
 a,b,c=commands;assert a['command']==c['command'] and a['env']==b['env']==c['env']
 i=a['command'].index('--chunked-prefill-size')+1;assert [a['command'][i],b['command'][i]]==['8192','16384']
 bc=b['command'].copy();bc[i]='8192';assert bc==a['command']
 assert a['env']['GDN35_VARIANT']=='multi' and a['env']['GDN35_PATCH']=='1'
 assert sha(R/'config/fixed10.json')=='d5e8c10e433ea5952eaba44346489dd9161637bd1ed94060f45347f32717b30e'
 hashes=load('evidence/runtime_hashes_at_launch.json')
 for n,h in hashes.items():assert sha(R/'scripts'/n)==h,n
 for v in ['chunk8k_a1','chunk16k','chunk8k_a2']:assert 'GDN35_MULTI_CHECKPOINT_PATCH_INSTALLED' in (R/f'logs/p_{v}.log').read_text()
 test=load('results/multi_checks.json');assert len(test)==52 and all(x['valid'] for x in test)
 mem=load('results/multi_memcheck.json');assert len(mem)==10 and all(x['valid'] for x in mem)
 assert 'ERROR SUMMARY: 0 errors' in (R/'logs/multi_memcheck.log').read_text()
 out=dict(formal_requests=len(formal),all_requests=len(rr),shadow_checks=len(checks),shadow_exact=True,arms={},rounds=[],cache_pairs=[])
 for cc in [2,4,8]:
  batch=[x for x in bb if x['phase']=='formal' and x['concurrency']==cc]
  assert [(x['mode'],x['rep']) for x in batch]==[('chunk8k',1),('chunk16k',1),('chunk16k',2),('chunk8k',2)]
  assert all(x['success']==10 and x['observed_client_max_inflight']==cc for x in batch)
  out['arms'][str(cc)]={}
  for mode in ['chunk8k','chunk16k']:
   req=[x for x in formal if x['concurrency']==cc and x['mode']==mode];bat=[x for x in batch if x['mode']==mode]
   out['arms'][str(cc)][mode]={m:stats([x[m] for x in req]) for m in ['ttft_ms','tpot_ms','e2e_ms']}
   out['arms'][str(cc)][mode].update(requests_per_s=20/sum(x['elapsed_s'] for x in bat),new_tokens=sum(x['total_new_tokens'] for x in bat),cached_tokens=sum(x['total_cached_tokens'] for x in bat),prefill_batches=sum(len(x['prefill_batches']) for x in bat),mean_output_tokens=statistics.mean(x['usage']['completion_tokens'] for x in req))
  for rep in [1,2]:
   aa,bb1=[next(x for x in batch if x['rep']==rep and x['mode']==m) for m in ['chunk8k','chunk16k']]
   out['rounds'].append(dict(concurrency=cc,rep=rep,ttft_change_pct=100*(bb1['metrics']['ttft_ms']['mean']/aa['metrics']['ttft_ms']['mean']-1),p95_change_pct=100*(bb1['metrics']['ttft_ms']['p95']/aa['metrics']['ttft_ms']['p95']-1),throughput_change_pct=100*(bb1['requests_per_s']/aa['requests_per_s']-1)))
   for sample in [f'sample_{i:02d}' for i in range(10)]:
    aa,bb1=[next(x for x in formal if x['concurrency']==cc and x['rep']==rep and x['mode']==m and x['sample_id']==sample) for m in ['chunk8k','chunk16k']]
    assert aa['usage']['prompt_tokens']==bb1['usage']['prompt_tokens']
    ac,bc=[(x['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0 for x in [aa,bb1]]
    out['cache_pairs'].append(dict(concurrency=cc,rep=rep,sample_id=sample,a_cached=ac,b_cached=bc,equal=ac==bc,ttft_saved_ms=aa['ttft_ms']-bb1['ttft_ms']))
 out['same_arm_repeatability']=[]
 for cc in [2,4,8]:
  for mode in ['chunk8k','chunk16k']:
   aa,bb2=[next(x for x in bb if x['phase']=='formal' and x['concurrency']==cc and x['mode']==mode and x['rep']==rep) for rep in [1,2]]
   out['same_arm_repeatability'].append(dict(concurrency=cc,mode=mode,new_tokens_equal=aa['total_new_tokens']==bb2['total_new_tokens'],logged_batch_layout_equal=aa['prefill_batches']==bb2['prefill_batches'],r1_ttft_mean_ms=aa['metrics']['ttft_ms']['mean'],r2_ttft_mean_ms=bb2['metrics']['ttft_ms']['mean']))
 out['formal_cpu_throttled_periods']=sum(x.get('cpu_quota_delta',{}).get('nr_throttled',0) for x in bb if x['phase']=='formal')
 out['bcg_logged_batches']={}
 for mode in ['chunk8k','chunk16k']:
  values=[]
  for logfile in (R/'logs').glob(f'formal_*_{mode}.p_excerpt.log'):
   values.extend(re.findall(r'cuda graph: (True|False)',logfile.read_text()))
  out['bcg_logged_batches'][mode]={'true':values.count('True'),'false':values.count('False')}
 for variant in ['chunk8k_a1','chunk16k','chunk8k_a2']:
  log=(R/f'logs/p_{variant}.log').read_text()
  assert 'Capture target prefill CUDA graph begin. backend=breakable' in log
 out['universal_default_screen_passed']=all(x['ttft_change_pct']<=0 and x['throughput_change_pct']>=0 and x['p95_change_pct']<=3 for x in out['rounds'])
 out['decision_rule']=load('config/decision_rule.json')
 (R/'results/summary.json').write_text(json.dumps(out,indent=2))
 lines=['# 35B TP1：新版 GDN，prefill chunk 8192 vs 16384','','运行结果：completed。结论：本次16K候选未通过，保持8K；三个并发档位的两轮平均TTFT均退化。该结论针对本次固定10条负载，不外推为所有业务的最优chunk。','','GPU6=P、GPU7=D+DFLASH8，TP1，保留BCG，NIXL。两臂相同多序列GDN；P唯一参数差异chunked-prefill-size，D/frontend全过程保持。每并发固定去图10条，AB、BA两轮；每个P部署先预热C2/4/8各10条，每个批次开始前清P/D缓存，批内保留。240/240请求成功，包括120正式、90预热、30 shadow。自然结束max_tokens512，不固定输出长度。','','## 汇总','','| 并发 | 指标 | chunk8192 | chunk16384 | 相对变化 |','|---:|---|---:|---:|---:|']
 for cc in [2,4,8]:
  a,b=[out['arms'][str(cc)][m] for m in ['chunk8k','chunk16k']]
  for m,k in [('ttft_ms','mean'),('ttft_ms','p50'),('ttft_ms','p95'),('ttft_ms','p99'),('tpot_ms','mean'),('e2e_ms','mean'),('requests_per_s',None),('new_tokens',None),('prefill_batches',None)]:
   av=a[m][k] if k else a[m];bv=b[m][k] if k else b[m];lines.append(f'| {cc} | {m} {k or ""} | {av:.3f} | {bv:.3f} | {(bv/av-1)*100:+.2f}% |')
 lines+=['','## 两轮重复性','','| 并发 | 轮 | 平均TTFT变化 | P95变化 | 吞吐变化 |','|---:|---:|---:|---:|---:|']
 for x in out['rounds']:lines.append(f"| {x['concurrency']} | {x['rep']} | {x['ttft_change_pct']:+.2f}% | {x['p95_change_pct']:+.2f}% | {x['throughput_change_pct']:+.2f}% |")
 lines+=['',f"预先固定的通用默认值筛选：{'通过' if out['universal_default_screen_passed'] else '未通过，不推荐将16K作为并发2–8统一默认值'}。条件为每并发每轮平均TTFT不增加、请求吞吐不下降、TTFT P95恶化不超过3%；这是保守筛选，不是显著性检验。",'']
 lines+=['','## 全部分位数（ms，TPOT为ms/token）','','| 并发 | 配置 | 指标 | mean | P50 | P75 | P90 | P95 | P99 |','|---:|---|---|---:|---:|---:|---:|---:|---:|']
 for cc in [2,4,8]:
  for mode in ['chunk8k','chunk16k']:
   for m in ['ttft_ms','tpot_ms','e2e_ms']:
    v=out['arms'][str(cc)][mode][m];lines.append(f'| {cc} | {mode} | {m} | '+' | '.join(f'{v[k]:.3f}' for k in ['mean','p50','p75','p90','p95','p99'])+' |')
 lines+=['','## 缓存与解释边界','']
 for cc in [2,4,8]:
  cp=[x for x in out['cache_pairs'] if x['concurrency']==cc];lines.append(f'- C{cc}：20对请求中{sum(x["equal"] for x in cp)}对API cached_tokens相同。')
 lines+=['','改分块会改变checkpoint可复用位置、合批和排队。即使payload一致，也不能把不同实际计算量下的全部端到端变化称为GDN内核收益。客户端闭环完成后补请求，自然生成长度不同亦影响后续到达时刻。P日志新token量和逐请求API缓存均保存；没有删除慢请求或失败结果。每臂每并发仅20条，P95/P99为线性插值描述统计，不能代表长期生产尾部。','','## 正确性与复现','',f'- 新增16k边界/多序列验证{len(test)}例：输出、最终状态、选中checkpoint、checkpoint续算、CUDA Graph同地址改变target回放均与原版逐位一致。',f'- memcheck {len(mem)}例、0 error；真实在线shadow {len(checks)}次全部一致。', f"- 正式批次BCG日志：{out['bcg_logged_batches']}；所有P部署backend=breakable。", '- 原版等价性通过不等于独立FP32数值验证全部通过，沿用此前已报告的原版FP32阈值失败限制。', '- 完整命令config/p_*.json、d_default.json；scripts/03_launch.py和20_concurrent.py可复现；results/requests.jsonl与batches.jsonl为原始数据；evidence/runtime_hashes_at_launch.json冻结内核。', f"- 所有正式批次CPU cgroup新增限流周期数：{out['formal_cpu_throttled_periods']}。这不排除其他CPU调度/资源共享干扰。相同配置两轮的布局和均值比较见summary.json same_arm_repeatability；布局相同也不能证明端到端时序相同。", '- 无安装目录变更。只停止本实验PID；GPU释放见evidence/gpu_final.csv。共享主机其他GPU任务未操作，CPU资源干扰不能排除。']
 lines+=['','## 后续优化方向（本轮未执行）','','- 保持8K、新GDN及其他参数，单独验证P的缓存感知调度，例如FCFS对LPM；目标是减少共享前缀请求在可复用状态形成前的重复prefill。当前schedule_policy.py含waiting-queue前缀匹配及共享前缀降优先级逻辑，但在本负载上的收益尚未测量，公平性和尾延迟必须一并验证。', '- 端到端归因需要关联每请求的P排队、GPU执行、KV交接和D首token时间；客户端TTFT、总新tokens及聚合batch日志不足以完全定位时序波动。', '- 继续保留同输入真实GDN回放作为算子证据。固定到达时刻/固定输出长度只适合单独诊断，不替换本轮真实闭环自然输出数据。', '- 本安装版本enable_dynamic_chunking用于PP；gdn_backend.py还会因该开关退出默认FlashInfer选择，因此不把它当成TP1并发优化的一键方案。源码快照见evidence/。']
 (R/'README.md').write_text('\n'.join(lines)+'\n')
 with (R/'results/metrics.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['concurrency','mode','metric','mean','p50','p75','p90','p95','p99'])
  for cc,v in out['arms'].items():
   for mode,metrics in v.items():
    for m in ['ttft_ms','tpot_ms','e2e_ms']:w.writerow([cc,mode,m,*[metrics[m][k] for k in ['mean','p50','p75','p90','p95','p99']]])
 (R/'evidence/final_audit.json').write_text(json.dumps(dict(status='passed',only_p_command_difference={'chunked-prefill-size':[8192,16384]},same_fixed_hook=True,data_sha256=sha(R/'config/fixed10.json'),runtime_matches_launch=True,formal_requests=120,total_requests=240,shadow_checks=len(checks),order='ABBA for each C2/4/8'),indent=2))
 manifest={str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for d in ['scripts','config','results','logs','evidence'] for p in (R/d).rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json'}
 manifest['README.md']={'sha256':sha(R/'README.md'),'bytes':(R/'README.md').stat().st_size}
 (R/'evidence/artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:out[k] for k in ['formal_requests','all_requests','shadow_checks','rounds']},indent=2))
if __name__=='__main__':main()
