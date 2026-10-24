"""Generate public tables from archived measurements; --check verifies freshness.

No GPU or private prompt data required. E2E quantiles are independently recomputed
from per-request timings and checked against each original summary.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def stats(values):
    values = sorted(values)
    out = {'mean': statistics.mean(values)}
    for q in (50, 75, 90, 95, 99):
        index = (len(values) - 1) * q / 100
        lo = int(index)
        out[f'p{q}'] = values[lo] + (values[math.ceil(index)] - values[lo]) * (index-lo)
    return out

def csv_text(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()

def generate():
    tables = {}
    quantiles, changes = [], []
    reported = []
    for line in (ROOT/'docs/122b-original-report.md').read_text().splitlines():
        if line.startswith('| ') and (' ms |' in line or line.startswith(('| 8192 +','| 16384 +','| 32768 +','| 4096 +','| 2×8192','| 4×2048'))):
            cells = [v.strip().replace('**','') for v in line.strip('|').split('|')]
            if len(cells) == 4:
                reported.append(dict(lengths=cells[0],baseline_ms=cells[1].replace(' ms',''),split_ms=cells[2].replace(' ms',''),reported_gain=cells[3],source_commit='9a3aaeabf8efee7394cf46ff37f163101c72c5e9',evidence='original README rounded values; no raw timing samples'))
    assert len(reported) == 10
    tables['122b_reported.csv'] = csv_text(reported)
    runs = [
        ('20261024-gdn-checkpoint-opt', 'formal_summary.json', 'formal_stock_abba', 'baseline', 'sparse', [1]),
        ('20261024-gdn-concurrency24', 'summary.json', 'formal', 'baseline', 'sparse', [2,4]),
        ('20261024-gdn-multiseq-opt', 'summary.json', 'formal', 'baseline', 'multi', [2,4,8]),
        ('20261024-gdn-chunk16k', 'summary.json', 'formal', 'chunk8k', 'chunk16k', [2,4,8]),
    ]
    for run, summary_file, phase, a, b, concurrencies in runs:
        summary = read(EXP/run/'results'/summary_file)
        requests = lines(EXP/run/'results/requests.jsonl')
        for concurrency in concurrencies:
            arms = summary['arms'] if concurrency == 1 else summary['arms'][str(concurrency)]
            for arm in (a,b):
                selected = [r for r in requests if r['phase']==phase and r['mode']==arm and r.get('concurrency',1)==concurrency]
                assert len(selected)==20, (run,concurrency,arm,len(selected))
                assert all(r['status']==200 for r in selected)
                assert len({(r['rep'],r['sample_id']) for r in selected})==20
                for metric in ('ttft_ms','tpot_ms','e2e_ms'):
                    result = stats([r[metric] for r in selected])
                    for key,value in result.items():
                        assert math.isclose(value,arms[arm][metric][key],abs_tol=1e-7), (run,arm,metric,key)
                    quantiles.append(dict(experiment=run,concurrency=concurrency,arm=arm,metric=metric,n=20,**result))
            for metric in ('ttft_ms','tpot_ms','e2e_ms'):
                av,bv=arms[a][metric]['mean'],arms[b][metric]['mean']
                changes.append(dict(experiment=run,concurrency=concurrency,metric=metric,baseline=a,candidate=b,baseline_mean=av,candidate_mean=bv,change_percent=(bv/av-1)*100))
    tables['e2e_quantiles.csv']=csv_text(quantiles)
    tables['e2e_changes.csv']=csv_text(changes)
    shape=EXP/'20261023-shape-study/evidence'
    full=lines(shape/'full.jsonl');dv=lines(shape/'dv_full.jsonl')
    rows=[]
    for lengths in [(512,),(4096,),(8192,),(16384,),(32768,),(8192,1024,1024,1024),(32768,1024,1024,1024),(8192,8192),(2048,2048,2048,2048)]:
        group=[r for r in full if tuple(r['lengths'])==lengths and r['valid']]
        base=next(r for r in group if r['strategy']=='noncp')
        cp=min((r for r in group if r['strategy'].startswith(('cp_','split_'))),key=lambda r:r['median_ms'])
        candidates=[r for r in dv if tuple(r['lengths'])==lengths and r['valid'] and r['strategy']=='dvsplit']
        assert len(candidates)==1, (lengths,[r['strategy'] for r in dv if tuple(r['lengths'])==lengths])
        v=candidates[0]
        rows.append(dict(lengths=str(list(lengths)),noncp_ms=base['median_ms'],best_scanned_cp_strategy=cp['strategy'],best_scanned_cp_ms=cp['median_ms'],dv_ms=v['median_ms'],caveat='CP and V measured in separate historical scans; synthetic, no checkpoints'))
    tables['35b_cp_vs_v.csv']=csv_text(rows)
    for filename,source in [('35b_checkpoint_shapes.csv','dispatch_sweep.json'),('35b_real_replay.csv','real_replay.json')]:
        rows=[]
        for row in read(EXP/'20261024-gdn-multiseq-opt/results'/source):
            choice=row.get('choice',row.get('method'))
            rows.append(dict(lengths=str(row['lengths']),choice=choice,baseline_ms=row['baseline']['graph_median_ms'],full_v_ms=row['full']['graph_median_ms'],split_v_ms=row['dv']['graph_median_ms'],chosen_ms=row[choice]['graph_median_ms'],speedup=row['speedup']))
        tables[filename]=csv_text(rows)
    report=['<!-- Generated by tools/build_tables.py; do not edit. -->','# 统一数据表','',
            '数值来自公开归档；E2E 各分位数从逐请求记录重新计算并与历史 summary 核对。不同实验不可拼接成一次 A/B。',
            '延迟变化为 `(B/A−1)×100%`，负值更快；算子 speedup 为 `A/B`。每臂每并发 20 条，P99 不代表生产尾延迟。','']
    for title,file in [('122B 原报告数值（非原始逐次计时）','122b_reported.csv'),('端到端均值变化','e2e_changes.csv'),('端到端全部分位数','e2e_quantiles.csv'),('35B：CP 与 V 扫描','35b_cp_vs_v.csv'),('35B：checkpoint 合成形状','35b_checkpoint_shapes.csv'),('35B：真实输入回放','35b_real_replay.csv')]:
        data=list(csv.DictReader(io.StringIO(tables[file])));keys=list(data[0])
        report += [f'## {title}', '', f'[下载 CSV](../data/{file})','', '| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']
        for row in data:
            cells=[]
            for k in keys:
                v=row[k]
                try: v=f'{float(v):.4f}' if '.' in v and k not in ('experiment','lengths','caveat') else v
                except ValueError:pass
                cells.append(v.replace('|','\\|'))
            report.append('| '+' | '.join(cells)+' |')
        report.append('')
    return tables,'\n'.join(report).rstrip()+'\n'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    tables,report=generate()
    outputs={ROOT/'data'/name:text for name,text in tables.items()}
    outputs[ROOT/'docs/tables.md']=report
    for path,text in outputs.items():
        if args.check:
            assert path.read_text()==text, f'Stale generated table: {path}'
        else:path.write_text(text)
    print(f'{len(outputs)} generated tables/reports; 360 formal request records verified')

if __name__=='__main__':main()
