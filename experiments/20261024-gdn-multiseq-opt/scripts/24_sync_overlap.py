"""Quantify overlap of host stream synchronization with observed GPU kernels.

Overlap is temporal evidence, not stream-dependency or root-cause proof.
"""
import gzip,json,pathlib,sys,bisect
p=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);raw=json.load(gzip.open(p,'rt'))
events=[x for x in raw['traceEvents'] if x.get('ph')=='X' and x.get('dur',0)>0]
intervals=sorted((x['ts'],x['ts']+x['dur']) for x in events if x.get('cat')=='kernel');merged=[]
for a,b in intervals:
 if not merged or a>merged[-1][1]:merged.append([a,b])
 else:merged[-1][1]=max(merged[-1][1],b)
ends=[x[1] for x in merged]
def overlap(a,b):
 i=bisect.bisect_right(ends,a);total=0
 while i<len(merged) and merged[i][0]<b:
  total+=max(0,min(b,merged[i][1])-max(a,merged[i][0]));i+=1
 return total
sync=[x for x in events if x['name']=='cudaStreamSynchronize'];records=[]
for x in sync:
 a,b=x['ts'],x['ts']+x['dur'];parents=[z for z in events if z.get('cat') in ('cpu_op','user_annotation') and z.get('pid')==x.get('pid') and z.get('tid')==x.get('tid') and z['ts']<=a and z['ts']+z['dur']>=b]
 records.append(dict(start_us=a,duration_ms=x['dur']/1000,kernel_overlap_ms=overlap(a,b)/1000,parent_scopes=[z['name'] for z in sorted(parents,key=lambda z:z['dur'])[:6]]))
result=dict(trace=str(p),sync_calls=len(sync),sync_summed_ms=sum(x['duration_ms'] for x in records),sync_summed_kernel_overlap_ms=sum(x['kernel_overlap_ms'] for x in records),longest=sorted(records,key=lambda x:-x['duration_ms'])[:15],caveat='Temporal GPU activity overlap only; does not prove the exact dependency stream or avoidable host delay.')
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
