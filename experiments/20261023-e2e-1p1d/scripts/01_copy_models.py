import concurrent.futures as cf, json, os, pathlib, shutil, subprocess
root=pathlib.Path('/root/gdn35/20261023-e2e-1p1d')
jobs=[];manifest=[]
for name in ['prefill_model','decode_model','spec_model']:
 src=pathlib.Path('/'+name); dst=root/'models'/name;dst.mkdir(parents=True,exist_ok=True)
 for f in src.iterdir():
  if not f.is_file():continue
  target=dst/f.name; size=f.stat().st_size
  manifest.append(dict(source=str(f),target=str(target),size=size))
  if target.exists() and target.stat().st_size==size:continue
  if size>512*1024*1024:
   with target.open('wb') as h:h.truncate(size)
   for off in range(0,size,512*1024*1024):jobs.append(('dd',f,target,off,min(512*1024*1024,size-off)))
  else:jobs.append(('copy',f,target,0,size))
def run(j):
 kind,src,dst,off,n=j
 if kind=='copy':shutil.copy2(src,dst)
 else:subprocess.run(['dd',f'if={src}',f'of={dst}','bs=8M',f'skip={off}',f'seek={off}',f'count={n}','iflag=skip_bytes,count_bytes','oflag=seek_bytes','conv=notrunc','status=none'],check=True)
with cf.ThreadPoolExecutor(max_workers=8) as pool:
 for i,_ in enumerate(pool.map(run,jobs)):
  if i%16==0:print('copied',i+1,'/',len(jobs),flush=True)
for m in manifest:assert pathlib.Path(m['target']).stat().st_size==m['size']
(root/'evidence/model_manifest.json').write_text(json.dumps(manifest,indent=2))
print('COPY_COMPLETE',len(manifest),flush=True)
