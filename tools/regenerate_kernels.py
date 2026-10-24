"""Rebuild all packaged CuTe kernels from pinned sources without importing CUDA."""
import argparse,pathlib,shutil,subprocess,sys,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
 with tempfile.TemporaryDirectory() as directory:
  work=pathlib.Path(directory);(work/'scripts').mkdir();(work/'evidence').mkdir()
  for source in (ROOT/'kernels/upstream').glob('*.py'):shutil.copyfile(source,work/'evidence'/source.name)
  for source in (ROOT/'kernels/generators').glob('*.py'):shutil.copyfile(source,work/'scripts'/source.name)
  for name in ['generate_kernels.py','generate_cp_selected.py']:
   subprocess.run([sys.executable,str(work/'scripts'/name)],check=True,capture_output=True)
  for name in ['selected_kernel.py','dvsplit_kernel.py','cp_selected_kernel.py']:
   target=ROOT/'src/gdn_prefill/_kernels'/name;data=(work/'scripts'/name).read_bytes()
   if args.check:
    assert target.read_bytes()==data,f'Generated kernel drift: {name}'
   else:target.write_bytes(data)
 print('All three packaged kernels reproduce byte-for-byte')
if __name__=='__main__':main()
