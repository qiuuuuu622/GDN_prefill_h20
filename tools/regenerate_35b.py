"""Rebuild the published kernels with the exact historical generators (CPU only)."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail on drift without modifying files')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        (work / 'scripts').mkdir()
        (work / 'evidence').mkdir()
        shutil.copyfile(ROOT / 'qwen35/upstream/installed_delta_rule_sm90.py', work / 'evidence/installed_delta_rule_sm90.py')
        for name in ('generate_dvsplit.py', 'generate_kernels.py'):
            shutil.copyfile(ROOT / 'qwen35/generators' / name, work / 'scripts' / name)
        subprocess.run([sys.executable, str(work / 'scripts/generate_kernels.py')], check=True, capture_output=True)
        for name in ('dvsplit_kernel.py', 'selected_kernel.py'):
            data = (work / 'scripts' / name).read_bytes()
            target = ROOT / 'qwen35/runtime' / name
            if args.check:
                if target.read_bytes() != data:
                    raise SystemExit(f'Generated kernel drift: {target}')
            else:
                target.write_bytes(data)
    print('35B kernels reproduce byte-for-byte' if args.check else '35B kernels regenerated')

if __name__ == '__main__':
    main()
