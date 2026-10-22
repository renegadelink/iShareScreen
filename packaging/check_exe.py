"""Load each frozen native GUI and its Qt plugins without opening a remote session."""
import os
import subprocess
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dist-dir', type=Path, default=Path('dist'))
args = parser.parse_args()
for name in ('iShareScreen-Desktop', 'iShareScreen-LiquidGlass'):
    exe = args.dist_dir / f'{name}.exe'
    if not exe.exists():
        print(f'Skipped {name}: not built.')
        continue
    result = subprocess.run([str(exe.resolve()),'--smoke-test'],env={**os.environ,'QT_QPA_PLATFORM':'offscreen'},
                            capture_output=True,timeout=30)
    assert result.returncode==0, f'{name}: ' + result.stderr.decode(errors='replace')
    print(f'Passed: {name} starts, initializes Qt, and exits cleanly.')
