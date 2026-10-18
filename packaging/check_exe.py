"""Load each frozen native GUI and its Qt plugins without opening a remote session."""
import os
import subprocess
from pathlib import Path

for name in ('iShareScreen-Desktop', 'iShareScreen-LiquidGlass'):
    exe = Path(f'dist/{name}.exe')
    if not exe.exists():
        print(f'Skipped {name}: not built.')
        continue
    result = subprocess.run([str(exe.resolve()),'--smoke-test'],env={**os.environ,'QT_QPA_PLATFORM':'offscreen'},
                            capture_output=True,timeout=30)
    assert result.returncode==0, f'{name}: ' + result.stderr.decode(errors='replace')
    print(f'Passed: {name} starts, initializes Qt, and exits cleanly.')
