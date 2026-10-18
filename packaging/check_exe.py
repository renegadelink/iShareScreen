"""Load the frozen native GUI and Qt plugins without opening a remote session."""
import os
import subprocess
from pathlib import Path

exe=str(Path('dist/iShareScreen-Desktop.exe').resolve())
result=subprocess.run([exe,'--smoke-test'],env={**os.environ,'QT_QPA_PLATFORM':'offscreen'},
                      capture_output=True,timeout=30)
assert result.returncode==0, result.stderr.decode(errors='replace')
print('Passed: frozen native desktop starts, initializes Qt, and exits cleanly.')
