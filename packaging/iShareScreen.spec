from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH).parent
datas = [(str(root / 'src/isharescreen/gui/static'), 'isharescreen/gui/static'),
         (str(root / 'src/isharescreen/frontend/wt/static'), 'isharescreen/frontend/wt/static'),
         (str(root / 'LICENSE'), '.'),
         (str(root / 'THIRD_PARTY_NOTICES.md'), '.')]
binaries = []
hiddenimports = collect_submodules('isharescreen')
for package in ('av', 'wgpu', 'glfw', 'rendercanvas', 'aioquic', 'sounddevice', '_sounddevice_data'):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
a = Analysis([str(root / 'packaging/launcher.py')], pathex=[str(root / 'src')],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=['tkinter', 'PySide6', 'PyQt6', 'PyQt5', 'pytest', 'numpy.tests'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='iShareScreen',
          debug=False, strip=False, upx=False, console=True,
          icon=str(root / 'packaging/isharescreen.ico'))
