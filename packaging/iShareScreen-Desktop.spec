from pathlib import Path
root = Path(SPECPATH).parent
a = Analysis([str(root/'packaging/desktop_launcher.py')],pathex=[str(root/'src')],
    datas=[(str(root/'LICENSE'),'.'),(str(root/'THIRD_PARTY_NOTICES.md'),'.'),
           (str(root/'src/isharescreen/gui/static/icon.svg'),'isharescreen/gui/static')],
    hiddenimports=[],
    # PyAV stays in: the experimental High Performance path decodes with it.
    excludes=['tkinter','PyQt5','PyQt6','PySide6.QtWebEngineCore','PySide6.QtWebEngineWidgets',
              'PySide6.QtQml','PySide6.QtQuick','pytest','wgpu','glfw','aioquic','aiohttp'])
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='iShareScreen-Desktop',
    debug=False,strip=False,upx=False,console=False,icon=str(root/'packaging/isharescreen.ico'))
