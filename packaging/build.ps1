# Native tools write progress to stderr; Windows PowerShell 5.1 would treat that as
# fatal under 'Stop'. Every native call below checks $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent) -ErrorAction Stop
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create build environment' }
}
& .venv/Scripts/python.exe -m pip install -r packaging/requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw 'Build dependency installation failed' }
& .venv/Scripts/python.exe -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .venv/Scripts/python.exe packaging/make_icon.py
if ($LASTEXITCODE -ne 0) { throw 'Icon generation failed' }
& .venv/Scripts/python.exe -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& .venv/Scripts/python.exe -m PyInstaller --noconfirm packaging/iShareScreen-Desktop.spec
if ($LASTEXITCODE -ne 0) { throw 'Executable build failed' }
& .venv/Scripts/python.exe -m PyInstaller --noconfirm packaging/iShareScreen-LiquidGlass.spec
if ($LASTEXITCODE -ne 0) { throw 'Liquid Glass executable build failed' }
Copy-Item -LiteralPath dist/iShareScreen-Desktop.exe -Destination dist/iShareScreen.exe -Force -ErrorAction Stop
Write-Output 'Built dist/iShareScreen-Desktop.exe (also dist/iShareScreen.exe) and dist/iShareScreen-LiquidGlass.exe'

