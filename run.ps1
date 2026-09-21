# Run GloriosaAI, setting up the virtual environment first if needed.
Set-Location $PSScriptRoot
if (-not (Test-Path psdenv)) { & .\setup.ps1 }
& .\psdenv\Scripts\python.exe main.py
