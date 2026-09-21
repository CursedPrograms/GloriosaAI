# Create the virtual environment (psdenv) and install dependencies.
Set-Location $PSScriptRoot
if (-not (Test-Path psdenv)) { python -m venv psdenv }
& .\psdenv\Scripts\python.exe -m pip install --upgrade pip
& .\psdenv\Scripts\python.exe -m pip install -r requirements.txt
