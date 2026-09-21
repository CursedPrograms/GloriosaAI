@echo off
rem Create the virtual environment (psdenv) and install dependencies.
cd /d "%~dp0"
if not exist psdenv python -m venv psdenv || exit /b 1
psdenv\Scripts\python -m pip install --upgrade pip
psdenv\Scripts\python -m pip install -r requirements.txt
