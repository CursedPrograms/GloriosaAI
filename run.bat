@echo off
rem Run GloriosaAI, setting up the virtual environment first if needed.
cd /d "%~dp0"
if not exist psdenv call setup.bat
psdenv\Scripts\python main.py
pause
