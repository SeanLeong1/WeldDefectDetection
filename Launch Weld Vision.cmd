@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" submission.py gui
) else (
  python submission.py gui
)
if errorlevel 1 pause
