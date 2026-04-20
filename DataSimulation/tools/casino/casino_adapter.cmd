@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PY_EXE=%PYTHON%"
if not defined PY_EXE set "PY_EXE=python"
"%PY_EXE%" "%SCRIPT_DIR%casino_adapter.py" %*
exit /b %ERRORLEVEL%
