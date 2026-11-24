@echo off
setlocal
set "AR_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "AR_PYTHON=%~dp0.venv\Scripts\python.exe"
pushd "%~dp0project"
if errorlevel 1 exit /b 1
"%AR_PYTHON%" -X utf8 -B stereo_live.py --swarm-only %*
set "ar_result=%errorlevel%"
popd
if not "%ar_result%"=="0" (
    echo AR Swarm stopped with an error. See the message above.
    pause
)
exit /b %ar_result%
