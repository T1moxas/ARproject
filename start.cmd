@echo off
setlocal
set "AR_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "AR_PYTHON=%~dp0.venv\Scripts\python.exe"
pushd "%~dp0project"
if errorlevel 1 exit /b 1
"%AR_PYTHON%" -X utf8 -B stereo_live.py --height-map --height-max 0.5 --height-gamma 0.5 --height-noise-mm 15 --change-mm 10 --size 640x360 --update-mode changes --check-interval 0.5 --update-interval 1 --fill surface --appearance smooth --near 0.5 --far 2.5 --aruco-if-ready --no-aruco-project-halo --lan-preview --lan-preview-port 8765 %*
set "ar_result=%errorlevel%"
popd
if not "%ar_result%"=="0" (
    echo The program stopped with an error. See the message above.
    pause
)
exit /b %ar_result%
