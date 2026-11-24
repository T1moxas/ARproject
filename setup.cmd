@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 exit /b 1
)
set "AR_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "wheelhouse\*.whl" (
    "%AR_PYTHON%" -m pip install --no-index --find-links wheelhouse -r project\requirements.txt
    if not errorlevel 1 goto check
)
"%AR_PYTHON%" -m pip install -r project\requirements.txt
if errorlevel 1 exit /b 1
:check
"%AR_PYTHON%" -c "import cv2, websockets; assert hasattr(cv2.aruco, 'ArucoDetector'); assert any(s.split(':', 1)[1].strip() != 'NONE' for s in cv2.getBuildInformation().splitlines() if s.lstrip().startswith('GUI:')); print('OpenCV', cv2.__version__, 'with ArUco and HighGUI, websockets', websockets.__version__)"
exit /b %errorlevel%
