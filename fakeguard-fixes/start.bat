@echo off
setlocal EnableDelayedExpansion
title FakeGuard Launcher

echo.
echo  ============================================================
echo    FakeGuard  --  Fake News Detection System
echo  ============================================================
echo.

:: ── Confirm we are in the project root ──────────────────────────
if not exist "src\app\main.py" (
    echo  [ERROR] Run this script from the project root folder.
    echo          e.g.  cd C:\Users\A\Desktop\fake-news-detector
    echo                start.bat
    echo.
    pause
    exit /b 1
)

:: ── Check virtual-env activation ────────────────────────────────
if defined VIRTUAL_ENV (
    echo  [OK] Virtual env detected: %VIRTUAL_ENV%
    set PYTHON=python
    set UVICORN=uvicorn
    set STREAMLIT=streamlit
) else (
    echo  [INFO] No active venv found. Using system Python.
    echo         Tip: activate with  .venv\Scripts\activate  first.
    set PYTHON=python
    set UVICORN=uvicorn
    set STREAMLIT=streamlit
)

echo.
echo  Starting FastAPI backend  (port 8000) ...
start "FakeGuard API  [http://localhost:8000]" cmd /k ^
    "cd /d %~dp0 && echo API starting... && uvicorn src.app.main:app --host 0.0.0.0 --port 8000"

echo  Waiting 4 s for the API to bind its port ...
timeout /t 4 /nobreak > nul

echo  Starting Streamlit dashboard  (port 8501) ...
start "FakeGuard UI   [http://localhost:8501]" cmd /k ^
    "cd /d %~dp0 && echo Dashboard starting... && streamlit run src/app/streamlit_app.py --server.port 8501"

echo.
echo  ============================================================
echo    Both services are starting in their own windows.
echo.
echo    API       :  http://localhost:8000
echo    API docs  :  http://localhost:8000/docs
echo    Dashboard :  http://localhost:8501
echo.
echo    To stop everything: close the two "FakeGuard" windows.
echo    This launcher window can be closed safely now.
echo  ============================================================
echo.
pause
