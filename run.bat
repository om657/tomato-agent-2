@echo off
title Agri Super Agent
echo ========================================
echo   Agricultural Super AI Agent
echo   Starting...
echo ========================================
echo.

REM Try system Python first, then venv
where python >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [OK] Using system Python
    pip install -q streamlit requests pillow streamlit-folium folium geopy matplotlib pandas python-dateutil 2>nul
    python -m streamlit run app.py --server.port 8501
) else if exist .venv\Scripts\python.exe (
    echo [OK] Using virtual environment
    .venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
) else (
    echo [ERROR] Python not found! Install Python from https://python.org
    pause
)
