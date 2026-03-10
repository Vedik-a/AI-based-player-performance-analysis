@echo off
REM AI based Performance Enhancement - Quick Start Script

echo.
echo ========================================
echo   AI based Performance Enhancement
echo   Quick Start
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo [1] Installing backend dependencies...
cd backend
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To run the project:
echo.
echo 1. Start Backend (in one terminal):
echo    cd backend
echo    python app.py
echo.
echo 2. Start Frontend (in another terminal):
echo    cd frontend
echo    python -m http.server 8000
echo.
echo 3. Open browser and go to:
echo    http://localhost:8000
echo.
echo Backend API: http://localhost:5000
echo.
pause
