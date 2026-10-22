@echo off
cd /d "%~dp0.."
echo ======================================================================
echo Starting Multi-Object Tracking Studio
echo ======================================================================
echo 1. Starting FastAPI backend server on port 8000...
start "MOT Backend" cmd /c "py -3.11 -m uvicorn server:app --host 0.0.0.0 --port 8000"
timeout /t 2 >nul
echo 2. Launching React Web Application on port 3000...
start "MOT Frontend" cmd /c "npm --prefix frontend run dev"
timeout /t 3 >nul
echo 3. Opening web browser to http://localhost:3000 ...
start http://localhost:3000
echo.
echo Application is running!
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8000
echo ======================================================================
pause
