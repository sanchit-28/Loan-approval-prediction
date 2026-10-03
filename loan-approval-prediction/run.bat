@echo off
REM Loan Approval Prediction - one-click runner for Windows
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed. Install it from https://www.python.org/downloads/ and tick "Add Python to PATH". & pause & exit /b 1)
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo Installing libraries (first run only)...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo Running the pipeline (takes a few minutes)...
python src\loan_approval.py
echo.
echo Done. Charts: reports\figures   Metrics: results\results_summary.json
pause
