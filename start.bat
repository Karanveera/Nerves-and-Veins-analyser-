@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed. Get it from python.org and tick "Add to PATH". & pause & exit /b)
if not exist .venv (
    echo First run: setting things up, this takes a minute...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip -q
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
echo Starting Vein Detector - your browser will open...
streamlit run app.py
pause
