@echo off
:: =================================================================
:: Batch helper to run PatchCore Evaluation
:: =================================================================
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ━━━ Launching Industrial Anomaly Detection Metrics Suite ━━━

if not exist .venv\ (
    echo [ERROR] Virtual environment .venv not found. Please create it or run from your active environment.
    goto end
)

echo [INFO] Activating virtual environment .venv...
call .venv\Scripts\activate.bat

echo [INFO] Executing evaluate.py %*...
python evaluate.py %*

:end
echo.
echo ━━━ Finished ━━━
pause
