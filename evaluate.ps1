# =================================================================
# PowerShell helper to run PatchCore Evaluation
# =================================================================
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "━━━ Launching Industrial Anomaly Detection Metrics Suite ━━━" -ForegroundColor Cyan

$VenvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $VenvPath)) {
    Write-Error "[ERROR] Virtual environment .venv not found. Please create it or run from your active environment."
    Exit
}

Write-Host "[INFO] Activating virtual environment .venv..." -ForegroundColor Gray
& "$VenvPath\Scripts\Activate.ps1"

Write-Host "[INFO] Executing evaluate.py $args..." -ForegroundColor Gray
python "$PSScriptRoot\evaluate.py" $args

Write-Host "`n━━━ Finished ━━━" -ForegroundColor Cyan
Read-Host -Prompt "Press Enter to exit"
