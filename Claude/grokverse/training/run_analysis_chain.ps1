# Detached launcher for the analysis driver (master prompt SS24 steps 10-11, INTERFACES SS0).
#
# Runs `grokverse.analysis.driver` over every COMPLETED run matching -Runs, at both
# pre-registered measurement points, after the training chain has finished. Written so it
# survives the end of an AI session, exactly like run_matrix_chain.ps1:
#
#   powershell -ExecutionPolicy Bypass -File run_analysis_chain.ps1
#   powershell -ExecutionPolicy Bypass -File run_analysis_chain.ps1 -SkipWait -Runs "*_frac0.3_seed*_arch25k"
#
# -SkipWait   start immediately (the driver only takes runs whose manifest says "completed",
#             so the primary block can be analysed while the control blocks still train)
# -Runs       glob of run directories (default: every study run)
# -Workers    driver threads (default 4; the 8 training workers share the same CPU)
#
# The git commit at launch is written to the log: it IS the analysis-code freeze commit of
# PREREGISTRATION.md SS12 if the driver is launched after the freeze, and the log is the
# evidence for it. Nothing here retrains, tunes, or interprets anything.
param(
    [switch]$SkipWait,
    [string]$Runs = "*_arch25k",
    [int]$Workers = 4,
    [int]$PollSeconds = 300
)
$ErrorActionPreference = 'Continue'
$train = 'C:\Users\henri\Documents\Brain\bwki\Claude\grokverse\training'
$py    = 'C:\Users\henri\Documents\Brain\bwki\Claude\grokverse\.venv\Scripts\python.exe'
$log   = Join-Path $train 'results\analysis_chain.log'
Set-Location $train

function Stamp { (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }
function TrainingRunning {
    $m = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
         Where-Object { $_.CommandLine -match 'grokverse\.matrix' }
    return ($m | Measure-Object).Count -gt 0
}

$commit = (git rev-parse --short HEAD)
$dirty  = if ((git status --porcelain -- grokverse tests).Length -gt 0) { 'DIRTY (uncommitted analysis code!)' } else { 'clean' }
Add-Content $log "`n=== analysis chain launch $(Stamp) commit $commit tree=$dirty runs='$Runs' workers=$Workers ==="

if (-not $SkipWait) {
    Add-Content $log "--- waiting for the training chain (grokverse.matrix) to exit; polling every $PollSeconds s"
    while (TrainingRunning) { Start-Sleep -Seconds $PollSeconds }
    Add-Content $log "--- training chain gone at $(Stamp)"
}

$t0 = Stamp
Add-Content $log "--- driver start $t0"
& $py -m grokverse.analysis.driver --runs $Runs --points crossing final --workers $Workers 2>&1 | Add-Content $log
$code = $LASTEXITCODE
Add-Content $log "--- driver exit $code end $(Stamp)"

# aggregation is optional: the module is specified (INTERFACES SS13) but may not exist yet
if (Test-Path (Join-Path $train 'grokverse\analysis\aggregate.py')) {
    Add-Content $log "--- aggregate start $(Stamp)"
    & $py -m grokverse.analysis.aggregate 2>&1 | Add-Content $log
    Add-Content $log "--- aggregate exit $LASTEXITCODE end $(Stamp)"
} else {
    Add-Content $log "--- aggregate skipped: grokverse/analysis/aggregate.py does not exist yet"
}
Add-Content $log "=== analysis chain finished $(Stamp) (driver exit $code) ==="
exit $code
