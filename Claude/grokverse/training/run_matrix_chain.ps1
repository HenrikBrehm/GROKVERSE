# Chained launcher for the remaining study blocks (master prompt SS14, SS20).
# Each block is launched in sequence; grokverse.matrix skips runs whose manifest
# already says "completed", so re-running this script is safe and resumable at
# block granularity. Started detached so it survives the end of an AI session.
#
#   powershell -ExecutionPolicy Bypass -File run_matrix_chain.ps1
$ErrorActionPreference = 'Continue'
$train = 'C:\Users\henri\Documents\Brain\bwki\Claude\grokverse\training'
$py    = 'C:\Users\henri\Documents\Brain\bwki\Claude\grokverse\.venv\Scripts\python.exe'
$log   = Join-Path $train 'results\matrix_launch.log'
Set-Location $train
$commit = (git rev-parse --short HEAD)
$stamp  = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
Add-Content $log "`n=== matrix launch $stamp commit $commit (restart after power loss) ==="
foreach ($block in @('confound','param_matched','twohot')) {
    $t0 = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    Add-Content $log "--- block $block start $t0"
    & $py -m grokverse.matrix --block $block --workers 8 2>&1 | Add-Content $log
    $code = $LASTEXITCODE
    $t1 = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    Add-Content $log "--- block $block exit $code end $t1"
}
Add-Content $log "=== matrix chain finished $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) ==="
