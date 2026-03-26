$ErrorActionPreference = "Continue"
$start = Get-Date
$proc = Start-Process "target\debug\rust-llm.exe" -ArgumentList '--model e:\models\qwen\Qwen3.5-9B-Q4_K_M.gguf --use-gpu --prompt "hello" --max-tokens 10 --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 --context-size 2048 --batch-size 1' -NoNewWindow -PassThru -RedirectStandardOutput "test_output.log" -RedirectStandardError "test_error.log"
Write-Host "Process started with PID: $($proc.Id)"
$count = 0
while (-not $proc.HasExited -and $count -lt 30) {
    Start-Sleep -Seconds 1
    $count++
    Write-Host "Waiting... ($count seconds)"
}
if (-not $proc.HasExited) {
    Write-Host "Process still running after 30 seconds, stopping..."
    Stop-Process -Id $proc.Id -Force
}
Write-Host "Exit code: $($proc.ExitCode)"
Write-Host "--- STDOUT ---"
Get-Content test_output.log -ErrorAction SilentlyContinue | Select-Object -First 100
Write-Host "--- STDERR ---"
Get-Content test_error.log -ErrorAction SilentlyContinue | Select-Object -First 100
