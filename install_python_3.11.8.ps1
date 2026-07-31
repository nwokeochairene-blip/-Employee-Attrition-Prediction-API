# PowerShell helper to download and install Python 3.11.8 (per-user)
# Run in an elevated PowerShell only if you need InstallAllUsers=1
# Usage: Right-click -> Run with PowerShell, or run in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\install_python_3.11.8.ps1

$ErrorActionPreference = 'Stop'
$installer = Join-Path $env:TEMP 'python-3.11.8-amd64.exe'
Write-Output "Downloading installer to: $installer"
Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe' -OutFile $installer

Write-Output 'Running installer (per-user, quiet)'
Start-Process -FilePath $installer -ArgumentList '/quiet','InstallAllUsers=0','PrependPath=1','Include_pip=1','Include_launcher=1' -Wait

Write-Output 'Installer finished. Checking installed python versions...'
try {
    & py -3.11 --version
} catch {
    try { & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" --version } catch { Write-Output 'Python 3.11 not found in expected locations.' }
}

Write-Output 'If installation failed, try running PowerShell as Administrator to install system-wide.'
