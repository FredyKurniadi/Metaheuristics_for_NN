param(
  [switch]$RecreateVenv = $false
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "[setup] project root: $root"
Push-Location $root

$pythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
  try {
    py -3.11 -c "import sys; print(sys.version)" | Out-Null
    $pythonCmd = "py -3.11"
  } catch {
    $pythonCmd = $null
  }
}

if (-not $pythonCmd) {
  $pythonExe = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonExe) {
    try {
      python -c "import sys; assert sys.version_info[:2] >= (3, 10)" | Out-Null
      $pythonCmd = "python"
    } catch {
      $pythonCmd = $null
    }
  }
}

if (-not $pythonCmd) {
  throw "Python 3.10+ tidak ditemukan."
}

if ($RecreateVenv -and (Test-Path ".venv")) {
  Remove-Item -Recurse -Force .venv
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Invoke-Expression "$pythonCmd -m venv .venv"
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r train/requirements.txt

$pyVer = .\.venv\Scripts\python.exe --version
$pipVer = .\.venv\Scripts\python.exe -m pip --version
$freeze = .\.venv\Scripts\python.exe -m pip freeze

$lines = @()
$lines += "# Reproducibility"
$lines += ""
$lines += "Last updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$lines += ""
$lines += "## Environment Versions"
$lines += "- Python: $pyVer"
$lines += "- pip: $pipVer"
$lines += ""
$lines += "## Library Versions"
$lines += '```'
$lines += $freeze
$lines += '```'

Set-Content -Path (Join-Path $root "docs\REPRODUCIBILITY.md") -Value ($lines -join "`r`n") -Encoding UTF8

Pop-Location
Write-Host "[setup] done"
