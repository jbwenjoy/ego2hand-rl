$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.uv-python'
$env:HF_HOME = Join-Path $PSScriptRoot '.cache/huggingface'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $PSScriptRoot '.cache/torchinductor'
$env:TRITON_CACHE_DIR = Join-Path $PSScriptRoot '.cache/triton'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
& uv @args
exit $LASTEXITCODE
