$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defaultRoot = (Resolve-Path (Join-Path $repoRoot ".")).Path
$defaultStorageRoot = Join-Path $defaultRoot ".localdata"

if (-not $env:PROJECT_STORAGE_ROOT) {
    $env:PROJECT_STORAGE_ROOT = $defaultStorageRoot
}

$projectStorageRoot = [System.IO.Path]::GetFullPath($env:PROJECT_STORAGE_ROOT)
$env:PROJECT_STORAGE_ROOT = $projectStorageRoot

if (-not $env:TORCH_HOME) { $env:TORCH_HOME = Join-Path $projectStorageRoot ".cache\torch" }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $projectStorageRoot ".cache\huggingface" }
if (-not $env:HF_DATASETS_CACHE) { $env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME "datasets" }
if (-not $env:HUGGINGFACE_HUB_CACHE) { $env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub" }
if (-not $env:VLLM_CACHE_ROOT) { $env:VLLM_CACHE_ROOT = Join-Path $projectStorageRoot ".cache\vllm" }
if (-not $env:TRITON_CACHE_DIR) { $env:TRITON_CACHE_DIR = Join-Path $projectStorageRoot ".cache\triton" }
if (-not $env:XDG_CACHE_HOME) { $env:XDG_CACHE_HOME = Join-Path $projectStorageRoot ".cache" }
if (-not $env:TMPDIR) { $env:TMPDIR = Join-Path $projectStorageRoot "tmp" }
if (-not $env:TMP) { $env:TMP = $env:TMPDIR }
if (-not $env:TEMP) { $env:TEMP = $env:TMPDIR }

$paths = @(
    (Join-Path $projectStorageRoot "data\grpo"),
    (Join-Path $projectStorageRoot "models\grpo"),
    (Join-Path $projectStorageRoot "models\eval"),
    (Join-Path $projectStorageRoot "venvs"),
    $env:TORCH_HOME,
    $env:HF_HOME,
    $env:HF_DATASETS_CACHE,
    $env:HUGGINGFACE_HUB_CACHE,
    $env:VLLM_CACHE_ROOT,
    $env:TRITON_CACHE_DIR,
    $env:TMPDIR
)

foreach ($path in $paths) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

Write-Host "PROJECT_STORAGE_ROOT=$($env:PROJECT_STORAGE_ROOT)"
Write-Host "TORCH_HOME=$($env:TORCH_HOME)"
Write-Host "HF_HOME=$($env:HF_HOME)"
Write-Host "HF_DATASETS_CACHE=$($env:HF_DATASETS_CACHE)"
Write-Host "HUGGINGFACE_HUB_CACHE=$($env:HUGGINGFACE_HUB_CACHE)"
Write-Host "VLLM_CACHE_ROOT=$($env:VLLM_CACHE_ROOT)"
Write-Host "TRITON_CACHE_DIR=$($env:TRITON_CACHE_DIR)"
Write-Host "XDG_CACHE_HOME=$($env:XDG_CACHE_HOME)"
Write-Host "TMPDIR=$($env:TMPDIR)"
