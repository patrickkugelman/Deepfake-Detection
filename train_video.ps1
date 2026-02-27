# Standalone Video Agent Training Script
# Usage: .\train_video.ps1 [--epochs 50] [--batch_size 4]

param(
    [int]$epochs = 50,
    [int]$batch_size = 4
)

$ErrorActionPreference = "Stop"

$ROOT_DIR = Get-Location
$SCRIPT_PATH = Join-Path $ROOT_DIR "agents\video\train_resnet_pro.py"
$DATA_DIR = Join-Path $ROOT_DIR "agents\video\dataset"
$OUTPUT_DIR = Join-Path $ROOT_DIR "agents\video\models"

Write-Host ">>> Training Video Agent" -ForegroundColor Cyan
Write-Host ">>> Epochs: $epochs, Batch Size: $batch_size" -ForegroundColor Gray

if (!(Test-Path $SCRIPT_PATH)) {
    Write-Error "Training script not found: $SCRIPT_PATH"
    exit 1
}

if (!(Test-Path $DATA_DIR)) {
    Write-Error "Dataset directory not found: $DATA_DIR"
    exit 1
}

New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null

Write-Host ">>> Starting training..." -ForegroundColor Cyan

python $SCRIPT_PATH `
    --data_dir "$DATA_DIR" `
    --output_dir "$OUTPUT_DIR" `
    --epochs $epochs `
    --batch_size $batch_size

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed with exit code $LASTEXITCODE"
    exit 1
}

Write-Host ">>> Video Agent Training Complete!" -ForegroundColor Green

