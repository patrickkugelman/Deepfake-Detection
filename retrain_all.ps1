# Deepfake Detector - Retrain All Pipeline
# Powershell Script to retrain all agents sequentially

$ErrorActionPreference = "Stop"

# Configuration
$TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"
$ROOT_DIR = Get-Location
$BACKUP_DIR = Join-Path $ROOT_DIR "backup_models"
$LOGS_DIR = Join-Path $ROOT_DIR "logs"

Write-Host ">>> Starting Deepfake Detector Retraining Pipeline - $TIMESTAMP" -ForegroundColor Cyan

# Ensure Directories Exist
if (!(Test-Path $BACKUP_DIR)) { New-Item -ItemType Directory -Path $BACKUP_DIR | Out-Null }
if (!(Test-Path $LOGS_DIR)) { New-Item -ItemType Directory -Path $LOGS_DIR | Out-Null }

# Helper Function for Training
function Train-Agent {
    param (
        [string]$AgentName,
        [string]$ScriptPath,
        [string]$DataDir,
        [string]$OutputDir,
        [string]$ExtraArgs = ""
    )

    Write-Host "`n--------------------------------------------------" -ForegroundColor Yellow
    Write-Host ">>> Training Agent: $AgentName" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------" -ForegroundColor Yellow

    # Paths
    $AgentDir = Split-Path -Parent $ScriptPath
    $FullDataDir = Join-Path $AgentDir $DataDir
    $FullOutputDir = Join-Path $AgentDir $OutputDir
    
    # Backup Existing Models
    if (Test-Path $FullOutputDir) {
        $AgentBackupDir = Join-Path $BACKUP_DIR "$AgentName\#$TIMESTAMP"
        New-Item -ItemType Directory -Path $AgentBackupDir -Force | Out-Null
        Write-Host ">>> Backing up existing models to: $AgentBackupDir" -ForegroundColor Gray
        Copy-Item -Path "$FullOutputDir\*" -Destination $AgentBackupDir -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        New-Item -ItemType Directory -Path $FullOutputDir -Force | Out-Null
    }

    # Construct Command
    $LogFile = Join-Path $LOGS_DIR "${AgentName}_train.log"
    
    # Check if script exists
    if (!(Test-Path $ScriptPath)) {
        Write-Error "xxx Training script not found: $ScriptPath"
    }

    Write-Host ">>> Data Directory: $FullDataDir"
    Write-Host ">>> Output Directory: $FullOutputDir"
    
    # Execute Training
    $CmdArgs = "$ScriptPath --data_dir `"$FullDataDir`" --output_dir `"$FullOutputDir`" $ExtraArgs"
    Write-Host "Executing: python $CmdArgs" -ForegroundColor DarkGray
    
    try {
        # Execute Python script with proper output handling
        Write-Host ">>> Starting training (this may take a while, especially for first-time model downloads)..." -ForegroundColor Cyan
        
        # Build argument list properly
        $argList = @(
            $ScriptPath,
            "--data_dir", $FullDataDir,
            "--output_dir", $FullOutputDir
        )
        
        # Add extra args if provided
        if ($ExtraArgs) {
            $argList += $ExtraArgs.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
        }
        
        # Execute with real-time output
        $process = Start-Process -FilePath "python" `
            -ArgumentList $argList `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput $LogFile `
            -RedirectStandardError $LogFile
        
        # Monitor process with timeout (2 hours max per agent)
        $timeout = 7200  # 2 hours in seconds
        $elapsed = 0
        $checkInterval = 30  # Check every 30 seconds
        
        while (-not $process.HasExited -and $elapsed -lt $timeout) {
            Start-Sleep -Seconds $checkInterval
            $elapsed += $checkInterval
            
            # Show progress every 2 minutes
            if (($elapsed % 120) -eq 0) {
                $minutes = [math]::Floor($elapsed / 60)
                Write-Host ">>> Still training... ($minutes minutes elapsed)" -ForegroundColor Gray
                if (Test-Path $LogFile) {
                    $lastLine = Get-Content $LogFile -Tail 1 -ErrorAction SilentlyContinue
                    if ($lastLine) {
                        Write-Host "   Last log: $lastLine" -ForegroundColor DarkGray
                    }
                }
            }
        }
        
        if (-not $process.HasExited) {
            Write-Error "xxx Training timed out after $([math]::Floor($timeout/60)) minutes. Process killed."
            $process.Kill()
            $process.WaitForExit(5000)
            exit 1
        }
        
        # Check exit code
        $exitCode = $process.ExitCode
        if ($exitCode -ne 0) {
            Write-Error "xxx Agent $AgentName Failed with exit code $exitCode! Check log: $LogFile"
            Write-Host "Last 20 lines of log:" -ForegroundColor Yellow
            if (Test-Path $LogFile) {
                Get-Content $LogFile -Tail 20 | Write-Host
            }
            exit 1
        }
        
        Write-Host ">>> Agent $AgentName Training Complete!" -ForegroundColor Green
    } catch {
        Write-Error "xxx Agent $AgentName Failed! Check log: $LogFile"
        Write-Error $_
        if (Test-Path $LogFile) {
            Write-Host "Last 20 lines of log:" -ForegroundColor Yellow
            Get-Content $LogFile -Tail 20 | Write-Host
        }
        exit 1
    }
}

# --- 1. VISION ---
Train-Agent -AgentName "Vision" `
            -ScriptPath "$ROOT_DIR\agents\vision\train_vision.py" `
            -DataDir "dataset" `
            -OutputDir "models" `
            -ExtraArgs "--epochs 30 --batch_size 4"

# --- 2. VIDEO ---
Train-Agent -AgentName "Video" `
            -ScriptPath "$ROOT_DIR\agents\video\train_resnet_pro.py" `
            -DataDir "dataset" `
            -OutputDir "models" `
            -ExtraArgs "--epochs 50 --batch_size 4"

# --- 3. AUDIO ---
Train-Agent -AgentName "Audio" `
            -ScriptPath "$ROOT_DIR\agents\audio\train_audio.py" `
            -DataDir "dataset" `
            -OutputDir "models" `
            -ExtraArgs "--epochs 6 --batch_size 2"

# --- 4. TEXT ---
Train-Agent -AgentName "Text" `
            -ScriptPath "$ROOT_DIR\agents\text\train_text.py" `
            -DataDir "dataset" `
            -OutputDir "models" `
            -ExtraArgs "--epochs 5 --batch_size 8"

Write-Host "`n>>> ALL AGENTS TRAINED SUCCESSFULLY! <<<" -ForegroundColor Green
