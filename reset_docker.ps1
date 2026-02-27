# ============================================================================
# DEEPGUARD PRO - COMPLETE DOCKER RESET SCRIPT
# ============================================================================

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  DEEPGUARD PRO - RESET COMPLET DOCKER" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 1: STOP & CLEANUP
# ============================================================================
Write-Host "[1/11] Oprire containere si stergere volume..." -ForegroundColor Yellow
docker-compose down -v --rmi local --remove-orphans

Write-Host ""
Write-Host "[2/11] Curatare cache build Docker..." -ForegroundColor Yellow
docker builder prune -f

Write-Host ""
Write-Host "[3/11] Curatare imagini neutilizate..." -ForegroundColor Yellow
docker image prune -f

# ============================================================================
# STEP 2: DELETE CORRUPTED LOCAL DATA
# ============================================================================
Write-Host ""
Write-Host "[4/11] Stergere directoare locale corupte..." -ForegroundColor Yellow

$dirsToClean = @(
    "data\neo4j",
    "seaweedfs",
    "data\evidence"
)

foreach ($dir in $dirsToClean) {
    if (Test-Path $dir) {
        Write-Host "  -> Stergere: $dir" -ForegroundColor Red
        Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
    }
}

# ============================================================================
# STEP 3: RECREATE DIRECTORIES
# ============================================================================
Write-Host ""
Write-Host "[5/11] Recreare directoare fresh..." -ForegroundColor Yellow

$dirsToCreate = @(
    "data\neo4j",
    "data\evidence",
    "seaweedfs"
)

foreach ($dir in $dirsToCreate) {
    Write-Host "  -> Creare: $dir" -ForegroundColor Green
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# ============================================================================
# STEP 4: FIX TEXT AGENT REQUIREMENTS
# ============================================================================
Write-Host ""
Write-Host "[6/11] Fix Text Agent - Add 'requests' library..." -ForegroundColor Yellow

$textReqPath = "agents\text\requirements.txt"

if (Test-Path $textReqPath) {
    $content = Get-Content $textReqPath -Raw
    if ($content -notmatch "requests") {
        Write-Host "  -> Adaugare 'requests' in requirements.txt" -ForegroundColor Green
        Add-Content $textReqPath "`nrequests"
    } else {
        Write-Host "  -> 'requests' exista deja" -ForegroundColor Green
    }
} else {
    Write-Host "  -> Creare requirements.txt nou" -ForegroundColor Green
    @"
fastapi
uvicorn[standard]
requests
python-multipart
"@ | Out-File -Encoding utf8 $textReqPath
}

# ============================================================================
# STEP 5: REBUILD IMAGES
# ============================================================================
Write-Host ""
Write-Host "[7/11] Rebuild imagini Docker (poate dura 5-10 min)..." -ForegroundColor Yellow
Write-Host "  Te rog asteapta..." -ForegroundColor DarkGray

docker-compose build --no-cache --parallel 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "  -> Build complet!" -ForegroundColor Green
} else {
    Write-Host "  -> Build cu warning-uri (continuum)" -ForegroundColor Yellow
}

# ============================================================================
# STEP 6: START CORE SERVICES FIRST
# ============================================================================
Write-Host ""
Write-Host "[8/11] Pornire servicii CORE (Neo4j, Redis, S3)..." -ForegroundColor Green

docker-compose up -d neo4j redis seaweedfs mailhog

Write-Host ""
Write-Host "[9/11] Asteptare initializare Neo4j (40 secunde)..." -ForegroundColor Yellow

for ($i = 40; $i -gt 0; $i--) {
    Write-Host "`r  Countdown: $i secunde ramase..." -NoNewline -ForegroundColor DarkGray
    Start-Sleep -Seconds 1
}
Write-Host ""

# ============================================================================
# STEP 7: TEST NEO4J
# ============================================================================
Write-Host ""
Write-Host "[10/11] Test Neo4j connection..." -ForegroundColor Green

$neo4jTest = docker exec deepguard_neo4j cypher-shell -u neo4j -p deepguard_secret "RETURN 1 as test" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "  -> Neo4j ONLINE si functional!" -ForegroundColor Green
} else {
    Write-Host "  -> Neo4j nu raspunde inca (se va stabili)" -ForegroundColor Yellow
    Write-Host "  Ultimele logs:" -ForegroundColor DarkGray
    docker logs deepguard_neo4j --tail 10 | Write-Host -ForegroundColor DarkGray
}

# ============================================================================
# STEP 8: START ALL SERVICES
# ============================================================================
Write-Host ""
Write-Host "[11/11] Pornire TOATE serviciile..." -ForegroundColor Green

docker-compose up -d

Write-Host ""
Write-Host "Asteptare stabilizare finala (20 secunde)..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

# ============================================================================
# FINAL STATUS REPORT
# ============================================================================
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  STATUS FINAL - CONTAINERE" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  HEALTH CHECKS" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$allContainers = docker ps --format "{{.Names}}\t{{.Status}}"
$healthyCount = 0
$unhealthyCount = 0
$restartingCount = 0

foreach ($line in $allContainers) {
    if ($line -match "healthy") {
        Write-Host "[OK] $line" -ForegroundColor Green
        $healthyCount++
    } elseif ($line -match "unhealthy") {
        Write-Host "[FAIL] $line" -ForegroundColor Red
        $unhealthyCount++
    } elseif ($line -match "Restarting") {
        Write-Host "[WARN] $line" -ForegroundColor Yellow
        $restartingCount++
    } else {
        Write-Host "[UP] $line" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  SERVICE URLS" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Frontend:  http://localhost:8501" -ForegroundColor White
Write-Host "  API:       http://localhost:8000/health" -ForegroundColor White
Write-Host "  Neo4j:     http://localhost:7474" -ForegroundColor White
Write-Host "             (user: neo4j, pass: deepguard_secret)" -ForegroundColor DarkGray
Write-Host "  Flower:    http://localhost:5555" -ForegroundColor White
Write-Host "             (user: admin, pass: deepguard123)" -ForegroundColor DarkGray
Write-Host "  MailHog:   http://localhost:8025" -ForegroundColor White
Write-Host ""

# ============================================================================
# API HEALTH TEST
# ============================================================================
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  API HEALTH TEST" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 5

try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10
    
    Write-Host "  -> API ONLINE!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Response:" -ForegroundColor White
    $health | ConvertTo-Json -Depth 3 | Write-Host -ForegroundColor Cyan
    
} catch {
    Write-Host "  -> API nu raspunde inca" -ForegroundColor Yellow
    Write-Host "  Asteapta 30-60 secunde si reincearca:" -ForegroundColor DarkGray
    Write-Host "     curl http://localhost:8000/health" -ForegroundColor DarkGray
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  RESET COMPLET!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

if ($unhealthyCount -gt 0 -or $restartingCount -gt 0) {
    Write-Host "  Containere cu probleme: $unhealthyCount unhealthy, $restartingCount restarting" -ForegroundColor Yellow
    Write-Host "  Asteapta 1-2 minute pentru health checks" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Verifica logs pentru debugging:" -ForegroundColor White
    Write-Host "     docker logs deepguard_brain" -ForegroundColor DarkGray
    Write-Host "     docker logs deepguard_neo4j" -ForegroundColor DarkGray
    Write-Host "     docker logs deepguard_celery_worker" -ForegroundColor DarkGray
} else {
    Write-Host "  Toate serviciile functioneaza!" -ForegroundColor Green
    Write-Host "  Sistemul este gata de utilizare!" -ForegroundColor Green
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""