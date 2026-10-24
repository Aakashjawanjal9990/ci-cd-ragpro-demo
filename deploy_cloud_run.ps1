# ==============================================================================
# Supreme Court Judgment RAG — Cloud Run Jobs Production Deployment Script (PowerShell)
# Target: 1950-2026 Judgment PDFs (~43,000 files in GCS)
# Sharding: 77 Parallel Cloud Run Tasks (1 task per year)
# Author: Kedar (Data Ingestion & Pipeline Engineer)
# ==============================================================================

param(
    [string]$ProjectId = $env:GCP_PROJECT,
    [string]$Region = "asia-south1",
    [string]$BucketName = "sc-judgments-rag-prod-data",
    [string]$JobName = "scj-ingest-1950-2026",
    [string]$RepoName = "scj-rag-repo"
)

if (-not $ProjectId) {
    $ProjectId = "sc-judgments-rag"
}

$ImageTag = "${Region}-docker.pkg.dev/${ProjectId}/${RepoName}/scj-pipeline:latest"
$ServiceAccount = "scj-ingestion-sa@${ProjectId}.iam.gserviceaccount.com"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " SUPREME COURT RAG — CLOUD RUN JOBS DEPLOYMENT (POWERSHELL)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " Project:         $ProjectId"
Write-Host " Region:          $Region"
Write-Host " Bucket:          gs://$BucketName"
Write-Host " Job Name:        $JobName"
Write-Host " Image:           $ImageTag"
Write-Host " Service Account: $ServiceAccount"
Write-Host " Target Range:    1950 - 2026 (77 Tasks in Parallel)"
Write-Host "================================================================================"

# 1. Set GCP Active Project
Write-Host "`n[Step 1/5] Setting active project to $ProjectId..." -ForegroundColor Yellow
gcloud config set project $ProjectId

# 2. Enable Required GCP APIs
Write-Host "`n[Step 2/5] Enabling required GCP APIs..." -ForegroundColor Yellow
gcloud services enable `
    run.googleapis.com `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    aiplatform.googleapis.com `
    sqladmin.googleapis.com `
    secretmanager.googleapis.com `
    storage.googleapis.com

# 3. Create Artifact Registry Docker Repository (if needed)
Write-Host "`n[Step 3/5] Verifying Artifact Registry repository: $RepoName..." -ForegroundColor Yellow
$repoExists = gcloud artifacts repositories describe $RepoName --location=$Region 2>$null
if (-not $repoExists) {
    Write-Host "Creating Artifact Registry repository '$RepoName' in $Region..."
    gcloud artifacts repositories create $RepoName `
        --repository-format=docker `
        --location=$Region `
        --description="Docker repository for Supreme Court RAG pipeline images"
}

# 4. Build and Push Container Image using Cloud Build
Write-Host "`n[Step 4/5] Building and pushing container image via Cloud Build..." -ForegroundColor Yellow
gcloud builds submit --tag $ImageTag .

# 5. Create or Update Cloud Run Job
Write-Host "`n[Step 5/5] Deploying Cloud Run Job '$JobName' with 77 tasks..." -ForegroundColor Yellow
$jobExists = gcloud run jobs describe $JobName --region=$Region 2>$null
$action = if ($jobExists) { "update" } else { "create" }

gcloud run jobs $action $JobName `
    --image=$ImageTag `
    --region=$Region `
    --tasks=77 `
    --max-retries=3 `
    --task-timeout=24h `
    --cpu=4 `
    --memory=8Gi `
    --service-account=$ServiceAccount `
    --set-env-vars="GCP_PROJECT=$ProjectId,GCS_BUCKET=$BucketName,GCP_REGION=$Region,CONCURRENT_WORKERS=4" `
    --set-secrets="DATABASE_URL=scj-database-url:latest" `
    --args="--start-year,1950,--end-year,2026,--mode,full"

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host " DEPLOYMENT COMPLETE! " -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "To execute the 77-task batch job across all years (1950-2026), run:" -ForegroundColor Cyan
Write-Host "    gcloud run jobs execute $JobName --region=$Region`n" -ForegroundColor White
Write-Host "To monitor execution:" -ForegroundColor Cyan
Write-Host "    gcloud run jobs executions list --job=$JobName --region=$Region" -ForegroundColor White
Write-Host "================================================================================"

