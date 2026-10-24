#!/bin/bash
# ==============================================================================
# Supreme Court Judgment RAG — Cloud Run Jobs Production Deployment Script
# Target: 1950-2026 Judgment PDFs (~43,000 files in GCS)
# Sharding: 77 Parallel Cloud Run Tasks (1 task per year)
# Author: Kedar (Data Ingestion & Pipeline Engineer)
# ==============================================================================

set -euo pipefail

# 1. Configuration Defaults
PROJECT_ID="${GCP_PROJECT:-sc-judgments-rag}"
REGION="${GCP_REGION:-asia-south1}"
BUCKET_NAME="${GCS_BUCKET:-sc-judgments-rag-prod-data}"
JOB_NAME="scj-ingest-1950-2026"
REPO_NAME="scj-rag-repo"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/scj-pipeline:latest"
SERVICE_ACCOUNT="scj-ingestion-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "================================================================================"
echo " SUPREME COURT RAG — CLOUD RUN JOBS DEPLOYMENT"
echo "================================================================================"
echo " Project:         ${PROJECT_ID}"
echo " Region:          ${REGION}"
echo " Bucket:          gs://${BUCKET_NAME}"
echo " Job Name:        ${JOB_NAME}"
echo " Image:           ${IMAGE_TAG}"
echo " Service Account: ${SERVICE_ACCOUNT}"
echo " Target Range:    1950 - 2026 (77 Tasks in Parallel)"
echo "================================================================================"

# 2. Set GCP Active Project
echo "[Step 1/5] Setting active project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

# 3. Enable Required GCP APIs
echo "[Step 2/5] Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    sqladmin.googleapis.com \
    secretmanager.googleapis.com \
    storage.googleapis.com

# 4. Create Artifact Registry Docker Repository (if not existing)
echo "[Step 3/5] Verifying Artifact Registry repository: ${REPO_NAME}..."
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" >/dev/null 2>&1; then
    echo "Creating Artifact Registry repository '${REPO_NAME}' in ${REGION}..."
    gcloud artifacts repositories create "${REPO_NAME}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="Docker repository for Supreme Court RAG pipeline images"
fi

# 5. Build and Push Container Image using Cloud Build
echo "[Step 4/5] Building and pushing container image via Cloud Build..."
gcloud builds submit --tag "${IMAGE_TAG}" .

# 6. Create or Update Cloud Run Job
echo "[Step 5/5] Deploying Cloud Run Job '${JOB_NAME}'..."
if gcloud run jobs describe "${JOB_NAME}" --region="${REGION}" >/dev/null 2>&1; then
    echo "Updating existing Cloud Run Job '${JOB_NAME}'..."
    ACTION="update"
else
    echo "Creating new Cloud Run Job '${JOB_NAME}'..."
    ACTION="create"
fi

# Deploy with 77 tasks (1 task per year from 1950 to 2026)
gcloud run jobs ${ACTION} "${JOB_NAME}" \
    --image="${IMAGE_TAG}" \
    --region="${REGION}" \
    --tasks=77 \
    --max-retries=3 \
    --task-timeout=24h \
    --cpu=4 \
    --memory=8Gi \
    --service-account="${SERVICE_ACCOUNT}" \
    --set-env-vars="GCP_PROJECT=${PROJECT_ID},GCS_BUCKET=${BUCKET_NAME},GCP_REGION=${REGION},CONCURRENT_WORKERS=4" \
    --set-secrets="DATABASE_URL=scj-database-url:latest" \
    --args="--start-year,1950,--end-year,2026,--mode,full"

echo "================================================================================"
echo " DEPLOYMENT COMPLETE! "
echo "================================================================================"
echo "To execute the 77-task batch job across all years (1950-2026), run:"
echo ""
echo "    gcloud run jobs execute ${JOB_NAME} --region=${REGION}"
echo ""
echo "To monitor execution:"
echo ""
echo "    gcloud run jobs executions list --job=${JOB_NAME} --region=${REGION}"
echo "================================================================================"

