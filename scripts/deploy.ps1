#!/usr/bin/env pwsh
<#
.SYNOPSIS
CareerLens one-command deploy script for Windows.

.DESCRIPTION
Supports local docker-compose, Render, and Google Cloud Run deployments.

.PARAMETER Target
Deployment target: local, render, or cloud-run

.PARAMETER AuthToken
Authentication token for Render deployment

.EXAMPLE
./scripts/deploy.ps1 local
./scripts/deploy.ps1 render my-auth-token
./scripts/deploy.ps1 cloud-run
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('local', 'render', 'cloud-run', 'cloud_run')]
    [string]$Target = 'local',
    
    [Parameter(Position = 1)]
    [string]$AuthToken = ''
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red; exit 1 }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }

function Test-HealthEndpoint {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Deploy-Local {
    Write-Info "Deploying CareerLens locally with docker-compose..."
    
    Push-Location $ProjectRoot
    try {
        # Check for .env
        if (-not (Test-Path .env)) {
            if (Test-Path .env.example) {
                Write-Warn ".env not found; creating from .env.example"
                Copy-Item .env.example .env
            }
            else {
                Write-Error ".env and .env.example not found"
            }
        }
        
        # Build images
        Write-Info "Building Docker images..."
        & docker-compose build --no-cache
        if ($LASTEXITCODE -ne 0) { Write-Error "Docker build failed" }
        
        # Start services
        Write-Info "Starting services..."
        & docker-compose up -d
        if ($LASTEXITCODE -ne 0) { Write-Error "Docker-compose up failed" }
        
        # Wait for health
        Write-Info "Waiting for services to be healthy..."
        $maxRetries = 30
        $retries = 0
        while ($retries -lt $maxRetries) {
            if (Test-HealthEndpoint "http://localhost:8000/api/v1/health") {
                Write-Success "API is healthy"
                break
            }
            Write-Host "." -NoNewline
            Start-Sleep -Seconds 2
            $retries++
        }
        
        if ($retries -eq $maxRetries) {
            Write-Warn "Health check timed out after 60 seconds"
        }
        
        Write-Success "Deployment complete!"
        Write-Host ""
        Write-Host "Endpoints:" -ForegroundColor Yellow
        Write-Host "  Frontend:  http://localhost:5173"
        Write-Host "  API:       http://localhost:8000"
        Write-Host "  API Docs:  http://localhost:8000/docs"
        Write-Host "  pgAdmin:   http://localhost:5050"
        Write-Host ""
        Write-Info "View logs with: docker-compose logs -f"
    }
    finally {
        Pop-Location
    }
}

function Deploy-Render {
    param([string]$Token)
    
    if ([string]::IsNullOrEmpty($Token)) {
        Write-Error "Render deployment requires auth token. Usage: ./scripts/deploy.ps1 render <RENDER_AUTH_TOKEN>"
    }
    
    Write-Info "Deploying to Render..."
    
    # Check for render CLI
    if (-not (Get-Command render -ErrorAction SilentlyContinue)) {
        Write-Error "Render CLI not found. Install from: https://render.com/docs/deploy-from-git"
    }
    
    Push-Location $ProjectRoot
    try {
        Write-Info "Pushing to Render..."
        & render deploy --auth-token=$Token
        
        Write-Success "Deployment to Render initiated!"
        Write-Info "Monitor progress at: https://dashboard.render.com"
    }
    finally {
        Pop-Location
    }
}

function Deploy-CloudRun {
    Write-Info "Deploying to Google Cloud Run..."
    
    # Check for gcloud CLI
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        Write-Error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
    }
    
    Push-Location $ProjectRoot
    try {
        # Get GCP project
        $projectId = & gcloud config get-value project 2>$null
        if ([string]::IsNullOrEmpty($projectId)) {
            Write-Error "No GCP project configured. Run: gcloud config set project PROJECT_ID"
        }
        
        Write-Info "Using GCP project: $projectId"
        
        Write-Info "Building and pushing API image..."
        & gcloud builds submit `
            --tag="gcr.io/$projectId/careerlens-api:latest" `
            backend/
        
        Write-Info "Building and pushing Web image..."
        & gcloud builds submit `
            --tag="gcr.io/$projectId/careerlens-web:latest" `
            frontend/
        
        Write-Info "Deploying API service to Cloud Run..."
        & gcloud run deploy careerlens-api `
            --image="gcr.io/$projectId/careerlens-api:latest" `
            --platform=managed `
            --region=us-central1 `
            --allow-unauthenticated `
            --set-env-vars="CORS_ORIGINS=https://careerlens-web.run.app"
        
        Write-Info "Deploying Web service to Cloud Run..."
        & gcloud run deploy careerlens-web `
            --image="gcr.io/$projectId/careerlens-web:latest" `
            --platform=managed `
            --region=us-central1 `
            --allow-unauthenticated
        
        # Get URLs
        $apiUrl = & gcloud run services describe careerlens-api --platform=managed --region=us-central1 --format='value(status.url)'
        $webUrl = & gcloud run services describe careerlens-web --platform=managed --region=us-central1 --format='value(status.url)'
        
        Write-Success "Deployment to Cloud Run complete!"
        Write-Host ""
        Write-Host "Services:" -ForegroundColor Yellow
        Write-Host "  API:  $apiUrl"
        Write-Host "  Web:  $webUrl"
    }
    finally {
        Pop-Location
    }
}

# Main
switch ($Target) {
    'local' { Deploy-Local }
    'render' { Deploy-Render $AuthToken }
    { $_ -in @('cloud-run', 'cloud_run') } { Deploy-CloudRun }
    default {
        Write-Host @"
CareerLens Deploy Script

Usage: ./scripts/deploy.ps1 [-Target] <TARGET> [-AuthToken] <TOKEN>

Targets:
  local          Deploy locally with docker-compose (default)
  render         Deploy to Render (requires auth token)
  cloud-run      Deploy to Google Cloud Run (requires gcloud CLI)

Examples:
  ./scripts/deploy.ps1 local
  ./scripts/deploy.ps1 render my-auth-token
  ./scripts/deploy.ps1 cloud-run

Environment Variables:
  FRONTEND_PORT   Frontend port (default: 5173)
  API_PORT        API port (default: 8000)
  DB_PORT         Database port (default: 5432)
"@
    }
}
