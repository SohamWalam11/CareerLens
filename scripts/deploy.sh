#!/usr/bin/env bash
set -euo pipefail

# CareerLens one-command deploy script
# Supports: local docker-compose, Render, Google Cloud Run
# Usage:
#   ./scripts/deploy.sh local          # Local development
#   ./scripts/deploy.sh render TOKEN   # Deploy to Render
#   ./scripts/deploy.sh cloud-run      # Deploy to Cloud Run (requires gcloud CLI)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Local deployment using docker-compose
deploy_local() {
    log_info "Deploying CareerLens locally with docker-compose..."
    
    cd "$PROJECT_ROOT"
    
    # Check if .env exists, create from .env.example if not
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            log_warn ".env not found; creating from .env.example"
            cp .env.example .env
        else
            log_error ".env and .env.example not found"
            exit 1
        fi
    fi
    
    # Build images
    log_info "Building Docker images..."
    docker-compose build --no-cache
    
    # Start services
    log_info "Starting services..."
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    for i in {1..30}; do
        if docker-compose exec -T api curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            log_success "API is healthy"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    # Display endpoints
    log_success "Deployment complete!"
    echo -e "\n${YELLOW}Endpoints:${NC}"
    echo "  Frontend:  http://localhost:${FRONTEND_PORT:-5173}"
    echo "  API:       http://localhost:${API_PORT:-8000}"
    echo "  API Docs:  http://localhost:${API_PORT:-8000}/docs"
    echo "  pgAdmin:   http://localhost:${PGADMIN_PORT:-5050}"
    echo ""
    log_info "View logs with: docker-compose logs -f"
}

# Render deployment
deploy_render() {
    local token="${1:-}"
    
    if [ -z "$token" ]; then
        log_error "Render deployment requires auth token"
        echo "Usage: ./scripts/deploy.sh render <RENDER_AUTH_TOKEN>"
        exit 1
    fi
    
    log_info "Deploying to Render..."
    
    # Check for render CLI
    if ! command -v render &> /dev/null; then
        log_error "Render CLI not found. Install from: https://render.com/docs/deploy-from-git"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    log_info "Pushing to Render..."
    render deploy --auth-token="$token"
    
    log_success "Deployment to Render initiated!"
    log_info "Monitor progress at: https://dashboard.render.com"
}

# Google Cloud Run deployment
deploy_cloud_run() {
    log_info "Deploying to Google Cloud Run..."
    
    # Check for gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Get GCP project
    PROJECT_ID=$(gcloud config get-value project)
    if [ -z "$PROJECT_ID" ]; then
        log_error "No GCP project configured. Run: gcloud config set project PROJECT_ID"
        exit 1
    fi
    
    log_info "Using GCP project: $PROJECT_ID"
    
    # Build and push images
    log_info "Building and pushing API image to Container Registry..."
    gcloud builds submit \
        --tag=gcr.io/$PROJECT_ID/careerlens-api:latest \
        --config=cloudbuild.yaml \
        backend/
    
    log_info "Building and pushing Web image to Container Registry..."
    gcloud builds submit \
        --tag=gcr.io/$PROJECT_ID/careerlens-web:latest \
        --config=cloudbuild.yaml \
        frontend/
    
    # Deploy API service
    log_info "Deploying API service to Cloud Run..."
    gcloud run deploy careerlens-api \
        --image=gcr.io/$PROJECT_ID/careerlens-api:latest \
        --platform=managed \
        --region=us-central1 \
        --allow-unauthenticated \
        --set-env-vars="CORS_ORIGINS=https://careerlens-web.run.app"
    
    # Deploy Web service
    log_info "Deploying Web service to Cloud Run..."
    gcloud run deploy careerlens-web \
        --image=gcr.io/$PROJECT_ID/careerlens-web:latest \
        --platform=managed \
        --region=us-central1 \
        --allow-unauthenticated
    
    # Get service URLs
    API_URL=$(gcloud run services describe careerlens-api --platform=managed --region=us-central1 --format='value(status.url)')
    WEB_URL=$(gcloud run services describe careerlens-web --platform=managed --region=us-central1 --format='value(status.url)')
    
    log_success "Deployment to Cloud Run complete!"
    echo -e "\n${YELLOW}Services:${NC}"
    echo "  API:  $API_URL"
    echo "  Web:  $WEB_URL"
}

# Main entry point
main() {
    local target="${1:-local}"
    
    case "$target" in
        local)
            deploy_local
            ;;
        render)
            deploy_render "$2"
            ;;
        cloud-run|cloud_run)
            deploy_cloud_run
            ;;
        *)
            cat <<EOF
CareerLens Deploy Script

Usage: ./scripts/deploy.sh <TARGET> [OPTIONS]

Targets:
  local          Deploy locally with docker-compose (default)
  render TOKEN   Deploy to Render (requires auth token)
  cloud-run      Deploy to Google Cloud Run (requires gcloud CLI)

Examples:
  ./scripts/deploy.sh local
  ./scripts/deploy.sh render my-auth-token
  ./scripts/deploy.sh cloud-run

Environment Variables:
  FRONTEND_PORT   Frontend port (default: 5173)
  API_PORT        API port (default: 8000)
  DB_PORT         Database port (default: 5432)
EOF
            exit 0
            ;;
    esac
}

main "$@"
