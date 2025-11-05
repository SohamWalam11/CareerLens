# CareerLens Deployment Guide

One-command deploy scripts for local development and cloud production environments.

## Quick Start

### Local Development

```bash
# macOS/Linux
chmod +x scripts/deploy.sh
./scripts/deploy.sh local

# Windows
./scripts/deploy.ps1 local
```

Endpoints:
- Frontend: http://localhost:5173
- API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs
- pgAdmin: http://localhost:5050

### View Logs

```bash
docker-compose logs -f api
docker-compose logs -f frontend
```

### Stop Services

```bash
docker-compose down
```

---

## Cloud Deployments

### Render.com (Recommended for simplicity)

1. Create account at [render.com](https://render.com)
2. Get auth token from dashboard
3. Deploy:

```bash
# macOS/Linux
./scripts/deploy.sh render YOUR_RENDER_AUTH_TOKEN

# Windows
./scripts/deploy.ps1 render YOUR_RENDER_AUTH_TOKEN
```

**Render.yaml** defines:
- API service (Python 3.11 + FastAPI)
- Web service (Node.js 20 + SPA)
- PostgreSQL 16 database
- Automatic deployments on git push

### Google Cloud Run

#### Prerequisites

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set default project
gcloud config set project YOUR_PROJECT_ID
```

#### Deploy

```bash
# macOS/Linux
./scripts/deploy.sh cloud-run

# Windows
./scripts/deploy.ps1 cloud-run
```

**What this does:**
1. Builds Docker images for API and Web
2. Pushes to Google Container Registry
3. Deploys API service to Cloud Run (auto-scales 1–10 instances)
4. Deploys Web service to Cloud Run (auto-scales 1–5 instances)
5. Configures Cloud SQL for database

**Costs:**
- API & Web: ~$0.40/month (within free tier for light usage)
- Cloud SQL: ~$10/month (smallest instance)

---

## Environment Variables

All services use environment variables from `.env` (local) or secrets management (cloud).

### Local (.env)

```env
# Database
DATABASE_URL=postgresql+psycopg2://postgres:postgres@db:5432/careerlens
POSTGRES_PASSWORD=postgres

# Frontend
VITE_API_BASE_URL=http://localhost:8000/api/v1
FRONTEND_PORT=5173

# API
API_PORT=8000
CORS_ORIGINS=http://localhost:5173

# Pinecone (optional)
PINECONE_API_KEY=
PINECONE_INDEX_NAME=careerlens
PINECONE_NAMESPACE=recommendations
```

### Render.com

Secrets managed via Render dashboard. Set `PINECONE_API_KEY` and database credentials there.

### Cloud Run

Secrets stored in Google Secret Manager. Connect via `gcloud secrets create` before deploying.

---

## Health Checks

All services expose `/health` (or `/api/v1/health` for API):

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check frontend
curl http://localhost:5173/
```

Docker Compose waits for these health endpoints before marking services as ready.

---

## Multi-Stage Builds (Optimization)

### Backend Dockerfile
- **Builder stage**: Installs dependencies and builds wheels
- **Runtime stage**: Copies only wheels, reducing image size from ~1GB → ~400MB
- **Health check**: Integrated liveness/readiness probe

### Frontend Dockerfile
- **Build stage**: Compiles TypeScript + React with npm
- **Runtime stage**: Serves static dist/ with nginx (~100MB)
- **SPA routing**: Nginx configured to route all paths to index.html
- **Cache headers**: 1-year expiry for versioned assets

---

## Troubleshooting

### Docker build fails

```bash
# Clear cache and rebuild
docker-compose build --no-cache --force-rm

# Check image sizes
docker images

# View build logs
docker-compose build --verbose
```

### API won't start

```bash
# Check logs
docker-compose logs api

# Common issues:
# - Port 8000 already in use: change API_PORT in .env
# - Database not ready: wait 10s and retry
# - Missing env vars: check .env file
```

### Frontend shows blank page

```bash
# Clear browser cache (Ctrl+Shift+Del)
# Check browser console (F12)
# Verify VITE_API_BASE_URL points to running API

docker-compose logs frontend
```

### Database connection error

```bash
# Wait for postgres to start (takes ~5s)
# Check pg logs
docker-compose logs db

# Reset database
docker-compose down -v          # Remove volumes
docker-compose up db -d         # Start fresh
```

---

## Scaling & Performance

### Local (docker-compose)

- Suitable for development and testing
- Single instance of each service
- Database stored in named volume (`pg_data`)

### Production (Render)

- Auto-scales based on CPU/memory
- Free SSL certificate for custom domains
- Native git deployments
- Database backups included

### Production (Cloud Run)

- Scales to zero (saves costs)
- Pay only for execution time
- Optional Cloud SQL Proxy for database
- Global edge network via Cloud Load Balancer

---

## Monitoring & Logs

### Local

```bash
# Real-time logs for all services
docker-compose logs -f

# Logs for specific service
docker-compose logs -f api
```

### Render

Dashboard → Services → Logs tab

### Cloud Run

```bash
# Stream logs
gcloud run services logs read careerlens-api --limit 50 --follow

# View specific revision
gcloud run revisions list --service careerlens-api
gcloud run revisions logs read <REVISION_ID>
```

---

## Database Migrations

```bash
# In docker container (local)
docker-compose exec api alembic upgrade head

# Or manually via alembic CLI
cd backend
alembic upgrade head
```

---

## Cleanup

### Local

```bash
# Stop services, keep volumes
docker-compose stop

# Stop and remove containers, keep volumes
docker-compose down

# Full cleanup (removes volumes)
docker-compose down -v
```

### Render

Dashboard → Service Settings → Delete Service

### Cloud Run

```bash
# Delete API service
gcloud run services delete careerlens-api --region=us-central1

# Delete Web service
gcloud run services delete careerlens-web --region=us-central1

# Delete images
gcloud container images delete gcr.io/$PROJECT_ID/careerlens-api --quiet
gcloud container images delete gcr.io/$PROJECT_ID/careerlens-web --quiet
```

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs <service>`
2. Review `.env` configuration
3. Verify all prerequisites are installed (Docker, Node, Python)
4. Check GitHub Issues
