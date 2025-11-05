# CareerLens 2.0 Monorepo

CareerLens 2.0 is a full-stack platform for personalized career and education guidance. This monorepo delivers production-ready scaffolding for the frontend experience, backend APIs, machine-learning workflows, and containerized infrastructure.

## Repository Layout

```
CareerLens/
├── backend/             # FastAPI service, database, alembic migrations, tests
├── frontend/            # React + Vite + Tailwind single-page application
├── ml/                  # Data science notebooks, pipelines, and artifacts
├── dataset/             # Source datasets placed alongside the repo
├── docker-compose.yml   # Orchestrates web, API, Postgres, and pgAdmin
├── Makefile             # Common developer automation targets
├── requirements.txt     # Shared Python dependencies (API + ML)
└── .env.example         # Environment template for secrets and configuration
```

## Quick Start

1. **Clone & Configure**
   ```bash
   copy .env.example .env
   ```
   Update the database credentials and allowed CORS origins as needed.

2. **Install Dependencies**
   ```bash
   make install
   ```
   Installs Python requirements and frontend node modules.

3. **Launch Dev Stack (Docker)**
   ```bash
   make dev
   ```
   - Frontend: http://localhost:5173
   - API: http://localhost:8000/docs
   - Postgres: localhost:5432 (`postgres/postgres`)
   - pgAdmin: http://localhost:5050 (admin@careerlens.local / admin)

4. **Database Migrations**
   ```bash
   make migrate
   ```
   Executes Alembic migrations against the running Postgres container.

5. **Run Quality Gates**
   ```bash
   make lint
   make format
   make test
   ```

## Key Technologies

- **Frontend**: React 18, Vite, TypeScript, TailwindCSS, React Router
- **Backend**: FastAPI, SQLAlchemy, Alembic, PostgreSQL, dependency-injected services
- **ML & Data**: Pandas, NumPy, scikit-learn, optional PyTorch, structured pipeline stubs
- **Tooling**: Docker Compose, Ruff, Black, Pytest, ESLint, Prettier

## Architecture Diagram (ASCII)

```
                ┌────────────────┐        REST        ┌────────────────────┐
                │  React/Vite    │◄─────────────────►│   FastAPI Service   │
                │  Tailwind UI   │   (JSON, Web)     │  (app/api, app/ml)  │
                └──────┬─────────┘                   └─────────┬──────────┘
                       │                                        │
                       │ GraphQL/RAG (future)                  │ SQLAlchemy ORM
                       │                                        │
                ┌───────▼────────┐                      ┌───────▼────────┐
                │  AI Assistant  │                      │  PostgreSQL    │
                │  (LLM Gateway) │                      │  + Alembic     │
                └───────┬────────┘                      └───────┬────────┘
                        │                                        │
                        │ Vector Search (Pinecone)               │
                        │                                        │
                ┌────────▼────────┐                     ┌────────▼────────┐
                │  ML Pipelines   │──────Artifacts──────│  ML Artifacts    │
                │  (notebooks,    │                     │  (ml/artifacts)  │
                │   pipelines)    │                     └──────────────────┘
                └─────────────────┘
```

## Backend Notes

- `app/core/config.py` centralizes environment-driven settings via Pydantic Settings.
- `app/services/recommendations.py` encapsulates recommendation logic; swap heuristics with ML outputs seamlessly.
- Database session management lives in `app/db/session.py`. Dependency overrides simplify testing.
- Alembic is preconfigured (`backend/alembic`) for schema migrations, with `alembic.ini` auto-wiring the settings.

## Frontend Notes

- Routes live under `src/pages` with dedicated screens for onboarding, recommendations, progress tracking, chat, and admin dashboards.
- `src/components/Layout.tsx` hosts the shell/navigation with a synthwave-inspired theme.
- API calls use a centralized Axios client (`src/lib/api.ts`) that respects the `VITE_API_BASE_URL` environment variable.
- ESLint + Prettier enforce consistent coding standards; Tailwind powers utility-first styling.

## Machine Learning Workspace

- Place exploratory notebooks in `ml/notebooks/` and version lightweight artifacts in `ml/artifacts/`.
- `ml/pipelines/preprocessing.py` provides helper utilities for dataset ingestion and cleaning.
- Keep large models outside of git and mount them at runtime via object storage or volume mounts.

## Makefile Targets

| Target    | Description                                    |
|-----------|------------------------------------------------|
| install   | Install Python and Node dependencies           |
| format    | Run Black, Ruff (fix mode), and frontend format |
| lint      | Run Ruff and ESLint                            |
| test      | Execute backend Pytest suite                   |
| migrate   | Apply Alembic migrations                       |
| dev       | Start full Docker stack (foreground)           |
| up        | Start Docker stack in detached mode            |
| down      | Stop and clean Docker resources                |

## Next Steps

- Implement real SQLAlchemy models and migrations for user profiles and recommendation feedback.
- Swap placeholder recommendation heuristics with the ML pipeline outputs via `/ml/artifacts`.
- Integrate authentication/authorization for admin dashboards and user personalization.
- Instrument observability (OpenTelemetry) and configure CI/CD pipelines for automated testing and deployment.
