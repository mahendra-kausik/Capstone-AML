# Deployment Guide

## Prerequisites

- Docker & Docker Compose
- Research artifacts present: `models/static_gcn_best.pt`, `models/evolvegcn_best.pt`, `data/processed/scaler.pkl`

## Quick start

```bash
cd Capstone-AML
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/health |

## Demo login

| Email | Password | Role |
|-------|----------|------|
| analyst@aml.local | demoaml2024 | analyst |
| admin@aml.local | demoaml2024 | admin |

## Local backend (no Docker)

Uses **SQLite by default** (`backend/aml_platform.db`) — no Postgres setup required.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export RESEARCH_ROOT="$(cd .. && pwd)"
# Do NOT set DATABASE_URL unless you want Postgres (see below)
uvicorn app.main:app --reload --port 8000
```

### Optional: local PostgreSQL via Docker

```bash
cd Capstone-AML
docker compose up postgres -d
export DATABASE_URL=postgresql://aml:aml_secret@localhost:5432/aml_platform
```

**Troubleshooting:** If you see `FATAL: role "aml" does not exist`, your Mac’s Postgres is on port 5432 but was not created by Compose. Either unset `DATABASE_URL` (use SQLite) or start Compose Postgres as above.

If port 5432 is already taken by Homebrew Postgres, stop it (`brew services stop postgresql`) or map Compose to another port in `docker-compose.yml`.

## Local frontend

```bash
cd frontend
npm install
npm run dev
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `RESEARCH_ROOT` | Path to Capstone-AML repo root |
| `DATABASE_URL` | PostgreSQL connection string |
| `JWT_SECRET` | Signing key for JWT |
| `CORS_ORIGINS` | Comma-separated frontend origins |
| `NEXT_PUBLIC_API_URL` | Frontend → API base |

## Capstone demo flow

1. Login at http://localhost:3000
2. **Upload** → `samples/demo_transactions.csv`
3. **Dashboard** → high-risk count, model AUROC
4. **Analysis** → single transaction with 165 features
5. **Explainability** → SHAP top features
6. **Drift** → Kendall τ W3→W4, T43 F1 drop

## Production notes

- Replace `JWT_SECRET` and demo passwords
- Put TLS termination on reverse proxy (nginx)
- Scale backend horizontally; single ML worker recommended (GPU optional)
- Persist `postgres` volume; backup predictions tables
