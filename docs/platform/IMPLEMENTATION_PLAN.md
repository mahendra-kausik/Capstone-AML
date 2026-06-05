# Implementation Plan (incremental)

## Done in this pass
- [x] Architecture + API + DB docs
- [x] FastAPI backend with ML serving (no retrain)
- [x] PostgreSQL models + init
- [x] JWT auth + RBAC skeleton
- [x] Next.js 15 frontend scaffold + pages
- [x] Docker Compose

## Phase A — Backend (complete first)
1. `app/ml/loader.py` — load checkpoints, scaler, drift JSON
2. `app/ml/inference.py` — predict, batch, risk score
3. `app/ml/shap_explainer.py` — KernelSHAP + fast top-feature fallback
4. Routes wired in `main.py`
5. Seed demo user + drift events on startup

## Phase B — Frontend
1. Auth context + API client
2. `/upload` CSV → batch-predict
3. `/analysis` single tx form
4. `/dashboard` cards from metrics + history
5. `/drift` Recharts tau timeline
6. `/explainability` SHAP bar chart

## Phase C — Hardening
1. Celery + Redis for async batch (optional)
2. React Flow graph from edges
3. Alembic migrations
4. Prometheus metrics
5. HTTPS + production secrets

## Demo workflow (Capstone)
1. Login `analyst@aml.local` / `demoaml2024`
2. Upload `samples/demo_transactions.csv`
3. View dashboard high-risk count
4. Open explainability for top risk tx
5. Show drift W3→W4 τ ≈ 0.15
