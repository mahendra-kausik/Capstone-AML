# AML Intelligence Platform — Launch Checklist

Use this checklist before investor demos, enterprise pilots, or production deployment.

---

## Security

- [ ] Rotate `JWT_SECRET` — generate with `openssl rand -hex 32`
- [ ] Rotate Postgres password — update `DATABASE_URL` and compose env
- [ ] Set `SEED_DEMO_USERS=false` in production
- [ ] Set `DEBUG=false` in production (disables OpenAPI `/docs`)
- [ ] Configure CORS to production domain only
- [ ] Enable TLS via nginx with valid certificates
- [ ] Remove demo credential pre-fill from login page (production build)
- [ ] Review audit logs for failed login attempts
- [ ] Confirm rate limits are appropriate for expected load
- [ ] Do not commit `.env` files — use secrets manager in cloud

---

## Performance

- [ ] Run `npm run build` — verify no build errors
- [ ] Run load test on `/api/v1/predict` (target: 60 req/min per IP)
- [ ] Verify batch upload handles 100+ row CSV within SLA
- [ ] Confirm Postgres connection pool settings for expected concurrency
- [ ] Enable nginx gzip compression
- [ ] Set frontend `output: standalone` and use optimized Docker image

---

## Scalability

- [ ] Deploy Postgres with persistent volume (not SQLite)
- [ ] Run Alembic migrations: `alembic upgrade head`
- [ ] Consider gunicorn with multiple uvicorn workers for backend
- [ ] Plan async job queue (Celery/RQ) for large batch uploads
- [ ] Monitor disk usage for prediction/audit log growth

---

## Monitoring

- [ ] Verify `/health` returns DB + ML status
- [ ] Configure log aggregation (CloudWatch, Datadog, etc.)
- [ ] Set up uptime monitoring on `/health`
- [ ] Alert on 5xx error rate > 1%
- [ ] Track prediction latency p95
- [ ] Monitor drift event alerts

---

## Backups

- [ ] Schedule daily `pg_dump` of `aml_platform` database
- [ ] Backup ML artifacts (`models/`, `data/processed/scaler.pkl`)
- [ ] Document restore procedure
- [ ] Test restore on staging environment

---

## Documentation

- [ ] `PRODUCTION_READINESS_AUDIT.md` reviewed
- [ ] `docs/platform/DEPLOYMENT.md` updated with production steps
- [ ] API docs available at `/docs` (dev only) or exported OpenAPI JSON
- [ ] Demo credentials documented for investors
- [ ] Postman collection generated from OpenAPI

---

## Deployment

### Local Demo
```bash
docker compose up --build
# Open http://localhost:3000
# Login: analyst@example.com / demoaml2024
```

### Production (Docker + nginx)
```bash
cp .env.example .env
# Edit .env with production secrets
docker compose -f docker-compose.prod.yml up -d --build
alembic -c backend/alembic.ini upgrade head
```

### Cloud (AWS/Azure/GCP/Render/Railway)
See `docs/platform/DEPLOYMENT.md` for provider-specific instructions.

---

## Investor Demo

- [ ] Demo accounts working: admin, analyst, viewer @example.com
- [ ] Sample CSV uploaded: `samples/demo_transactions.csv`
- [ ] Preloaded cases visible in Investigations
- [ ] Alerts visible in Alerts Center + notification bell
- [ ] SHAP explainability demo on transaction detail
- [ ] Drift monitoring charts populated
- [ ] Network graph renders for seed transactions
- [ ] Report export downloads successfully
- [ ] No console errors on key pages

### Demo Script (15 min)
1. **Login** as analyst — show role-based dashboard
2. **Upload** demo CSV — show batch scoring results
3. **Transactions** — filter high-risk, open detail
4. **Explainability** — SHAP waterfall + narrative
5. **Network** — graph exploration with risk highlighting
6. **Alerts** — show auto-generated high-risk alerts
7. **Investigations** — create case, add note, assign, close
8. **Drift** — Kendall τ monitoring, model comparison
9. **Reports** — export compliance CSV
10. **Admin** — audit log, user management

---

## Pre-Launch Sign-Off

| Area | Owner | Date | Status |
|------|-------|------|--------|
| Security review | | | ☐ |
| Functional QA (all 10 flows) | | | ☐ |
| Performance baseline | | | ☐ |
| Backup verified | | | ☐ |
| Demo rehearsal | | | ☐ |
| Documentation complete | | | ☐ |

---

## Production Readiness Score

**Current: 58/100** — Demo-ready, pilot-ready with P0 security fixes

**Target for enterprise pilot: 75/100**

**Target for production GA: 90/100**
