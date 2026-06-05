# AML Intelligence Platform — Production Readiness Audit

**Date:** June 2026  
**Version:** 1.0.0  
**Auditor scope:** Full-stack (Frontend, Backend, Database, API, Auth, Security, Docker, Deployment)

---

## Executive Summary

The AML Intelligence Platform is a **functional research-to-product bridge** with real GCN inference, SHAP explainability, drift monitoring, and investigation workflows. It is suitable for **investor demos and capstone evaluation** but requires hardening before enterprise pilot deployment.

| Dimension | Score | Status |
|-----------|-------|--------|
| ML Integration | 85/100 | Real models, SHAP, drift artifacts |
| API Completeness | 75/100 | 22+ endpoints, core flows covered |
| Frontend UX | 70/100 | Polished UI, some mock widgets |
| Database | 55/100 | PostgreSQL ready, migrations added |
| Authentication | 50/100 | JWT + RBAC, refresh tokens added |
| Security | 45/100 | Rate limits partial, secrets need rotation |
| Observability | 40/100 | Structured logging added, metrics partial |
| Deployment | 55/100 | Docker Compose, nginx config added |
| Enterprise AML | 35/100 | No sanctions screening, no multi-tenant |
| **Overall Production Readiness** | **58/100** | Demo-ready; pilot requires P0 fixes |

---

## Critical Findings

| ID | Finding | Impact | Status |
|----|---------|--------|--------|
| C1 | Default JWT secret in config | Token forgery risk | ⚠️ Must rotate in production |
| C2 | Demo credentials auto-seeded | Known accounts in every deploy | ⚠️ Gate with `SEED_DEMO_USERS` |
| C3 | No Alembic migrations (was `create_all` only) | Schema drift in production | ✅ Fixed — Alembic added |
| C4 | Reports page was 100% placeholder | Broken compliance workflow | ✅ Fixed — Reports API |
| C5 | No alert entity — client-side filter only | No alert history/SLA | ✅ Fixed — Alerts table + API |
| C6 | Investigations missing case detail/notes UI | Broken case workflow | ✅ Fixed — Case detail page |
| C7 | Dashboard API limit=500 exceeded backend max 200 | Runtime 422 errors | ✅ Fixed |
| C8 | Next.js 15.5 devtools HMR corruption | Dev server crashes | ✅ Fixed — devtoolSegmentExplorer disabled |
| C9 | No structured logging | Blind in production | ✅ Fixed |
| C10 | Shallow health check | False healthy pods | ✅ Fixed — deep health check |

---

## High Priority Findings

| ID | Finding | File(s) | Status |
|----|---------|---------|--------|
| H1 | Drift data is static research JSON, not live | `backend/app/services/drift_service.py` | Open |
| H2 | Single `/predict` ignores graph edges | `backend/app/services/prediction_service.py` | Open |
| H3 | No multi-tenant org isolation | `backend/app/routes/history.py` | Partial — org model added |
| H4 | JWT in localStorage (XSS risk) | `frontend/lib/api.ts` | Open — httpOnly cookies recommended |
| H5 | No login rate limiting | `backend/app/routes/auth.py` | ✅ Fixed |
| H6 | uploadCsv API base URL inconsistent | `frontend/lib/api.ts` | ✅ Fixed |
| H7 | Silent API failures on most pages | `frontend/app/(platform)/*/page.tsx` | ✅ Fixed — error states |
| H8 | Docker secrets hardcoded in compose | `docker-compose.yml` | ⚠️ Use `.env` in production |
| H9 | Frontend Dockerfile ignores standalone output | `frontend/Dockerfile` | ✅ Fixed |
| H10 | Documentation email mismatch (`@aml.local` vs `@example.com`) | `docs/platform/DEPLOYMENT.md` | ✅ Fixed |
| H11 | Dashboard SHAP widget uses static data | `frontend/lib/dashboard-data.ts` | Open — labeled as research |
| H12 | No SSO/MFA | N/A | Future milestone |
| H13 | Explain endpoint stores incorrect probabilities | `prediction_service.py` | ✅ Fixed |
| H14 | Top bar search non-functional | `frontend/components/layout/top-bar.tsx` | ✅ Fixed |
| H15 | Notification bell was decorative | `frontend/components/layout/top-bar.tsx` | ✅ Fixed |

---

## Medium Priority Findings

| ID | Finding | Status |
|----|---------|--------|
| M1 | Graph loads full Elliptic edgelist into memory | Open |
| M2 | No pagination on history beyond 200 items | Open |
| M3 | No automated backend tests | Open |
| M4 | No CI/CD pipeline | Open |
| M4 | Admin health status hardcoded | ✅ Fixed — uses `/health` |
| M5 | No password reset / email verification | Open — refresh tokens added |
| M6 | Batch jobs always status=completed | Open |
| M7 | No evidence attachment on cases | Open |
| M8 | Cluster exposure chart is synthetic | Open — labeled |
| M9 | No gzip / request ID middleware | Partial |
| M10 | OpenAPI docs public in production | ⚠️ Disable when `DEBUG=false` |

---

## Low Priority Findings

| ID | Finding |
|----|---------|
| L1 | Sidebar AUROC hardcoded vs live metrics |
| L2 | Landing page static benchmark stats |
| L3 | No idempotency keys on write endpoints |
| L4 | Case notes lack edit/delete |
| L5 | No webhook/SIEM integration |
| L6 | Python version mismatch Docker 3.11 vs local 3.14 |

---

## Frontend Audit

### Pages Status

| Route | Real API | Mock/Placeholder | Flow Status |
|-------|----------|------------------|-------------|
| `/` | — | Static stats | ✅ Works |
| `/login` | ✅ | Demo pre-fill | ✅ Works |
| `/dashboard` | ✅ | SHAP widget, cluster chart | ⚠️ Partial |
| `/upload` | ✅ | — | ✅ Works |
| `/transactions` | ✅ | — | ✅ Works |
| `/transactions/[id]` | ✅ | Template narrative | ✅ Works |
| `/explainability` | ✅ | Default tx, template NL | ✅ Works |
| `/investigations` | ✅ | Fallback rows when no cases | ✅ Works |
| `/investigations/[id]` | ✅ | — | ✅ Added |
| `/alerts` | ✅ | — | ✅ Works |
| `/drift` | ✅ | Fallback score 0.72 | ✅ Works |
| `/network` | ✅ | — | ✅ Works |
| `/reports` | ✅ | — | ✅ Fixed |
| `/admin` | ✅ | — | ✅ Works |

### User Flow Verification

| # | Flow | Status |
|---|------|--------|
| 1 | Registration → Login → Dashboard | ✅ Works (admin registers users) |
| 2 | Upload CSV → Validate → Predict → Display | ✅ Works |
| 3 | Single Transaction Analysis | ✅ Works via explainability |
| 4 | Batch Analysis | ✅ Works via upload |
| 5 | View Explainability | ✅ Works |
| 6 | Create Investigation Case | ✅ Works |
| 7 | Assign Case | ✅ Works via case detail |
| 8 | Close Case | ✅ Works |
| 9 | Export Report | ✅ Fixed |
| 10 | Review Drift Monitoring | ✅ Works |

---

## Backend Audit

### Endpoints (28+)

Auth, predict, explain, upload, metrics, drift, history, cases, graph, admin, **alerts**, **reports**, **health (deep)**

### Database Tables

| Table | Status |
|-------|--------|
| users | ✅ Exists |
| organizations | ✅ Added |
| batch_jobs (uploads) | ✅ Exists |
| transactions | ✅ Exists |
| predictions | ✅ Exists |
| shap_results | ✅ Exists |
| cases | ✅ Exists |
| case_notes | ✅ Exists |
| alerts | ✅ Added |
| drift_events | ✅ Exists |
| reports | ✅ Wired to API |
| audit_logs | ✅ Exists |
| refresh_tokens | ✅ Added |

### Security Controls

| Control | Status |
|---------|--------|
| JWT Authentication | ✅ |
| Refresh Tokens | ✅ Added |
| RBAC (admin/analyst/manager/viewer) | ✅ manager added |
| bcrypt password hashing | ✅ |
| Rate limiting (predict/explain/upload) | ✅ |
| Rate limiting (login) | ✅ Added |
| CORS configuration | ✅ |
| Security headers | ✅ Added |
| Audit logging | ✅ |
| Input validation (Pydantic) | ✅ |
| Upload size limit | ✅ |
| SQL injection prevention (ORM) | ✅ |
| CSRF | ⚠️ SameSite cookies recommended |
| Secrets management | ⚠️ Use env vars / vault |

---

## Deployment Audit

| Component | Demo | Production |
|-----------|------|------------|
| docker-compose.yml | ✅ | ⚠️ Use prod overlay |
| docker-compose.prod.yml | ✅ Added | ✅ |
| nginx.conf | ✅ Added | ✅ |
| .env.example | ✅ Added | ✅ |
| Alembic migrations | ✅ Added | ✅ |
| CI/CD | ❌ | Future |
| TLS/HTTPS | ⚠️ Via nginx | Configure certs |
| Postgres backups | ❌ | Document in LAUNCH_CHECKLIST |

---

## Remaining Blockers for Enterprise Pilot

1. **Rotate all secrets** (JWT, DB password) before any external deployment
2. **Disable demo user seeding** in production (`SEED_DEMO_USERS=false`)
3. **Configure TLS** via nginx with real certificates
4. **Move JWT to httpOnly cookies** (requires frontend auth refactor)
5. **Add multi-tenant org scoping** on all queries
6. **Implement live drift computation** from production traffic
7. **Add automated test suite** and CI pipeline
8. **Sanctions/PEP screening integration** (external vendor)

---

## Recommended Next Milestones

| Milestone | Timeline | Priority |
|-----------|----------|----------|
| M1: Secrets + TLS + prod deploy | 1 week | P0 |
| M2: httpOnly cookie auth | 1 week | P0 |
| M3: Multi-tenant org isolation | 2 weeks | P1 |
| M4: Live drift + alert rules engine | 2 weeks | P1 |
| M5: CI/CD + test coverage | 2 weeks | P1 |
| M6: SSO/OIDC (Okta, Azure AD) | 3 weeks | P2 |
| M7: Sanctions screening API | 4 weeks | P2 |
| M8: Async job queue (Celery) for batch/SHAP | 3 weeks | P2 |

---

*This audit reflects the platform state after Phase 1 production hardening. Re-run audit after each milestone.*
