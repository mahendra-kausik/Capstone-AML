# AML Intelligence Platform — Investor Demo Guide

## Quick Start (5 minutes)

```bash
# From project root
docker compose up --build

# Open http://localhost:3000
```

## Demo Accounts

| Role | Email | Password |
|------|-------|----------|
| Administrator | admin@example.com | demoaml2024 |
| Compliance Analyst | analyst@example.com | demoaml2024 |
| Viewer (read-only) | viewer@example.com | demoaml2024 |

## 15-Minute Demo Script

### 1. Executive Dashboard (2 min)
- Login as **analyst@example.com**
- Show KPI cards: transactions, high-risk alerts, model accuracy (AUROC 0.8573), drift score
- Highlight SHAP explainability and Kendall τ drift monitoring badges

### 2. Batch Upload (2 min)
- Navigate to **Batch Upload** (dashboard button or `/upload`)
- Upload `samples/demo_transactions.csv`
- Show real-time scoring results with risk scores and illicit/licit classifications

### 3. Transaction Analysis (2 min)
- Go to **Transactions** — filter by illicit
- Open a high-risk transaction detail
- Show SHAP feature attribution chart and risk metrics

### 4. Explainability (2 min)
- Navigate to **Explainability**
- Click "Generate SHAP" or load from prediction
- Show waterfall chart and natural language narrative explanation

### 5. Network Investigation (2 min)
- Open **Network Analysis**
- Enter seed transaction IDs from upload results
- Demonstrate graph exploration with risk-colored nodes

### 6. Alerts & Investigations (3 min)
- **Alerts Center** — show auto-generated high-risk alerts
- Click notification bell in top bar
- **Investigations** — create case from alert, open case detail
- Add investigation note, assign analyst, update status to closed

### 7. Drift Monitoring (1 min)
- **Drift Monitor** — Kendall τ stability charts
- Compare Static GCN vs EvolveGCN-H model degradation

### 8. Reports & Admin (1 min)
- **Reports** — export Compliance CSV
- **Admin** (login as admin) — audit log, user management

## Key Talking Points

- **Static GCN AUROC: 0.8573** — validated on Elliptic Bitcoin dataset
- **EvolveGCN-H AUROC: 0.7666** — temporal graph neural network
- **SHAP Explainability** — regulatory-grade feature attribution
- **Kendall τ Drift Monitoring** — concept drift detection
- **Enterprise case management** — full investigation workflow
- **Real-time alerting** — automated high-risk transaction alerts

## Sample Data

- `samples/demo_transactions.csv` — 9 transactions with 165 features each
- Pre-seeded drift events from research phase
- Demo organization: "Demo Financial Institution"

## Production Deployment

See `LAUNCH_CHECKLIST.md` and `docs/platform/DEPLOYMENT.md`.

```bash
cp .env.example .env
# Edit secrets
docker compose -f docker-compose.prod.yml up -d --build
```
