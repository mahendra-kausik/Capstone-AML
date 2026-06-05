# Full Folder Structure

```
Capstone-AML/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── startup.py
│       ├── core/
│       │   ├── security.py
│       │   └── deps.py
│       ├── database/
│       │   ├── models.py
│       │   └── session.py
│       ├── schemas/
│       │   ├── auth.py
│       │   └── predict.py
│       ├── routes/
│       │   ├── auth.py
│       │   ├── predict.py
│       │   ├── metrics.py
│       │   ├── drift.py
│       │   └── history.py
│       ├── services/
│       │   ├── prediction_service.py
│       │   ├── drift_service.py
│       │   └── upload_service.py
│       └── ml/
│           ├── architectures.py
│           ├── loader.py
│           ├── inference.py
│           ├── feature_pipeline.py
│           └── shap_explainer.py
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── app/
│       ├── page.tsx              # Login
│       ├── dashboard/page.tsx
│       ├── upload/page.tsx
│       ├── analysis/page.tsx
│       ├── history/page.tsx
│       ├── drift/page.tsx
│       ├── explainability/page.tsx
│       └── admin/page.tsx
├── docs/platform/
│   ├── ARCHITECTURE.md
│   ├── API_CONTRACTS.md
│   ├── DATABASE_SCHEMA.md
│   ├── DEPLOYMENT.md
│   ├── IMPLEMENTATION_PLAN.md
│   └── FOLDER_STRUCTURE.md
├── docker-compose.yml
├── samples/demo_transactions.csv
├── models/                       # existing checkpoints
└── data/processed/               # scaler, summaries, shap json
```
