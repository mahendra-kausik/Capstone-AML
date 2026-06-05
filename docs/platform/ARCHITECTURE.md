# AML Intelligence Platform — System Architecture

## Overview

Production wrapper around the Capstone-AML research stack (Elliptic Bitcoin, Static GCN + EvolveGCN-H, KernelSHAP drift). **No retraining** — serves frozen checkpoints from `models/` and research artifacts from `data/`.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Next.js 15 Frontend                          │
│  /dashboard /upload /analysis /history /drift /explainability   │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS + JWT
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend (backend/)                   │
│  Routes: auth, predict, batch, explain, metrics, drift, history │
│  Services: Prediction, SHAP, Drift, Upload, Audit                 │
└─────┬──────────────────┬──────────────────┬─────────────────────┘
      │                  │                  │
      ▼                  ▼                  ▼
 PostgreSQL          ML Runtime         Research artifacts
 (users, txs,         PyTorch + PyG      models/*.pt
  predictions,        165-dim scaler     data/processed/
  shap, drift)                             data/shap/
```

## ML inference contract

| Input | Format |
|-------|--------|
| Single / batch | JSON or CSV: `tx_id`, optional `time_step`, `feat_0`…`feat_164` |
| Edges (optional) | `src_tx_id`, `dst_tx_id` for mini-graph batch inference |
| Scaling | `StandardScaler` from `data/processed/scaler.pkl` (train T1–T34) |

| Model | Checkpoint | Inference mode |
|-------|------------|----------------|
| Static GCN | `models/static_gcn_best.pt` | Default; fixed weights |
| EvolveGCN-H | `models/evolvegcn_best.pt` | `model=evolve`; warm state optional |

**Risk score** = P(illicit) from softmax. **Classification** = illicit if score ≥ 0.5.

**SHAP** = KernelSHAP (tabular, empty `edge_index`) aligned with research notebooks; background from pre-sampled licit nodes.

**Drift** = Kendall τ from `data/shap/kendall_tau_results.json` + Evolve twin; T43 F1 from summary JSONs.

## Security

- JWT (access + refresh)
- Roles: `admin`, `analyst`, `viewer`
- Rate limit on `/predict*` and `/explain`
- Pydantic validation on all payloads
- `audit_logs` table for API actions

## Deployment

`docker-compose.yml`: `postgres`, `backend`, `frontend`. Research repo mounted read-only into backend at `/research`.
