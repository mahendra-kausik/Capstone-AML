# API Contracts

Base URL: `http://localhost:8000/api/v1`

## Auth

### POST `/auth/login`
```json
{ "email": "analyst@aml.local", "password": "changeme" }
```
Response: `{ "access_token", "token_type": "bearer", "role" }`

### POST `/auth/register` (admin only in production)

---

## POST `/predict`

Single transaction AML scoring.

**Request**
```json
{
  "model": "static",
  "tx_id": "230425980",
  "time_step": 37,
  "features": [0.12, -0.4, ...],
  "feature_dict": { "feat_0": 0.12, "feat_1": -0.4 }
}
```
Either `features` (length 165) or `feature_dict` required.

**Response**
```json
{
  "tx_id": "230425980",
  "model": "static",
  "risk_score": 0.93,
  "prediction": "illicit",
  "confidence": 0.91,
  "prob_licit": 0.07,
  "prob_illicit": 0.93,
  "top_features": [
    { "name": "feat_42", "index": 42, "contribution": 0.18 }
  ],
  "prediction_id": "uuid"
}
```

---

## POST `/batch-predict`

**Request**
```json
{
  "model": "static",
  "transactions": [ { "tx_id": "1", "features": [...] } ],
  "edges": [ { "src_tx_id": "1", "dst_tx_id": "2" } ]
}
```

**Response**
```json
{
  "job_id": "uuid",
  "count": 100,
  "results": [ /* same shape as /predict */ ],
  "summary": { "high_risk": 12, "mean_risk_score": 0.34 }
}
```

---

## POST `/explain`

KernelSHAP for one transaction (slower).

**Request**: same as `/predict` + optional `nsamples` (default 100).

**Response**
```json
{
  "tx_id": "230425980",
  "risk_score": 0.93,
  "prediction": "illicit",
  "confidence": 0.91,
  "top_features": [ { "name": "feat_12", "shap_value": 0.05, "feature_value": 1.2 } ],
  "shap_id": "uuid"
}
```

---

## GET `/metrics`

Model performance from research summaries (no retrain).

```json
{
  "static_gcn": { "test_auroc": 0.8573, "test_f1": 0.4677, ... },
  "evolvegcn_h": { "test_auroc": 0.7666, "test_f1": 0.3269, ... }
}
```

---

## GET `/drift`

SHAP Kendall τ + post-T43 F1 drift probe.

```json
{
  "kendall_tau": { "static": [...], "evolve": [...] },
  "t43_drift": { "static": {...}, "evolve": {...} },
  "alerts": [ { "type": "shap_drift", "window": "W3→W4", "tau": 0.1474 } ]
}
```

---

## GET `/history`

Query params: `limit`, `offset`, `risk_min`, `model`.

Returns stored predictions + optional SHAP links.

---

## POST `/upload`

`multipart/form-data`: `file` (CSV).

Expected columns: `tx_id`, `feat_0`…`feat_164` (or 167-col Elliptic layout).

Returns `batch_id` + runs async-style batch predict (sync MVP).
