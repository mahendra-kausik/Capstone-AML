"""Model inference: predict, batch, risk scores."""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from app.ml.architectures import EvolveGCNH, StaticGCN
from app.ml.loader import MLArtifacts, get_artifacts


ModelName = Literal["static", "evolve"]


def _empty_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((2, 0), dtype=torch.long, device=device)


def _build_edge_index(
    tx_ids: list[str],
    edges: list[dict] | None,
    device: torch.device,
) -> torch.Tensor:
    if not edges:
        return _empty_edge_index(len(tx_ids), device)
    idx_map = {tid: i for i, tid in enumerate(tx_ids)}
    src, dst = [], []
    for e in edges:
        s = str(e.get("src_tx_id", e.get("src", "")))
        d = str(e.get("dst_tx_id", e.get("dst", "")))
        if s in idx_map and d in idx_map:
            src.append(idx_map[s])
            dst.append(idx_map[d])
    if not src:
        return _empty_edge_index(len(tx_ids), device)
    ei = torch.tensor([src, dst], dtype=torch.long, device=device)
    return ei


def _forward(
    model_name: ModelName,
    X: np.ndarray,
    edge_index: torch.Tensor,
    artifacts: MLArtifacts,
) -> tuple[np.ndarray, np.ndarray]:
    device = artifacts.device
    x = torch.tensor(X, dtype=torch.float32, device=device)

    if model_name == "static":
        model = artifacts.static_model
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, edge_index)
    else:
        model = artifacts.evolve_model
        model.eval()
        model.reset_state(device)
        with torch.no_grad():
            logits, _ = model(x, edge_index)

    probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs[:, 0], probs[:, 1]


def score_to_result(
    prob_licit: float,
    prob_illicit: float,
    top_features: list[dict] | None = None,
) -> dict:
    risk = float(prob_illicit)
    pred = "illicit" if risk >= 0.5 else "licit"
    confidence = float(max(prob_licit, prob_illicit))
    return {
        "risk_score": round(risk, 4),
        "prediction": pred,
        "confidence": round(confidence, 4),
        "prob_licit": round(float(prob_licit), 4),
        "prob_illicit": round(float(prob_illicit), 4),
        "top_features": top_features or [],
    }


def predict(
    features: np.ndarray,
    model_name: ModelName = "static",
    edges: list[dict] | None = None,
    tx_id: str = "single",
) -> dict:
    artifacts = get_artifacts()
    X = features.reshape(1, -1)
    ei = _build_edge_index([tx_id], edges, artifacts.device)
    p0, p1 = _forward(model_name, X, ei, artifacts)
    return score_to_result(float(p0[0]), float(p1[0]))


def predict_batch(
    features_list: list[np.ndarray],
    tx_ids: list[str],
    model_name: ModelName = "static",
    edges: list[dict] | None = None,
) -> list[dict]:
    artifacts = get_artifacts()
    X = np.stack(features_list, axis=0)
    ei = _build_edge_index(tx_ids, edges, artifacts.device)
    p0, p1 = _forward(model_name, X, ei, artifacts)
    return [score_to_result(float(p0[i]), float(p1[i])) for i in range(len(tx_ids))]


def generate_risk_score(prob_illicit: float) -> float:
    return round(float(prob_illicit), 4)
