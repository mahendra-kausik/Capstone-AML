"""SHAP explanations (KernelSHAP, research-aligned tabular mode)."""
from __future__ import annotations

import numpy as np
import shap
import torch
import torch.nn.functional as F

from app.config import get_settings
from app.ml.inference import ModelName
from app.ml.loader import get_artifacts


def _make_predictor(model_name: ModelName):
    artifacts = get_artifacts()
    device = artifacts.device
    empty_ei = torch.zeros((2, 0), dtype=torch.long, device=device)

    if model_name == "static":
        model = artifacts.static_model
        model.eval()

        def fn(x_numpy: np.ndarray) -> np.ndarray:
            x = torch.tensor(x_numpy, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = model(x, empty_ei)
                return F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    else:
        model = artifacts.evolve_model
        model.eval()
        model.reset_state(device)
        frozen = (
            model.layer1.W_state.clone() if model.layer1.W_state is not None else None,
            model.layer2.W_state.clone() if model.layer2.W_state is not None else None,
        )

        def fn(x_numpy: np.ndarray) -> np.ndarray:
            if frozen[0] is not None:
                model.layer1.W_state = frozen[0].to(device)
                model.layer2.W_state = frozen[1].to(device)
            x = torch.tensor(x_numpy, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = model(x, empty_ei)
                return F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    return fn


def _fallback_importance(features: np.ndarray, risk_score: float) -> list[dict]:
    """Fast demo path when SHAP background unavailable."""
    ranked = np.argsort(np.abs(features))[::-1][:10]
    return [
        {
            "name": f"feat_{int(i)}",
            "index": int(i),
            "feature_value": round(float(features[i]), 4),
            "contribution": round(float(abs(features[i]) * risk_score), 4),
        }
        for i in ranked
    ]


def generate_shap(
    features: np.ndarray,
    model_name: ModelName = "static",
    nsamples: int | None = None,
) -> dict:
    settings = get_settings()
    artifacts = get_artifacts()
    nsamples = nsamples or settings.shap_default_nsamples

    x = features.reshape(1, -1)
    pred = _make_predictor(model_name)
    with torch.no_grad():
        risk = float(pred(x)[0])

    background = artifacts.background
    if background is None or len(background) < 5:
        tops = _fallback_importance(features, risk)
        return {
            "risk_score": round(risk, 4),
            "prediction": "illicit" if risk >= 0.5 else "licit",
            "confidence": round(max(risk, 1 - risk), 4),
            "top_features": tops,
            "method": "magnitude_fallback",
        }

    try:
        explainer = shap.KernelExplainer(pred, background)
        shap_vals = explainer.shap_values(x, nsamples=nsamples, silent=True)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.asarray(shap_vals).reshape(-1)
        importance = np.abs(shap_vals)
        top_idx = np.argsort(importance)[::-1][:15]
        tops = [
            {
                "name": f"feat_{int(i)}",
                "index": int(i),
                "shap_value": round(float(shap_vals[i]), 6),
                "feature_value": round(float(features[i]), 4),
                "contribution": round(float(importance[i]), 6),
            }
            for i in top_idx
        ]
        return {
            "risk_score": round(risk, 4),
            "prediction": "illicit" if risk >= 0.5 else "licit",
            "confidence": round(max(risk, 1 - risk), 4),
            "top_features": tops,
            "method": "kernel_shap",
            "nsamples": nsamples,
        }
    except Exception:
        tops = _fallback_importance(features, risk)
        return {
            "risk_score": round(risk, 4),
            "prediction": "illicit" if risk >= 0.5 else "licit",
            "confidence": round(max(risk, 1 - risk), 4),
            "top_features": tops,
            "method": "magnitude_fallback",
        }
