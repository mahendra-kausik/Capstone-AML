"""Load research checkpoints and artifacts once at startup."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import Settings
from app.ml.architectures import EvolveGCNH, StaticGCN


@dataclass
class MLArtifacts:
    static_model: StaticGCN
    evolve_model: EvolveGCNH
    scaler: Any
    meta: dict[str, Any]
    static_summary: dict[str, Any]
    evolve_summary: dict[str, Any]
    static_tau: list[dict]
    evolve_tau: list[dict]
    background: np.ndarray | None
    device: torch.device


_artifacts: MLArtifacts | None = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_static(settings: Settings, device: torch.device) -> StaticGCN:
    path = settings.models_dir / "static_gcn_best.pt"
    model = StaticGCN().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_evolve(settings: Settings, device: torch.device) -> EvolveGCNH:
    path = settings.models_dir / "evolvegcn_best.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = EvolveGCNH(
        hidden1=int(cfg.get("hidden1", 64)),
        hidden2=int(cfg.get("hidden2", 32)),
        dropout=float(cfg.get("dropout", 0.5)),
    ).to(device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def _load_background(settings: Settings) -> np.ndarray | None:
    """Precomputed licit background for SHAP (optional)."""
    bg_path = settings.processed_dir / "shap_background.npy"
    if bg_path.exists():
        return np.load(bg_path)
    snaps_path = settings.processed_dir / "snapshots.pt"
    if not snaps_path.exists():
        return None
    try:
        snapshots = torch.load(snaps_path, map_location="cpu", weights_only=False)
        licit_rows = []
        for s in snapshots[:5]:
            x = s.x.numpy()
            y = s.y.numpy()
            licit = x[y == 0]
            if len(licit):
                licit_rows.append(licit[: min(20, len(licit))])
        if licit_rows:
            bg = np.concatenate(licit_rows, axis=0)
            n = min(50, len(bg))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(bg), n, replace=False)
            return bg[idx]
    except Exception:
        pass
    return None


def load_artifacts(settings: Settings) -> MLArtifacts:
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    device = get_device()
    scaler_path = settings.processed_dir / "scaler.pkl"
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    meta_path = settings.processed_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    static_summary = {}
    evolve_summary = {}
    ss = settings.processed_dir / "static_gcn_summary.json"
    es = settings.processed_dir / "evolvegcn_summary.json"
    if ss.exists():
        static_summary = json.loads(ss.read_text())
    if es.exists():
        evolve_summary = json.loads(es.read_text())

    static_tau = []
    evolve_tau = []
    st = settings.shap_dir / "kendall_tau_results.json"
    et = settings.shap_dir / "evolvegcn_kendall_tau_results.json"
    if st.exists():
        static_tau = json.loads(st.read_text())
    if et.exists():
        evolve_tau = json.loads(et.read_text())

    _artifacts = MLArtifacts(
        static_model=_load_static(settings, device),
        evolve_model=_load_evolve(settings, device),
        scaler=scaler,
        meta=meta,
        static_summary=static_summary,
        evolve_summary=evolve_summary,
        static_tau=static_tau,
        evolve_tau=evolve_tau,
        background=_load_background(settings),
        device=device,
    )
    return _artifacts


def get_artifacts() -> MLArtifacts:
    from app.config import get_settings

    return load_artifacts(get_settings())
