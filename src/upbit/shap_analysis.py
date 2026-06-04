"""KernelSHAP drift analysis for Upbit Static GCN and EvolveGCN-H."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from src.evolvegcn.model import EvolveGCNH
from src.evolvegcn.shap_analysis import (
    BEESWARM_TOP,
    N_BACKGROUND,
    N_SHAP_SAMPLES,
    TOP_K,
    TOP_K_TAU,
    WINDOWS,
    collect_window_data,
    compute_kendall_tau,
    make_evolve_shap_predictor,
    plot_static_vs_evolve_tau,
    plot_tau_drift,
)
from src.static_gcn.model import StaticGCN
from src.upbit.config import IN_CHANNELS, SEED, SHAP_DIR, SNAPSHOTS_PATH


def _load_static_model(path: Path, device: torch.device) -> StaticGCN:
    model = StaticGCN(in_channels=IN_CHANNELS).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_evolve_model(path: Path, device: torch.device) -> EvolveGCNH:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        model = EvolveGCNH(
            in_channels=IN_CHANNELS,
            hidden1=int(cfg.get("hidden1", 64)),
            hidden2=int(cfg.get("hidden2", 32)),
            dropout=float(cfg.get("dropout", 0.5)),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = EvolveGCNH().to(device)
        model.load_state_dict(ckpt)
    model.eval()
    return model


def make_static_shap_predictor(model: StaticGCN, device: torch.device):
    empty_ei = torch.zeros((2, 0), dtype=torch.long, device=device)

    def predictor(x_numpy: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = model(x_tensor, empty_ei)
            probs = F.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy()

    return predictor


def compute_static_window_shap(
    model: StaticGCN,
    snapshots: list,
    window_name: str,
    idx_list: list[int],
    device: torch.device,
    *,
    force: bool = False,
    prefix: str = "upbit_static",
) -> dict[str, Any]:
    save_path = SHAP_DIR / f"{prefix}_shap_{window_name}.pkl"
    if save_path.exists() and not force:
        with save_path.open("rb") as f:
            return pickle.load(f)

    t_start, t_end = idx_list[0] + 1, idx_list[-1] + 1
    x, y = collect_window_data(snapshots, idx_list)
    x_licit, x_illicit = x[y == 0], x[y == 1]
    rng = np.random.default_rng(SEED)
    bg_n = min(N_BACKGROUND, len(x_licit))
    background = x_licit[rng.choice(len(x_licit), bg_n, replace=False)]

    predictor = make_static_shap_predictor(model, device)
    explainer = shap.KernelExplainer(predictor, background)
    if len(x_illicit) > 1600:
        sample_idx = rng.choice(len(x_illicit), 1600, replace=False)
        x_sample = x_illicit[sample_idx]
    else:
        x_sample = x_illicit

    shap_vals = explainer.shap_values(x_sample, nsamples=N_SHAP_SAMPLES, silent=True)
    importance = np.abs(shap_vals).mean(axis=0)
    ranked_idx = np.argsort(importance)[::-1]

    result = {
        "model": "Static GCN",
        "dataset": "UpbitHack",
        "window": window_name,
        "time_range": f"T{t_start}–T{t_end}",
        "shap_values": shap_vals,
        "X_illicit": x_sample,
        "importance": importance,
        "ranked_idx": ranked_idx,
        "top10_idx": ranked_idx[:TOP_K].tolist(),
        "top10_names": [f"feat_{i}" for i in ranked_idx[:TOP_K]],
    }
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump(result, f)
    return result


def compute_evolve_window_shap(
    model: EvolveGCNH,
    snapshots: list,
    window_name: str,
    idx_list: list[int],
    device: torch.device,
    *,
    force: bool = False,
    prefix: str = "upbit_evolve",
) -> dict[str, Any]:
    save_path = SHAP_DIR / f"{prefix}_shap_{window_name}.pkl"
    if save_path.exists() and not force:
        with save_path.open("rb") as f:
            return pickle.load(f)

    t_start, t_end = idx_list[0] + 1, idx_list[-1] + 1
    x, y = collect_window_data(snapshots, idx_list)
    x_licit, x_illicit = x[y == 0], x[y == 1]
    rng = np.random.default_rng(SEED)
    bg_n = min(N_BACKGROUND, len(x_licit))
    background = x_licit[rng.choice(len(x_licit), bg_n, replace=False)]

    predictor = make_evolve_shap_predictor(model, snapshots, idx_list, device)
    explainer = shap.KernelExplainer(predictor, background)
    if len(x_illicit) > 1600:
        sample_idx = rng.choice(len(x_illicit), 1600, replace=False)
        x_sample = x_illicit[sample_idx]
    else:
        x_sample = x_illicit

    shap_vals = explainer.shap_values(x_sample, nsamples=N_SHAP_SAMPLES, silent=True)
    importance = np.abs(shap_vals).mean(axis=0)
    ranked_idx = np.argsort(importance)[::-1]

    result = {
        "model": "EvolveGCN-H",
        "dataset": "UpbitHack",
        "window": window_name,
        "time_range": f"T{t_start}–T{t_end}",
        "shap_values": shap_vals,
        "X_illicit": x_sample,
        "importance": importance,
        "ranked_idx": ranked_idx,
    }
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump(result, f)
    return result


def run_upbit_shap_pipeline(
    *,
    static_ckpt: Path,
    evolve_ckpt: Path,
    snapshots_path: Path = SNAPSHOTS_PATH,
    device: torch.device | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if device is None:
        from src.evolvegcn.training import get_device

        device = get_device()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    snapshots = torch.load(snapshots_path, map_location="cpu", weights_only=False)

    static_model = _load_static_model(static_ckpt, device)
    evolve_model = _load_evolve_model(evolve_ckpt, device)

    static_results, evolve_results = {}, {}
    for wname, idx in WINDOWS.items():
        print(f"\n=== Static SHAP {wname} ===")
        static_results[wname] = compute_static_window_shap(
            static_model, snapshots, wname, idx, device, force=force
        )
        print(f"=== Evolve SHAP {wname} ===")
        evolve_results[wname] = compute_evolve_window_shap(
            evolve_model, snapshots, wname, idx, device, force=force
        )

    static_tau = compute_kendall_tau(static_results)
    evolve_tau = compute_kendall_tau(evolve_results)

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    static_tau_path = SHAP_DIR / "upbit_static_kendall_tau.json"
    evolve_tau_path = SHAP_DIR / "upbit_evolve_kendall_tau.json"
    static_tau_path.write_text(json.dumps(static_tau, indent=2))
    evolve_tau_path.write_text(json.dumps(evolve_tau, indent=2))

    from src.upbit.config import FIGURES_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Beeswarm with upbit prefix
    for wname, res in static_results.items():
        _plot_beeswarm_one(res, FIGURES_DIR / f"upbit_static_beeswarm_{wname}.png", "Static GCN")
    for wname, res in evolve_results.items():
        _plot_beeswarm_one(res, FIGURES_DIR / f"upbit_evolve_beeswarm_{wname}.png", "EvolveGCN-H")

    plot_tau_drift(static_tau, FIGURES_DIR, prefix="upbit_static")
    plot_tau_drift(evolve_tau, FIGURES_DIR, prefix="upbit_evolve")
    plot_static_vs_evolve_tau(static_tau, evolve_tau, FIGURES_DIR)
    plt.close("all")

    rankings_path = SHAP_DIR / "upbit_feature_rankings.json"
    rankings = {
        "static": {
            w: {
                "top15": [f"feat_{i}" for i in r["ranked_idx"][:TOP_K_TAU]],
                "top15_importance": r["importance"][r["ranked_idx"][:TOP_K_TAU]].tolist(),
            }
            for w, r in static_results.items()
        },
        "evolve": {
            w: {
                "top15": [f"feat_{i}" for i in r["ranked_idx"][:TOP_K_TAU]],
                "top15_importance": r["importance"][r["ranked_idx"][:TOP_K_TAU]].tolist(),
            }
            for w, r in evolve_results.items()
        },
    }
    rankings_path.write_text(json.dumps(rankings, indent=2))

    return {
        "static_tau": static_tau,
        "evolve_tau": evolve_tau,
        "static_results": static_results,
        "evolve_results": evolve_results,
    }


def _plot_beeswarm_one(res: dict, out_path: Path, title_model: str) -> None:
    top_idx = res["ranked_idx"][:BEESWARM_TOP]
    explanation = shap.Explanation(
        values=res["shap_values"][:, top_idx],
        data=res["X_illicit"][:, top_idx],
        feature_names=[f"feat_{i}" for i in top_idx],
    )
    shap.plots.beeswarm(explanation, max_display=BEESWARM_TOP, show=False)
    plt.title(f"{title_model} SHAP — {res['window']} ({res['time_range']})", fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
