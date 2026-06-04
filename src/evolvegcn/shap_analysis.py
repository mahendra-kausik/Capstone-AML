"""KernelSHAP drift analysis for EvolveGCN-H (mirrors Static GCN notebook)."""
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

from .config import (
    FIGURES_DIR,
    IN_CHANNELS,
    REPO_ROOT,
    SEED,
    SNAPSHOTS_PATH,
)
from .model import EvolveGCNH

SHAP_DIR = REPO_ROOT / "data" / "shap"
BEST_CHECKPOINT = REPO_ROOT / "models" / "evolvegcn_best.pt"
STATIC_TAU_PATH = SHAP_DIR / "kendall_tau_results.json"
EVOLVE_TAU_PATH = SHAP_DIR / "evolvegcn_kendall_tau_results.json"

WINDOWS = {
    "W1": list(range(0, 10)),
    "W2": list(range(10, 20)),
    "W3": list(range(20, 30)),
    "W4": list(range(30, 49)),
}

N_SHAP_SAMPLES = 200
N_BACKGROUND = 100
MAX_ILLICIT_PER_WINDOW = 1600
TOP_K = 10
TOP_K_TAU = 15
TAU_THRESHOLD = 0.70
BEESWARM_TOP = 15


def _load_evolve_model(checkpoint_path: Path, device: torch.device) -> EvolveGCNH:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def collect_window_data(snapshots: list, idx_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
    x_parts, y_parts = [], []
    for i in idx_list:
        s = snapshots[i]
        x_parts.append(s.x.cpu().numpy())
        y_parts.append(s.y.cpu().numpy())
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def warm_window_state(
    model: EvolveGCNH,
    snapshots: list,
    idx_list: list[int],
    device: torch.device,
) -> tuple:
    """Evolve W_state through window snapshots (empty graph, same as Static SHAP)."""
    empty_ei = torch.zeros((2, 0), dtype=torch.long, device=device)
    model.eval()
    model.reset_state(device)
    with torch.no_grad():
        for i in idx_list:
            data = snapshots[i].to(device)
            model(data.x, empty_ei)
    return model.get_state()


def make_evolve_shap_predictor(
    model: EvolveGCNH,
    snapshots: list,
    idx_list: list[int],
    device: torch.device,
):
    """Tabular KernelSHAP predictor; resets W_state to window-end before each call."""
    frozen_state = warm_window_state(model, snapshots, idx_list, device)
    empty_ei = torch.zeros((2, 0), dtype=torch.long, device=device)

    def predictor(x_numpy: np.ndarray) -> np.ndarray:
        model.set_state(frozen_state)
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = model(x_tensor, empty_ei)
            probs = F.softmax(logits, dim=1)[:, 1]
        model.set_state(frozen_state)
        return probs.cpu().numpy()

    return predictor


def compute_window_shap(
    model: EvolveGCNH,
    snapshots: list,
    window_name: str,
    idx_list: list[int],
    device: torch.device,
    *,
    force: bool = False,
) -> dict[str, Any]:
    save_path = SHAP_DIR / f"evolvegcn_shap_{window_name}.pkl"
    if save_path.exists() and not force:
        with save_path.open("rb") as f:
            return pickle.load(f)

    t_start = idx_list[0] + 1
    t_end = idx_list[-1] + 1
    x, y = collect_window_data(snapshots, idx_list)
    licit_mask = y == 0
    illicit_mask = y == 1
    x_licit = x[licit_mask]
    x_illicit = x[illicit_mask]

    rng = np.random.default_rng(SEED)
    bg_n = min(N_BACKGROUND, len(x_licit))
    bg_idx = rng.choice(len(x_licit), bg_n, replace=False)
    background = x_licit[bg_idx]

    predictor = make_evolve_shap_predictor(model, snapshots, idx_list, device)
    explainer = shap.KernelExplainer(predictor, background)

    if len(x_illicit) > MAX_ILLICIT_PER_WINDOW:
        sample_idx = rng.choice(len(x_illicit), MAX_ILLICIT_PER_WINDOW, replace=False)
        x_illicit_sample = x_illicit[sample_idx]
    else:
        x_illicit_sample = x_illicit

    shap_vals = explainer.shap_values(
        x_illicit_sample, nsamples=N_SHAP_SAMPLES, silent=True
    )
    importance = np.abs(shap_vals).mean(axis=0)
    ranked_idx = np.argsort(importance)[::-1]

    result = {
        "model": "EvolveGCN-H",
        "window": window_name,
        "time_range": f"T{t_start}–T{t_end}",
        "shap_values": shap_vals,
        "X_illicit": x_illicit_sample,
        "importance": importance,
        "ranked_idx": ranked_idx,
        "top10_idx": ranked_idx[:TOP_K].tolist(),
        "top10_names": [f"feat_{i}" for i in ranked_idx[:TOP_K]],
        "top10_vals": importance[ranked_idx[:TOP_K]].tolist(),
        "n_illicit_total": int(illicit_mask.sum()),
        "n_illicit_shap": int(len(x_illicit_sample)),
    }

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump(result, f)
    return result


def compute_kendall_tau(shap_results: dict[str, dict]) -> list[dict]:
    window_names = list(WINDOWS.keys())
    tau_results = []
    for i in range(len(window_names) - 1):
        w_curr, w_next = window_names[i], window_names[i + 1]
        top_curr = set(shap_results[w_curr]["ranked_idx"][:TOP_K_TAU])
        top_next = set(shap_results[w_next]["ranked_idx"][:TOP_K_TAU])
        union_features = sorted(top_curr | top_next)
        imp_curr = shap_results[w_curr]["importance"][union_features]
        imp_next = shap_results[w_next]["importance"][union_features]
        rank_curr = np.argsort(np.argsort(-imp_curr))
        rank_next = np.argsort(np.argsort(-imp_next))
        tau, pval = kendalltau(rank_curr, rank_next)
        tau_results.append(
            {
                "comparison": f"{w_curr}→{w_next}",
                "tau": round(float(tau), 4),
                "pval": round(float(pval), 6),
                "drift": bool(tau < TAU_THRESHOLD),
            }
        )
    return tau_results


def plot_beeswarm(shap_results: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

    for window_name, res in shap_results.items():
        shap_vals = res["shap_values"]
        x_illicit = res["X_illicit"]
        top_idx = res["ranked_idx"][:BEESWARM_TOP]
        feat_names = [f"feat_{i}" for i in top_idx]

        explanation = shap.Explanation(
            values=shap_vals[:, top_idx],
            data=x_illicit[:, top_idx],
            feature_names=feat_names,
        )
        shap.plots.beeswarm(explanation, max_display=BEESWARM_TOP, show=False)
        plt.title(
            f"EvolveGCN-H SHAP — {window_name} ({res['time_range']})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        path = out_dir / f"evolvegcn_beeswarm_{window_name}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


def plot_tau_drift(tau_results: list[dict], out_dir: Path, *, prefix: str = "evolvegcn") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = [r["comparison"] for r in tau_results]
    taus = [r["tau"] for r in tau_results]
    colors = ["#E05252" if r["drift"] else "#0D9488" for r in tau_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(comparisons, taus, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax.axhline(y=TAU_THRESHOLD, color="black", linestyle="--", linewidth=1.8)
    for bar, tau_val in zip(bars, taus):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{tau_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "EvolveGCN-H — Kendall τ Across SHAP Windows",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Window Transition")
    ax.set_ylabel("Kendall τ")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"{prefix}_tau_plot.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_static_vs_evolve_tau(
    static_tau: list[dict],
    evolve_tau: list[dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["comparison"] for r in static_tau]
    static_vals = [r["tau"] for r in static_tau]
    evolve_vals = [r["tau"] for r in evolve_tau]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, static_vals, width, label="Static GCN", color="#94A3B8")
    ax.bar(x + width / 2, evolve_vals, width, label="EvolveGCN-H", color="#0D9488")
    ax.axhline(TAU_THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"τ={TAU_THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Kendall τ")
    ax.set_title("SHAP Feature-Rank Stability: Static GCN vs EvolveGCN-H", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (s, e) in enumerate(zip(static_vals, evolve_vals)):
        ax.text(i - width / 2, s + 0.02, f"{s:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, e + 0.02, f"{e:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    path = out_dir / "static_vs_evolve_tau.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def run_evolvegcn_shap_pipeline(
    *,
    checkpoint_path: Path = BEST_CHECKPOINT,
    snapshots_path: Path = SNAPSHOTS_PATH,
    device: torch.device | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    snapshots = torch.load(snapshots_path, map_location="cpu", weights_only=False)
    model = _load_evolve_model(checkpoint_path, device)

    shap_results: dict[str, dict] = {}
    for window_name, idx_list in WINDOWS.items():
        print(f"\n=== {window_name} (T{idx_list[0]+1}–T{idx_list[-1]+1}) ===")
        shap_results[window_name] = compute_window_shap(
            model, snapshots, window_name, idx_list, device, force=force
        )
        top5 = shap_results[window_name]["top10_names"][:5]
        print(f"  Top 5: {top5}")

    tau_results = compute_kendall_tau(shap_results)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    with EVOLVE_TAU_PATH.open("w") as f:
        json.dump(tau_results, f, indent=2)
    print(f"\nSaved {EVOLVE_TAU_PATH}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_beeswarm(shap_results, FIGURES_DIR)
    plot_tau_drift(tau_results, FIGURES_DIR)

    static_tau = json.loads(STATIC_TAU_PATH.read_text()) if STATIC_TAU_PATH.exists() else []
    if static_tau:
        plot_static_vs_evolve_tau(static_tau, tau_results, FIGURES_DIR)

    return {"shap_results": shap_results, "tau_results": tau_results, "static_tau": static_tau}
