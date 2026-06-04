"""Publication figures for EvolveGCN-H experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES_DIR, SHUTDOWN_STEP


def _style():
    plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def plot_training_curves(history: dict, out_dir: Path) -> None:
    _style()
    epochs = range(len(history["train_loss"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, history["train_loss"], color="#1a56a0", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("EvolveGCN-H — Training Loss")
    axes[0].grid(alpha=0.3)

    if history.get("val_loss"):
        ax2 = axes[1].twinx()
        ax2.plot(epochs, history["val_loss"], color="#64748B", linewidth=1.5, linestyle="--", label="Val Loss")
        ax2.set_ylabel("Val Loss", color="#64748B")
    axes[1].plot(epochs, history["val_auroc"], color="#0D9488", linewidth=2, label="Val AUROC")
    axes[1].plot(epochs, history["val_f1"], color="#F59E0B", linewidth=2, label="Val F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("EvolveGCN-H — Validation Metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "evolvegcn_training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_per_snapshot_curves(
    evolve_per_snap: list,
    static_per_snap: list | None,
    out_dir: Path,
) -> None:
    _style()
    e_steps = [r["time_step"] for r in evolve_per_snap]
    e_f1 = [r["F1"] for r in evolve_per_snap]
    e_auroc = [r["AUROC"] for r in evolve_per_snap]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    if static_per_snap:
        s_map = {r["time_step"]: r for r in static_per_snap}
        s_f1 = [s_map[t]["F1"] for t in e_steps]
        s_auroc = [s_map[t]["AUROC"] for t in e_steps]
        axes[0].plot(e_steps, s_f1, "o-", color="#94A3B8", linewidth=2, label="Static GCN")
        axes[1].plot(e_steps, s_auroc, "o-", color="#94A3B8", linewidth=2, label="Static GCN")

    axes[0].plot(e_steps, e_f1, "o-", color="#0D9488", linewidth=2, label="EvolveGCN-H")
    axes[1].plot(e_steps, e_auroc, "o-", color="#0D9488", linewidth=2, label="EvolveGCN-H")

    for ax in axes:
        ax.axvline(SHUTDOWN_STEP, color="#b45309", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(SHUTDOWN_STEP + 0.2, ax.get_ylim()[1] * 0.95, "T43", color="#b45309", fontsize=9)
        ax.set_xlabel("Snapshot")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("F1 (illicit)")
    axes[0].set_title("Per-Snapshot F1 — Test Period")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Per-Snapshot AUROC — Test Period")

    plt.suptitle("EvolveGCN-H vs Static GCN — Concept Drift at T43", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "both_models_drift_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(e_steps, e_f1, "o-", color="#0D9488", linewidth=2, label="EvolveGCN-H F1")
    if static_per_snap:
        ax.plot(e_steps, s_f1, "o-", color="#1a56a0", linewidth=2, label="Static GCN F1")
    ax.axvline(SHUTDOWN_STEP, color="#b45309", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Snapshot")
    ax.set_ylabel("F1")
    ax.set_title("F1 Over Test Snapshots (T37–T49)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "evolvegcn_f1_curve.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(e_steps, e_auroc, "o-", color="#0D9488", linewidth=2, label="EvolveGCN-H AUROC")
    if static_per_snap:
        ax.plot(e_steps, s_auroc, "o-", color="#1a56a0", linewidth=2, label="Static GCN AUROC")
    ax.axvline(SHUTDOWN_STEP, color="#b45309", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Snapshot")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC Over Test Snapshots (T37–T49)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "evolvegcn_auroc_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_pre_post_comparison(
    evolve_pre: float,
    evolve_post: float,
    static_pre: float,
    static_post: float,
    out_dir: Path,
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.array([0, 1])
    width = 0.3
    ax.bar(x - width / 2, [static_pre, static_post], width, color="#94A3B8", label="Static GCN")
    ax.bar(x + width / 2, [evolve_pre, evolve_post], width, color="#0D9488", label="EvolveGCN-H")
    for bars in ax.containers:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["Pre-T43\n(T37–T42)", "Post-T43\n(T44–T49)"])
    ax.set_ylabel("Mean F1 (illicit)")
    ax.set_title("Pre vs Post Dark Market Shutdown (T43)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "pre_post_t43_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_t43_bar(evolve_per_snap: list, static_per_snap: list | None, out_dir: Path) -> None:
    _style()
    e_t43 = next((r for r in evolve_per_snap if r["time_step"] == SHUTDOWN_STEP), None)
    if not e_t43:
        return
    labels = ["F1", "AUROC", "Precision", "Recall"]
    e_vals = [e_t43[k] for k in labels]
    s_vals = None
    if static_per_snap:
        s_t43 = next((r for r in static_per_snap if r["time_step"] == SHUTDOWN_STEP), None)
        if s_t43:
            s_vals = [s_t43[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if s_vals:
        ax.bar(x - width / 2, s_vals, width, label="Static GCN", color="#94A3B8")
    ax.bar(x + (0 if not s_vals else width / 2), e_vals, width, label="EvolveGCN-H", color="#0D9488")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"Snapshot T{SHUTDOWN_STEP} — Shutdown Event Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "evolvegcn_t43_comparison.png", bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    history: dict,
    evolve_per_snap: list,
    static_per_snap: list | None,
    evolve_pre: float,
    evolve_post: float,
    static_pre: float,
    static_post: float,
    out_dir: Path = FIGURES_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, out_dir)
    plot_per_snapshot_curves(evolve_per_snap, static_per_snap, out_dir)
    plot_pre_post_comparison(evolve_pre, evolve_post, static_pre, static_post, out_dir)
    plot_t43_bar(evolve_per_snap, static_per_snap, out_dir)
