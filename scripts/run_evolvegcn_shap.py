#!/usr/bin/env python3
"""Run EvolveGCN-H KernelSHAP drift analysis (4 windows + Kendall τ)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evolvegcn.shap_analysis import (
    TOP_K_TAU,
    WINDOWS,
    run_evolvegcn_shap_pipeline,
)
from src.evolvegcn.training import get_device


def print_tau_table(static_tau, evolve_tau):
    static_map = {r["comparison"]: r["tau"] for r in static_tau}
    evolve_map = {r["comparison"]: r["tau"] for r in evolve_tau}
    print("\n" + "=" * 60)
    print("KENDALL τ COMPARISON (top-{} features per window)".format(TOP_K_TAU))
    print("=" * 60)
    print(f"{'Transition':<12} {'Static GCN':>12} {'EvolveGCN-H':>14}")
    print("-" * 42)
    for comp in ["W1→W2", "W2→W3", "W3→W4"]:
        s = static_map.get(comp, float("nan"))
        e = evolve_map.get(comp, float("nan"))
        print(f"{comp:<12} {s:>12.4f} {e:>14.4f}")

    static_w34 = static_map.get("W3→W4", 0.147)
    evolve_w34 = evolve_map.get("W3→W4", float("nan"))
    print("\n--- W3→W4 vs Static baseline (0.147) ---")
    if evolve_w34 > static_w34:
        delta = evolve_w34 - static_w34
        pct = 100 * delta / static_w34 if static_w34 else float("inf")
        print(
            f"EvolveGCN-H W3→W4 τ = {evolve_w34:.4f} > Static {static_w34:.4f} "
            f"(+{delta:.4f}, +{pct:.1f}% relative improvement)"
        )
    else:
        print(
            f"EvolveGCN-H W3→W4 τ = {evolve_w34:.4f} ≤ Static {static_w34:.4f}. "
            "Post-T43 feature rankings still shift under both models; "
            "EvolveGCN temporal weight evolution does not preserve late-window "
            "SHAP ranks more than the static snapshot model."
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Recompute even if pickles exist")
    p.add_argument("--device", default=None, help="cpu | cuda | mps")
    args = p.parse_args()

    device = get_device() if args.device is None else __import__("torch").device(args.device)

    print("EvolveGCN-H SHAP pipeline")
    print(f"  device: {device}")
    print(f"  windows: {list(WINDOWS.keys())}")

    out = run_evolvegcn_shap_pipeline(device=device, force=args.force)
    print_tau_table(out["static_tau"], out["tau_results"])

    print("\nREADY FOR EVOLVEGCN SHAP ANALYSIS — artifacts written under data/shap/ and figures/")


if __name__ == "__main__":
    main()
