from fastapi import APIRouter, Depends

from app.core.deps import get_current_user
from app.database.models import User
from app.ml.loader import get_artifacts

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def get_metrics(_: User = Depends(get_current_user)):
    art = get_artifacts()
    static_tm = art.static_summary.get("test_metrics", {})
    evolve_tm = art.evolve_summary.get("test_metrics", {})
    return {
        "dataset": art.meta.get("source", "Elliptic Bitcoin"),
        "num_features": art.meta.get("num_features", 165),
        "static_gcn": {
            "test_auroc": static_tm.get("AUROC", 0.8573),
            "test_f1": static_tm.get("F1", 0.4677),
            "test_precision": static_tm.get("Precision"),
            "test_recall": static_tm.get("Recall"),
            "best_val_auroc": art.static_summary.get("best_val_auroc"),
            "architecture": art.static_summary.get("architecture", "165 → 64 → 32 → 2"),
        },
        "evolvegcn_h": {
            "test_auroc": evolve_tm.get("AUROC", 0.7666),
            "test_f1": evolve_tm.get("F1", 0.3269),
            "test_precision": evolve_tm.get("Precision"),
            "test_recall": evolve_tm.get("Recall"),
            "best_val_auroc": art.evolve_summary.get("best_val_auroc"),
            "architecture": art.evolve_summary.get("architecture"),
            "model_version": art.evolve_summary.get("model_version"),
        },
    }
