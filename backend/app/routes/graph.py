from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database.models import Prediction, Transaction, User
from app.database.session import get_db
from app.services.graph_service import get_subgraph

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/subgraph")
def subgraph(
    tx_ids: str = Query(..., description="Comma-separated transaction IDs"),
    depth: int = Query(1, ge=0, le=2),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    seeds = [t.strip() for t in tx_ids.split(",") if t.strip()]
    graph = get_subgraph(seeds, depth=depth)

    risk_map: dict[str, float] = {}
    pred_map: dict[str, str] = {}
    if seeds:
        rows = (
            db.query(Transaction.tx_id, Prediction.risk_score, Prediction.prediction)
            .join(Prediction, Prediction.transaction_id == Transaction.id)
            .filter(Transaction.tx_id.in_(seeds + [n["id"] for n in graph["nodes"]]))
            .order_by(Prediction.created_at.desc())
            .all()
        )
        for tx_id, risk, pred in rows:
            if tx_id not in risk_map:
                risk_map[tx_id] = risk
                pred_map[tx_id] = pred

    for node in graph["nodes"]:
        nid = node["id"]
        node["risk_score"] = risk_map.get(nid)
        node["prediction"] = pred_map.get(nid)
        node["is_seed"] = nid in seeds

    return graph
