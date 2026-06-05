"""Transaction network subgraph from Elliptic edgelist."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.config import get_settings


@lru_cache(maxsize=1)
def _edge_index() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build adjacency maps: tx_id -> neighbors (outgoing and incoming)."""
    path = get_settings().research_root / "data" / "raw" / "elliptic_txs_edgelist.csv"
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    src_col = cols.get("txid1") or cols.get("src") or df.columns[0]
    dst_col = cols.get("txid2") or cols.get("dst") or df.columns[1]

    out_adj: dict[str, set[str]] = {}
    in_adj: dict[str, set[str]] = {}
    for src, dst in zip(df[src_col].astype(str), df[dst_col].astype(str)):
        out_adj.setdefault(src, set()).add(dst)
        in_adj.setdefault(dst, set()).add(src)
    return out_adj, in_adj


def get_subgraph(
    tx_ids: list[str],
    depth: int = 1,
    max_nodes: int = 80,
    max_edges: int = 200,
) -> dict:
    """Expand seed tx_ids by `depth` hops and return nodes + directed edges."""
    if not tx_ids:
        return {"nodes": [], "edges": [], "truncated": False}

    out_adj, in_adj = _edge_index()
    seeds = {str(t) for t in tx_ids}
    frontier = set(seeds)
    visited = set(seeds)

    for _ in range(max(0, depth)):
        next_frontier: set[str] = set()
        for tid in frontier:
            for nbr in out_adj.get(tid, ()):
                if nbr not in visited:
                    next_frontier.add(nbr)
            for nbr in in_adj.get(tid, ()):
                if nbr not in visited:
                    next_frontier.add(nbr)
            if len(visited) + len(next_frontier) >= max_nodes:
                break
        if len(visited) + len(next_frontier) >= max_nodes:
            next_frontier = set(list(next_frontier)[: max_nodes - len(visited)])
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    node_list = sorted(visited)[:max_nodes]
    node_set = set(node_list)
    edges: list[dict] = []
    for src in node_list:
        for dst in out_adj.get(src, ()):
            if dst in node_set:
                edges.append({"source": src, "target": dst})
                if len(edges) >= max_edges:
                    return {
                        "nodes": [{"id": n} for n in node_list],
                        "edges": edges,
                        "truncated": True,
                    }

    return {
        "nodes": [{"id": n} for n in node_list],
        "edges": edges,
        "truncated": len(visited) >= max_nodes or len(edges) >= max_edges,
    }
