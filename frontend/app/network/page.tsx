"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  MiniMap,
  MarkerType,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { AuthGuard } from "@/components/auth-guard";
import { Alert } from "@/components/ui/alert";
import { Loading } from "@/components/ui/loading";
import { PageHeader } from "@/components/ui/page-header";
import { getSubgraph } from "@/lib/api";

function riskColor(score?: number, prediction?: string) {
  if (prediction === "illicit" || (score != null && score >= 0.75)) return "#dc2626";
  if (score != null && score >= 0.5) return "#f97316";
  if (score != null && score >= 0.25) return "#eab308";
  return "#059669";
}

export default function NetworkPage() {
  return (
    <AuthGuard>
      <Suspense fallback={<Loading />}>
        <NetworkContent />
      </Suspense>
    </AuthGuard>
  );
}

function NetworkContent() {
  const params = useSearchParams();
  const initialTx = params.get("tx_ids") || "";

  const [txInput, setTxInput] = useState(initialTx);
  const [depth, setDepth] = useState(1);
  const [truncated, setTruncated] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const loadGraph = useCallback(async () => {
    const seeds = txInput.split(/[\s,]+/).filter(Boolean);
    if (!seeds.length) {
      setError("Enter at least one transaction ID.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const g = await getSubgraph(seeds, depth);
      setTruncated(g.truncated);

      const flowNodes: Node[] = g.nodes.map((n, i) => {
        const col = riskColor(n.risk_score, n.prediction);
        const angle = (i / Math.max(g.nodes.length, 1)) * 2 * Math.PI;
        const r = 180 + (n.is_seed ? 0 : 60);
        return {
          id: n.id,
          position: {
            x: 400 + r * Math.cos(angle),
            y: 300 + r * Math.sin(angle),
          },
          data: {
            label: (
              <div className="text-center text-[10px] leading-tight">
                <div className="font-mono font-bold">{n.id.slice(0, 8)}…</div>
                {n.risk_score != null && (
                  <div>{(n.risk_score * 100).toFixed(0)}%</div>
                )}
              </div>
            ),
          },
          style: {
            background: col,
            color: "#fff",
            border: n.is_seed ? "3px solid #1e3a5f" : "1px solid #fff",
            borderRadius: 8,
            width: 72,
            fontSize: 10,
            padding: 4,
          },
        };
      });

      const flowEdges: Edge[] = g.edges.map((e, i) => ({
        id: `e-${i}`,
        source: e.source,
        target: e.target,
        animated: g.nodes.find((n) => n.id === e.source)?.is_seed,
        style: { stroke: "#94a3b8", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#94a3b8" },
      }));

      setNodes(flowNodes);
      setEdges(flowEdges);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Graph load failed");
    } finally {
      setLoading(false);
    }
  }, [txInput, depth, setNodes, setEdges]);

  useEffect(() => {
    if (initialTx) loadGraph();
  }, [initialTx]); // eslint-disable-line react-hooks/exhaustive-deps

  const legend = useMemo(
    () => [
      { color: "#dc2626", label: "High risk / illicit" },
      { color: "#f97316", label: "Elevated (50–75%)" },
      { color: "#eab308", label: "Medium (25–50%)" },
      { color: "#059669", label: "Low risk / licit" },
    ],
    []
  );

  return (
    <div>
      <PageHeader
        title="Transaction Network"
        description="Elliptic edgelist subgraph — 1-hop neighbors from seed transactions."
      />

      <div className="mt-6 flex flex-wrap items-end gap-3 rounded-xl border bg-white p-4 shadow-sm">
        <div className="flex-1 min-w-[200px]">
          <label className="text-sm font-medium">Seed transaction IDs</label>
          <input
            className="mt-1 w-full rounded-lg border px-3 py-2 font-mono text-sm"
            placeholder="114641619, 30179316"
            value={txInput}
            onChange={(e) => setTxInput(e.target.value)}
          />
        </div>
        <div>
          <label className="text-sm font-medium">Depth</label>
          <select
            value={depth}
            onChange={(e) => setDepth(Number(e.target.value))}
            className="mt-1 block rounded-lg border px-3 py-2 text-sm"
          >
            <option value={0}>0 (seeds only)</option>
            <option value={1}>1 hop</option>
            <option value={2}>2 hops</option>
          </select>
        </div>
        <button
          onClick={loadGraph}
          disabled={loading}
          className="rounded-lg bg-brand px-5 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading ? "Loading…" : "Load subgraph"}
        </button>
      </div>

      {error && <div className="mt-4"><Alert variant="error">{error}</Alert></div>}
      {truncated && (
        <div className="mt-4"><Alert variant="info">Graph truncated for performance (max nodes/edges).</Alert></div>
      )}

      <div className="mt-6 h-[520px] rounded-xl border bg-white shadow-sm">
        {loading ? (
          <Loading label="Building network graph…" />
        ) : nodes.length > 0 ? (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            minZoom={0.2}
          >
            <Background gap={16} />
            <Controls />
            <MiniMap nodeStrokeWidth={2} zoomable pannable />
          </ReactFlow>
        ) : (
          <div className="flex h-full items-center justify-center text-slate-500">
            Enter seed TX IDs from upload or case history to visualize flows.
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap gap-4 text-sm">
        {legend.map((l) => (
          <div key={l.label} className="flex items-center gap-2">
            <span className="h-3 w-3 rounded" style={{ background: l.color }} />
            {l.label}
          </div>
        ))}
        <span className="text-slate-400">Bold border = seed node</span>
      </div>
    </div>
  );
}
