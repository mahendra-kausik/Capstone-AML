"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { Background, Controls, MiniMap, MarkerType, ReactFlow, useEdgesState, useNodesState, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { getSubgraph } from "@/lib/api";

export default function NetworkPage() {
  return (
    <Suspense fallback={<Skeleton className="h-[600px] w-full" />}>
      <NetworkContent />
    </Suspense>
  );
}

function NetworkContent() {
  const params = useSearchParams();
  const [txInput, setTxInput] = useState(params.get("tx_ids") || "");
  const [depth, setDepth] = useState(1);
  const [truncated, setTruncated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const loadGraph = useCallback(async () => {
    const seeds = txInput.split(/[\s,]+/).filter(Boolean);
    if (!seeds.length) return;
    setLoading(true);
    try {
      const g = await getSubgraph(seeds, depth);
      setTruncated(g.truncated);
      setNodes(
        g.nodes.map((n, i) => {
          const angle = (i / Math.max(g.nodes.length, 1)) * 2 * Math.PI;
          const risk = n.risk_score ?? 0;
          const color =
            n.prediction === "illicit" || risk >= 0.75
              ? "#ef4444"
              : risk >= 0.4
                ? "#f97316"
                : "#0ea5e9";
          return {
            id: n.id,
            position: { x: 400 + 200 * Math.cos(angle), y: 300 + 150 * Math.sin(angle) },
            data: { label: `${n.id.slice(0, 8)}…` },
            style: {
              background: color,
              color: "#fff",
              border: n.is_seed ? "3px solid #f8fafc" : "1px solid #334155",
              borderRadius: 8,
              fontSize: 10,
              width: 80,
              padding: 6,
            },
          };
        })
      );
      setEdges(
        g.edges.map((e, i) => ({
          id: `e-${i}`,
          source: e.source,
          target: e.target,
          animated: !!g.nodes.find((n) => n.id === e.source)?.is_seed,
          style: { stroke: "#64748b" },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#64748b" },
        }))
      );
    } finally {
      setLoading(false);
    }
  }, [txInput, depth, setNodes, setEdges]);

  useEffect(() => {
    if (params.get("tx_ids")) loadGraph();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3 glass-card p-4">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input className="pl-9 bg-secondary/50" placeholder="Seed transaction IDs" value={txInput} onChange={(e) => setTxInput(e.target.value)} />
        </div>
        <select value={depth} onChange={(e) => setDepth(Number(e.target.value))} className="h-10 rounded-lg border border-border bg-background px-3 text-sm">
          <option value={0}>Seeds only</option>
          <option value={1}>1 hop</option>
          <option value={2}>2 hops</option>
        </select>
        <Button onClick={loadGraph} disabled={loading}>{loading ? "Loading…" : "Trace network"}</Button>
      </div>

      {truncated && <Badge variant="medium">Graph truncated for performance</Badge>}

      <div className="h-[560px] rounded-xl border border-border overflow-hidden bg-card/30">
        {loading ? (
          <Skeleton className="h-full w-full" />
        ) : nodes.length > 0 ? (
          <ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} fitView minZoom={0.15}>
            <Background gap={20} color="#334155" />
            <Controls className="!bg-card !border-border" />
            <MiniMap nodeStrokeWidth={2} className="!bg-card" />
          </ReactFlow>
        ) : (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            Enter seed TX IDs to visualize illicit flow networks
          </div>
        )}
      </div>

      <div className="flex gap-6 text-xs text-muted-foreground">
        <span className="flex items-center gap-2"><span className="h-3 w-3 rounded bg-[#ef4444]" /> Suspicious</span>
        <span className="flex items-center gap-2"><span className="h-3 w-3 rounded bg-[#0ea5e9]" /> Normal</span>
        <span>White border = seed node</span>
      </div>
    </div>
  );
}
