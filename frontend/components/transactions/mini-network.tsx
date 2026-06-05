"use client";

import { useEffect, useState } from "react";
import {
  Background,
  MarkerType,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { getSubgraph } from "@/lib/api";

export function MiniNetwork({ txId }: { txId: string }) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    getSubgraph([txId], 1).then((g) => {
      const flowNodes = g.nodes.slice(0, 20).map((n, i) => {
        const angle = (i / Math.max(g.nodes.length, 1)) * 2 * Math.PI;
        const isSeed = n.is_seed;
        const risk = n.risk_score ?? 0;
        const color =
          risk >= 0.75 || n.prediction === "illicit"
            ? "#ef4444"
            : risk >= 0.5
              ? "#f97316"
              : "#0ea5e9";
        return {
          id: n.id,
          position: { x: 150 + 100 * Math.cos(angle), y: 120 + 80 * Math.sin(angle) },
          data: { label: n.id.slice(0, 6) },
          style: {
            background: color,
            color: "#fff",
            border: isSeed ? "2px solid #fff" : "none",
            borderRadius: 6,
            fontSize: 9,
            width: 48,
            height: 28,
          },
        };
      });
      const flowEdges = g.edges.slice(0, 30).map((e, i) => ({
        id: `e-${i}`,
        source: e.source,
        target: e.target,
        style: { stroke: "#475569", strokeWidth: 1 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#475569" },
      }));
      setNodes(flowNodes);
      setEdges(flowEdges);
    });
  }, [txId, setNodes, setEdges]);

  return (
    <div className="h-64 rounded-lg border border-border overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        nodesDraggable={false}
        nodesConnectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={12} color="#334155" />
      </ReactFlow>
    </div>
  );
}
