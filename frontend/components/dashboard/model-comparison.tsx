"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface ModelComparisonProps {
  metrics: Record<string, any>;
}

export function ModelComparison({ metrics }: ModelComparisonProps) {
  const data = [
    {
      metric: "AUROC",
      Static: metrics.static_gcn?.test_auroc ?? 0.8573,
      Evolve: metrics.evolvegcn_h?.test_auroc ?? 0.7666,
    },
    {
      metric: "F1",
      Static: metrics.static_gcn?.test_f1 ?? 0.4677,
      Evolve: metrics.evolvegcn_h?.test_f1 ?? 0.3269,
    },
    {
      metric: "Precision",
      Static: metrics.static_gcn?.test_precision ?? 0.383,
      Evolve: metrics.evolvegcn_h?.test_precision ?? 0.264,
    },
    {
      metric: "Recall",
      Static: metrics.static_gcn?.test_recall ?? 0.6,
      Evolve: metrics.evolvegcn_h?.test_recall ?? 0.43,
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Performance</CardTitle>
        <CardDescription>Static GCN vs EvolveGCN-H — Elliptic test set</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: 8,
                }}
              />
              <Legend />
              <Bar dataKey="Static" fill="hsl(199 89% 48%)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Evolve" fill="hsl(160 84% 39%)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
