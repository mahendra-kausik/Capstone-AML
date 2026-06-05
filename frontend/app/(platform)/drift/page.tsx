"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { KpiCard } from "@/components/ui/kpi-card";
import { Skeleton } from "@/components/ui/skeleton";
import { Activity, AlertTriangle, TrendingDown } from "lucide-react";
import { getDrift, getDriftEvents } from "@/lib/api";
import { driftScore, driftSeverity } from "@/lib/dashboard-data";
import type { DriftPayload } from "@/lib/types";

export default function DriftMonitorPage() {
  const [data, setData] = useState<DriftPayload>({});
  const [events, setEvents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getDrift(), getDriftEvents()])
      .then(([d, e]) => { setData(d); setEvents(e); })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Skeleton className="h-96 w-full rounded-xl" />;

  const score = driftScore(data);
  const severity = driftSeverity(score);
  const sevVariant = severity === "Critical" ? "critical" : severity === "High" ? "high" : severity === "Moderate" ? "medium" : "low";

  const combined = (data.kendall_tau?.static ?? []).map((s, i) => ({
    window: s.comparison,
    static: s.tau,
    evolve: data.kendall_tau?.evolve?.[i]?.tau,
  }));

  const t43 = data.t43_drift?.static;
  const f1Timeline = [
    ...(t43?.pre_snapshots?.map((f1, i) => ({ step: 37 + i, f1, phase: "Pre" })) ?? []),
    ...(t43?.post_snapshots?.map((f1, i) => ({ step: (t43?.shutdown_step ?? 43) + i, f1, phase: "Post" })) ?? []),
  ];

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard title="Drift Score (min τ)" value={score.toFixed(3)} icon={Activity} />
        <KpiCard title="Drift Severity" value={severity} icon={AlertTriangle} accent={sevVariant === "critical" || sevVariant === "high" ? "critical" : "default"} />
        <KpiCard title="F1 Drop (T43)" value={t43?.f1_drop != null ? `${(t43.f1_drop * 100).toFixed(1)}%` : "—"} icon={TrendingDown} accent="critical" />
        <KpiCard title="Active Alerts" value={(data.alerts ?? []).length} icon={AlertTriangle} accent="warning" />
      </div>

      <div className="flex gap-2">
        <Badge variant={sevVariant}>Severity: {severity}</Badge>
        <Badge variant="secondary">Kendall τ stability analysis</Badge>
        <Badge variant="secondary">T43 shutdown marker</Badge>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Kendall τ Trends</CardTitle>
            <CardDescription>SHAP rank stability across temporal windows</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={combined}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="window" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
                  <Legend />
                  <ReferenceLine y={0.7} stroke="#64748b" strokeDasharray="4 4" label="τ=0.70" />
                  <Bar dataKey="static" name="Static" fill="hsl(199 89% 48%)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="evolve" name="Evolve" fill="hsl(160 84% 39%)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Concept Drift Timeline</CardTitle>
            <CardDescription>T43 F1 collapse — Static GCN</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={f1Timeline}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="step" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
                  <ReferenceLine x={t43?.shutdown_step ?? 43} stroke="#ef4444" strokeDasharray="4 4" label="T43" />
                  <Line type="monotone" dataKey="f1" stroke="#ef4444" dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Drift Event Log</CardTitle></CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-[10px] uppercase text-muted-foreground">
                <tr>
                  <th className="pb-2 text-left">Time</th>
                  <th className="pb-2 text-left">Model</th>
                  <th className="pb-2 text-left">Window</th>
                  <th className="pb-2 text-left">τ</th>
                  <th className="pb-2 text-left">Alert</th>
                </tr>
              </thead>
              <tbody>
                {events.map((e) => (
                  <tr key={e.id} className="border-t border-border/60">
                    <td className="py-2 text-xs">{new Date(e.detected_at).toLocaleString()}</td>
                    <td className="py-2">{e.model_name}</td>
                    <td className="py-2">{e.window}</td>
                    <td className="py-2 font-mono">{e.metric_value?.toFixed(3)}</td>
                    <td className="py-2">{e.is_alert ? <Badge variant="high">Alert</Badge> : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
